import os
import numpy as np
import tensorflow as tf
import random
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from itertools import combinations, permutations
# import tensorflow.keras.backend as K
import pywt


# Configuration
class Config:
    # Data settings
    DATA_DIR = "data/cc_1"
    SAMPLE_RATE = 16000
    SEGMENT_LENGTH = 16000  # 1 second at 16kHz
    MAX_SOURCES = 4
    
    # Model settings
    NUM_COEFFS = 16000  # 1 second at 16kHz
    WAVELET_DEPTH = 5
    BATCH_SIZE = 8
    CHANNELS = 1  # Mono audio
    NUM_LAYERS = 5
    NUM_INIT_FILTERS = 16 ## was 24
    FILTER_SIZE = 8 # was 16
    MERGE_FILTER_SIZE = 4 # was 5
    L1_REG = 1e-5
    L2_REG = 1e-5
    
    # Training settings
    LEARNING_RATE = 1e-4
    EPOCHS = 50
    VAL_SPLIT = 0.1
    CHECKPOINT_DIR = "checkpoints"
    MAX_WORKERS = 4
    
    # Mixture generation
    MIN_SOURCES = 2
    MAX_SOURCES = 4
    NUM_EXAMPLES = 31000
    
    # Wavelet settings
    WAVELET_FAMILY = 'db4'  # Daubechies wavelet with 4 vanishing moments

config = Config()


# GELU Activation Function
def gelu(x):
    """Gaussian Error Linear Unit activation function"""
    return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))


# Data Processing Functions
def load_and_preprocess_audio_files(data_dir):
    """Load all audio files and preprocess them for training."""
    print("Loading and preprocessing audio files...")
    
    # Find all audio files
    audio_files = glob(os.path.join(data_dir, "**/*.wav"), recursive=True)
    if not audio_files:
        raise ValueError(f"No audio files found in {data_dir}")
    
    print(f"Found {len(audio_files)} audio files")
    
    # Create a directory for preprocessed segments
    # os.makedirs("/content/preprocessed", exist_ok=True)
    
    # Process each file into segments
    segments = []
    for file_path in tqdm(audio_files):
        try:
            # Load audio
            audio, _ = tf.audio.decode_wav(
                tf.io.read_file(file_path), 
                desired_channels=config.CHANNELS
            )
            
            # Convert to numpy for easier handling
            audio = audio.numpy().reshape(-1)
            
            # Split into segments
            num_segments = len(audio) // config.SEGMENT_LENGTH
            for i in range(num_segments):
                start_idx = i * config.SEGMENT_LENGTH
                end_idx = start_idx + config.SEGMENT_LENGTH
                segment = audio[start_idx:end_idx]
                
                # Normalize segment to [-1, 1]
                if np.max(np.abs(segment)) > 0:
                    segment = segment / np.max(np.abs(segment))
                
                # Save segment
                segment_path = f"content/preprocessed/segment_{len(segments)}.npy"
                np.save(segment_path, segment)
                segments.append(segment_path)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    print(f"Created {len(segments)} segments")
    return segments

def create_source_combinations(segments, num_examples=31000):
    """Create a balanced set of source combinations for mixtures."""
    print("Creating source combinations...")
    
    num_segments = len(segments)
    combinations_per_count = num_examples // 3  # Even split between 2, 3, and 4 sources
    
    all_combinations = []
    
    # Generate combinations for each source count
    for source_count in range(config.MIN_SOURCES, config.MAX_SOURCES + 1):
        print(f"Generating {combinations_per_count} combinations with {source_count} sources")
        
        if source_count > num_segments:
            raise ValueError(f"Not enough segments ({num_segments}) to create combinations of {source_count} sources")
        
        # Calculate how many unique combinations are possible
        max_possible = min(combinations_per_count, 
                           int(np.math.comb(num_segments, source_count)))
        
        # Randomly sample combinations without replacement if possible
        if max_possible < combinations_per_count:
            print(f"Warning: Only {max_possible} unique combinations possible for {source_count} sources")
            
            # Generate all possible combinations indices
            indices = list(range(num_segments))
            combinations_list = []
            
            # Create combinations until we reach the target or exhaust possibilities
            count = 0
            while count < max_possible:
                # Randomly sample source_count indices
                combo = sorted(random.sample(indices, source_count))
                combo_key = tuple(combo)
                
                # Only add if this exact combination hasn't been seen before
                if combo_key not in combinations_list:
                    combinations_list.append(combo_key)
                    all_combinations.append([segments[i] for i in combo])
                    count += 1
        else:
            # When we have enough segments, directly sample random combinations
            for _ in range(combinations_per_count):
                source_indices = random.sample(range(num_segments), source_count)
                all_combinations.append([segments[i] for i in source_indices])
    
    print(f"Created {len(all_combinations)} source combinations")
    return all_combinations

def zero_pad_sources(sources, max_sources):
    """Zero pad sources to have the same number of sources."""
    num_sources = len(sources)
    if num_sources < max_sources:
        # Create zero arrays for padding
        zero_pad = np.zeros((max_sources - num_sources, config.SEGMENT_LENGTH), dtype=np.float32)
        return np.vstack([sources, zero_pad])
    return sources


# Wavelet Transform Functions
@tf.keras.utils.register_keras_serializable()
class DaubechiesWaveletLayer(tf.keras.layers.Layer):
    def __init__(self, wavelet_family='db4', mode='periodization', name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.wavelet_family = wavelet_family
        self.mode = mode
        
        # Get Daubechies wavelet filter coefficients
        wavelet = pywt.Wavelet(wavelet_family)
        
        self.dec_lo = tf.constant(wavelet.dec_lo, dtype=tf.float32)
        self.dec_hi = tf.constant(wavelet.dec_hi, dtype=tf.float32)
        self.rec_lo = tf.constant(wavelet.rec_lo, dtype=tf.float32)
        self.rec_hi = tf.constant(wavelet.rec_hi, dtype=tf.float32)
    
    def build(self, input_shape):
        wavelet = pywt.Wavelet(self.wavelet_family)
        self.filter_length = self.dec_lo.shape[0]
        self.channels = input_shape[-1]
        
        # Create trainable filters (initialized with Daubechies coefficients)
        self.dec_lo_filter = self.add_weight(
            name='dec_lo',
            shape=(self.filter_length, 1, 1),
            
            initializer=tf.constant_initializer(wavelet.dec_lo),
            trainable=False  # Set to True for learnable wavelets
        )
        
        self.dec_hi_filter = self.add_weight(
            name='dec_hi',
            shape=(self.filter_length, 1, 1),
            initializer=tf.constant_initializer(wavelet.dec_hi),
            trainable=False  # Set to True for learnable wavelets
        )
        
        super().build(input_shape)
    
    def call(self, inputs):
        # Handle padding for odd length inputs
        orig_shape = tf.shape(inputs)
        need_padding = orig_shape[1] % 2 != 0
        
        # Add reflection padding for clean convolution
        # if orig_shape[1] % 2 != 0:
        #     inputs = tf.pad(inputs, [[0, 0], [0, 1], [0, 0]], mode='REFLECT')
        
        # Add boundary padding based on wavelet filter length
        pad_size = self.filter_length - 1
        padded_inputs = tf.pad(inputs, [[0, 0], [pad_size, pad_size], [0, 0]], mode='REFLECT')
        
        # Split channels and apply filtering to each
        batch_size = tf.shape(inputs)[0]
        approx_coeffs = []
        detail_coeffs = []
        
        for c in range(self.channels):
            channel_inputs = padded_inputs[:, :, c:c+1]
            
            # Apply low-pass filter (for approximation coefficients)
            approx = tf.nn.conv1d(
                channel_inputs,
                self.dec_lo_filter,
                stride=2,
                padding='VALID'
            )
            
            # Apply high-pass filter (for detail coefficients)
            detail = tf.nn.conv1d(
                channel_inputs,
                self.dec_hi_filter,
                stride=2,
                padding='VALID'
            )
            
            # Remove excess padding
            approx = approx[:, (pad_size // 2):-(pad_size // 2) if pad_size > 1 else None, :]
            detail = detail[:, (pad_size // 2):-(pad_size // 2) if pad_size > 1 else None, :]
            
            approx_coeffs.append(approx)
            detail_coeffs.append(detail)
        
        # Concatenate all channels
        if self.channels > 1:
            approx_coeffs = tf.concat(approx_coeffs, axis=-1)
            detail_coeffs = tf.concat(detail_coeffs, axis=-1)
        else:
            approx_coeffs = approx_coeffs[0]
            detail_coeffs = detail_coeffs[0]
        
        # Concatenate approximation and detail coefficients along the channel axis
        return tf.concat([approx_coeffs, detail_coeffs], axis=-1)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'wavelet_family': self.wavelet_family,
            'mode': self.mode
        })
        return config


@tf.keras.utils.register_keras_serializable()
class InverseDaubechiesWaveletLayer(tf.keras.layers.Layer):
    def __init__(self, wavelet_family='db4',batch_size=8, mode='periodization', name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.wavelet_family = wavelet_family
        self.mode = mode
        self.batch_size = batch_size   
        
        # Get Daubechies wavelet filter coefficients
        wavelet = pywt.Wavelet(wavelet_family)
        self.rec_lo = tf.constant(wavelet.rec_lo, dtype=tf.float32)
        self.rec_hi = tf.constant(wavelet.rec_hi, dtype=tf.float32)
    
    def build(self, input_shape):
        self.filter_length = self.rec_lo.shape[0]
        self.in_channels = input_shape[-1] // 2  # Half for approx, half for detail
        wavelet = pywt.Wavelet(self.wavelet_family)
        # Create trainable filters (initialized with Daubechies coefficients)
        self.rec_lo_filter = self.add_weight(
            name='rec_lo',
            shape=(self.filter_length, 1, 1),
            initializer=tf.constant_initializer(wavelet.rec_lo),
            trainable=False  # Set to True for learnable wavelets
        )
        
        self.rec_hi_filter = self.add_weight(
            name='rec_hi',
            shape=(self.filter_length, 1, 1),
            initializer=tf.constant_initializer(wavelet.rec_hi),
            trainable=False  # Set to True for learnable wavelets
        )
        
        super().build(input_shape)
    
    def call(self, inputs):
        # Split the channels into approximation and detail coefficients
        approx_coeffs = inputs[:, :, :self.in_channels]
        detail_coeffs = inputs[:, :, self.in_channels:]
        
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        output_channels = []
        
        for c in range(self.in_channels):
            # Get coefficients for this channel
            approx = approx_coeffs[:, :, c:c+1]
            detail = detail_coeffs[:, :, c:c+1]
            
            # Upsample (insert zeros)
            approx_up = self._upsample(approx)
            detail_up = self._upsample(detail)
            
            # Apply reconstruction filters
            approx_recon = tf.nn.conv1d(
                approx_up,
                self.rec_lo_filter,
                stride=1,
                padding='SAME'
            )
            
            detail_recon = tf.nn.conv1d(
                detail_up,
                self.rec_hi_filter,
                stride=1,
                padding='SAME'
            )
            
            # Combine approximation and detail for reconstruction
            recon = approx_recon + detail_recon
            output_channels.append(recon)
        
        # Concatenate all channels
        if self.in_channels > 1:
            output = tf.concat(output_channels, axis=-1)
        else:
            output = output_channels[0]
        
        return output
    
    def _upsample(self, x):
        """Upsample by inserting zeros between samples (for inverse DWT)"""
        batch_size = self.batch_size
        seq_len = tf.shape(x)[1]
        channels = tf.shape(x)[2]
        
        # Create upsampled tensor with zeros
        output = tf.zeros([batch_size, seq_len*2, channels], dtype=x.dtype)
        
        # Insert original values at even indices
        indices = tf.range(0, seq_len*2, 2)
        updates = tf.reshape(x, [batch_size, seq_len, channels])
        
        # Use tensor_scatter_nd_update
        indices_tensor = tf.expand_dims(indices, axis=1)
        
        results = []
        for b in range(batch_size):
            batch_result = tf.tensor_scatter_nd_update(
                output[b],
                indices_tensor,
                updates[b]
            )
            results.append(batch_result)
        
        return tf.stack(results, axis=0)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'wavelet_family': self.wavelet_family,
            'mode': self.mode
        })
        return config


@tf.keras.utils.register_keras_serializable()
class EnhancedDownsamplingLayer(tf.keras.layers.Layer):
    def __init__(self, num_filters, filter_size, l1_reg=0.0, l2_reg=0.0, **kwargs):
        super().__init__(**kwargs)
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg

    def build(self, input_shape):
        # Input projection if needed for residual connection
        self.input_channels = input_shape[-1]
        if self.input_channels != self.num_filters:
            self.input_proj = tf.keras.layers.Conv1D(
                self.num_filters,
                1,
                padding='same',
                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg),
                name=f'input_proj_{self.name}'
            )
        else:
            self.input_proj = None
            
        # Main convolution
        self.conv = tf.keras.layers.Conv1D(
            self.num_filters,
            self.filter_size,
            padding='same',
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg),
            name=f'ds_conv_{self.name}'
        )
        
        # Layer normalization
        self.layer_norm = tf.keras.layers.LayerNormalization(name=f'layer_norm_{self.name}')
        
        super().build(input_shape)

    def call(self, inputs):
        # Residual connection
        if self.input_proj is not None:
            residual = self.input_proj(inputs)
        else:
            residual = inputs
            
        # Main path
        x = self.conv(inputs)
        x = self.layer_norm(x)
        x = gelu(x)
        
        # Add residual
        return x + residual
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_filters': self.num_filters,
            'filter_size': self.filter_size,
            'l1_reg': self.l1_reg,
            'l2_reg': self.l2_reg
        })
        return config


@tf.keras.utils.register_keras_serializable()
class EnhancedUpsamplingLayer(tf.keras.layers.Layer):
    def __init__(self, num_filters, filter_size, l1_reg=0.0, l2_reg=0.0, **kwargs):
        super().__init__(**kwargs)
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg

    def build(self, input_shape):
        # Input projection for residual
        self.input_channels = input_shape[-1]
        if self.input_channels != self.num_filters:
            self.input_proj = tf.keras.layers.Conv1D(
                self.num_filters,
                1,
                padding='same',
                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg),
                name=f'input_proj_{self.name}'
            )
        else:
            self.input_proj = None
        
        # Convolutional layers for separate processing of approximation and detail coefficients
        half_channels = self.input_channels // 2
        self.approx_conv = tf.keras.layers.Conv1D(
            half_channels,
            self.filter_size,
            padding='same',
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg),
            name=f'approx_conv_{self.name}'
        )
        
        self.detail_conv = tf.keras.layers.Conv1D(
            half_channels,
            self.filter_size,
            padding='same',
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg),
            name=f'detail_conv_{self.name}'
        )
        
        # Gating mechanisms
        self.approx_gate = tf.keras.layers.Conv1D(
            half_channels,
            1,
            activation='sigmoid',
            padding='same',
            name=f'approx_gate_{self.name}'
        )
        
        self.detail_gate = tf.keras.layers.Conv1D(
            half_channels,
            1,
            activation='sigmoid',
            padding='same',
            name=f'detail_gate_{self.name}'
        )
        
        # Layer normalization
        self.approx_norm = tf.keras.layers.LayerNormalization(name=f'approx_norm_{self.name}')
        self.detail_norm = tf.keras.layers.LayerNormalization(name=f'detail_norm_{self.name}')
        
        # Final convolution after recombination
        self.output_conv = tf.keras.layers.Conv1D(
            self.num_filters,
            1,
            padding='same',
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg),
            name=f'output_conv_{self.name}'
        )
        
        self.output_norm = tf.keras.layers.LayerNormalization(name=f'output_norm_{self.name}')
        
        super().build(input_shape)

    def call(self, inputs):
        # Split into approximation and detail coefficients
        half_channels = self.input_channels // 2
        approx_coeff = inputs[:, :, :half_channels]
        detail_coeff = inputs[:, :, half_channels:]
        
        # Process each separately
        approx_processed = self.approx_conv(approx_coeff)
        approx_processed = self.approx_norm(approx_processed)
        approx_processed = gelu(approx_processed)
        
        detail_processed = self.detail_conv(detail_coeff)
        detail_processed = self.detail_norm(detail_processed)
        detail_processed = gelu(detail_processed)
        
        # Apply gating
        approx_gate = self.approx_gate(approx_processed)
        detail_gate = self.detail_gate(detail_processed)
        
        approx_gated = approx_processed * approx_gate
        detail_gated = detail_processed * detail_gate
        
        # Recombine
        combined = tf.concat([approx_gated, detail_gated], axis=-1)
        
        # Residual connection
        if self.input_proj is not None:
            residual = self.input_proj(inputs)
        else:
            residual = inputs
        
        # Final processing
        output = self.output_conv(combined)
        output = self.output_norm(output)
        output = gelu(output)
        
        return output + residual
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_filters': self.num_filters,
            'filter_size': self.filter_size,
            'l1_reg': self.l1_reg,
            'l2_reg': self.l2_reg
        })
        return config


# Enhanced Skip Connection
@tf.keras.utils.register_keras_serializable()
class EnhancedSkipConnection(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def build(self, input_shape):
        # Extract shapes
        decoder_shape, encoder_shape = input_shape
        
        # Check if the channel dimensions match
        self.decoder_channels = decoder_shape[-1]
        self.encoder_channels = encoder_shape[-1]
        
        # Gate for decoder features
        self.decoder_gate = tf.keras.layers.Conv1D(
            self.decoder_channels,
            1,
            activation='sigmoid',
            padding='same',
            name=f'decoder_gate_{self.name}'
        )
        
        # Gate for encoder features
        self.encoder_gate = tf.keras.layers.Conv1D(
            self.encoder_channels,
            1,
            activation='sigmoid',
            padding='same',
            name=f'encoder_gate_{self.name}'
        )
        
        # Layer normalization
        self.norm = tf.keras.layers.LayerNormalization(name=f'skip_norm_{self.name}')
        
        super().build(input_shape)
        
    def call(self, inputs):
        # Unpack inputs
        decoder_features, encoder_features = inputs
        
        # Apply gates
        decoder_gated = decoder_features * self.decoder_gate(decoder_features)
        encoder_gated = encoder_features * self.encoder_gate(encoder_features)
        
        # Concatenate along channel dimension
        concat = tf.concat([decoder_gated, encoder_gated], axis=-1)
        
        # Apply normalization
        return self.norm(concat)
    
    def get_config(self):
        config = super().get_config()
        return config


# Improved Wavelet U-Net Model
@tf.keras.utils.register_keras_serializable()
class ImprovedWaveletUNet(tf.keras.Model):
    def __init__(self, num_coeffs, wavelet_depth, batch_size, channels, num_layers, 
                 num_init_filters, filter_size, merge_filter_size, l1_reg, l2_reg,
                 max_sources=4, wavelet_family='db4', **kwargs):
        super().__init__(**kwargs)
        self.num_coeffs = num_coeffs
        self.wavelet_depth = wavelet_depth + 1
        self.batch_size = batch_size
        self.channels = channels
        self.num_layers = num_layers
        self.num_init_filters = num_init_filters
        self.filter_size = filter_size
        self.merge_filter_size = merge_filter_size
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.max_sources = max_sources
        self.wavelet_family = wavelet_family

    @classmethod
    def from_config(cls, config):
        # Extract the necessary arguments from the config dictionary
        num_coeffs = config.pop('num_coeffs')
        wavelet_depth = config.pop('wavelet_depth')
        batch_size = config.pop('batch_size')
        channels = config.pop('channels')
        num_layers = config.pop('num_layers')
        num_init_filters = config.pop('num_init_filters')
        filter_size = config.pop('filter_size')
        merge_filter_size = config.pop('merge_filter_size')
        l1_reg = config.pop('l1_reg')
        l2_reg = config.pop('l2_reg')
        max_sources = config.pop('max_sources', 4)
        wavelet_family = config.pop('wavelet_family', 'db4')

        return cls(
            num_coeffs=num_coeffs,
            wavelet_depth=wavelet_depth,
            batch_size=batch_size,
            channels=channels,
            num_layers=num_layers,
            num_init_filters=num_init_filters,
            filter_size=filter_size,
            merge_filter_size=merge_filter_size,
            l1_reg=l1_reg,
            l2_reg=l2_reg,
            max_sources=max_sources,
            wavelet_family=wavelet_family,
            **config
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_coeffs': self.num_coeffs,
            'wavelet_depth': self.wavelet_depth,
            'batch_size': self.batch_size,
            'channels': self.channels,
            'num_layers': self.num_layers,
            'num_init_filters': self.num_init_filters,
            'filter_size': self.filter_size,
            'merge_filter_size': self.merge_filter_size,
            'l1_reg': self.l1_reg,
            'l2_reg': self.l2_reg,
            'max_sources': self.max_sources,
            'wavelet_family': self.wavelet_family
        })
        return config

    def build(self, input_shape):
        # Initial convolution
        self.initial_conv = tf.keras.layers.Conv1D(
            self.num_init_filters,
            self.filter_size,
            padding='same',
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg),
            name='initial_conv'
        )
        self.initial_norm = tf.keras.layers.LayerNormalization(name='initial_norm')
        
        # Create enhanced downsampling blocks
        self.downsampling_blocks = {}
        self.dwt_layers = {}
        self.down_process_blocks = {}
        
        for i in range(self.num_layers):
            block_name = f'{i+1}'
            num_filters = self.num_init_filters + (self.num_init_filters * i)
            
            # Main downsampling block
            self.downsampling_blocks[block_name] = EnhancedDownsamplingLayer(
                num_filters, 
                self.filter_size, 
                l1_reg=self.l1_reg, 
                l2_reg=self.l2_reg,
                name=f'ds_block_{block_name}'
            )
            
            # DWT layer
            self.dwt_layers[block_name] = DaubechiesWaveletLayer(
                wavelet_family=self.wavelet_family,
                name=f'dwt_{block_name}'
            )
            
            # Post-DWT processing
            self.down_process_blocks[block_name] = EnhancedUpsamplingLayer(
                num_filters * 2,  # Double channels after DWT
                self.filter_size,
                l1_reg=self.l1_reg,
                l2_reg=self.l2_reg,
                name=f'down_process_{block_name}'
            )

        # Create bottle neck
        self.bottle_neck = tf.keras.Sequential([
            tf.keras.layers.Conv1D(
                self.num_init_filters * (self.num_layers + 1),
                self.filter_size,
                padding='same',
                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg),
                name='bottleneck_conv1'
            ),
            tf.keras.layers.LayerNormalization(name='bottleneck_norm2'),
            tf.keras.layers.Lambda(gelu)
        ])

        # Create upsampling blocks
        self.idwt_layers = {}
        self.up_process_blocks = {}
        self.skip_connections = {}
        self.upsampling_blocks = {}
        
        for i in range(self.num_layers):
            block_name = f'{self.num_layers - i}'
            num_filters = self.num_init_filters + (self.num_init_filters * (self.num_layers - i - 1))
            
            # Inverse DWT layer
            self.idwt_layers[block_name] = InverseDaubechiesWaveletLayer(
                wavelet_family=self.wavelet_family,
                name=f'idwt_{block_name}'
            )
            
            # Pre-skip connection processing
            self.up_process_blocks[block_name] = EnhancedUpsamplingLayer(
                num_filters,
                self.filter_size,
                l1_reg=self.l1_reg,
                l2_reg=self.l2_reg,
                name=f'up_process_{block_name}'
            )
            
            # Enhanced skip connection
            self.skip_connections[block_name] = EnhancedSkipConnection(
                name=f'skip_connection_{block_name}'
            )
            
            # Post-skip connection processing
            self.upsampling_blocks[block_name] = tf.keras.Sequential([
                tf.keras.layers.Conv1D(
                    num_filters,
                    self.merge_filter_size,
                    padding='same',
                    kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg),
                    name=f'up_conv1_{block_name}'
                ),
                tf.keras.layers.LayerNormalization(name=f'up_norm1_{block_name}'),
                tf.keras.layers.Lambda(gelu),
                tf.keras.layers.Conv1D(
                    num_filters,
                    self.merge_filter_size,
                    padding='same',
                    kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg),
                    name=f'up_conv2_{block_name}'
                ),
                tf.keras.layers.LayerNormalization(name=f'up_norm2_{block_name}'),
                tf.keras.layers.Lambda(gelu)
            ])

        # Final processing
        self.final_conv = tf.keras.Sequential([
            tf.keras.layers.Conv1D(
                self.num_init_filters,
                self.filter_size,
                padding='same',
                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg),
                name='final_conv1'
            ),
            tf.keras.layers.LayerNormalization(name='final_norm1'),
            tf.keras.layers.Lambda(gelu),
            tf.keras.layers.Conv1D(
                self.num_init_filters,
                self.filter_size,
                padding='same',
                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg),
                name='final_conv2'
            ),
            tf.keras.layers.LayerNormalization(name='final_norm2'),
            tf.keras.layers.Lambda(gelu)
        ])
        
        # Output layer for each source
        self.output_convs = []
        for i in range(self.max_sources):
            self.output_convs.append(
                tf.keras.layers.Conv1D(
                    1,
                    1,
                    activation='tanh',
                    padding='same',
                    name=f'output_conv_{i}'
                )
            )
            
        super().build(input_shape)
        self.summary()

    def call(self, inputs, training=True):
        # Initial processing
        current_layer = self.initial_conv(inputs)
        current_layer = self.initial_norm(current_layer)
        current_layer = gelu(current_layer)
        
        # Store the input for skip connection to final layer
        full_mix = tf.reduce_sum(inputs, axis=-1, keepdims=True)
        
        # Store encoder outputs for skip connections
        enc_outputs = {}

        # Downsampling path
        for i in range(self.num_layers):
            block_name = f'{i+1}'
            
            # Apply enhanced downsampling
            current_layer = self.downsampling_blocks[block_name](current_layer)
            
            # Save for skip connections
            enc_outputs[block_name] = current_layer
            
            # Apply DWT
            current_layer = self.dwt_layers[block_name](current_layer)
            
            # Post-DWT processing
            current_layer = self.down_process_blocks[block_name](current_layer)

        # Bottle neck
        current_layer = self.bottle_neck(current_layer)

        # Upsampling path
        for i in range(self.num_layers):
            block_name = f'{self.num_layers - i}'
            
            # Apply inverse DWT
            current_layer = self.idwt_layers[block_name](current_layer)
            
            # Pre-skip connection processing
            current_layer = self.up_process_blocks[block_name](current_layer)
            
            # Get skip connection from encoder
            skip_conn = enc_outputs[block_name]
            
            # Match dimensions if needed
            if current_layer.shape[1] != skip_conn.shape[1]:
                diff = skip_conn.shape[1] - current_layer.shape[1]
                if diff > 0:
                    # Pad if skip connection is larger
                    pad_start = diff // 2
                    pad_end = diff - pad_start
                    current_layer = tf.pad(current_layer, [[0, 0], [pad_start, pad_end], [0, 0]], mode='SYMMETRIC')
                else:
                    # Crop if current layer is larger
                    diff = -diff
                    crop_start = diff // 2
                    current_layer = tf.slice(current_layer, [0, crop_start, 0], [-1, skip_conn.shape[1], -1])
            
            # Apply enhanced skip connection
            current_layer = self.skip_connections[block_name]([current_layer, skip_conn])
            
            # Post-skip connection processing
            current_layer = self.upsampling_blocks[block_name](current_layer)

        # Final processing
        current_layer = self.final_conv(current_layer)
        
        # Ensure the final layer matches the input dimensions
        if current_layer.shape[1] != self.num_coeffs:
            diff = self.num_coeffs - current_layer.shape[1]
            if diff > 0:
                pad_start = diff // 2
                pad_end = diff - pad_start
                current_layer = tf.pad(current_layer, [[0, 0], [pad_start, pad_end], [0, 0]], mode='SYMMETRIC')
            else:
                diff = -diff
                crop_start = diff // 2
                current_layer = tf.slice(current_layer, [0, crop_start, 0], [-1, self.num_coeffs, -1])

        # Concatenate with input mixture
        current_layer = tf.concat([full_mix, current_layer], axis=-1)
        
        # Generate separate outputs for each source
        outputs = []
        for i in range(self.max_sources):
            outputs.append(self.output_convs[i](current_layer))
        
        # Stack outputs along a new axis
        return tf.stack(outputs, axis=1)  # Shape: [batch, max_sources, time, 1]


# Permutation Invariant Training Loss
def pit_loss(y_true, y_pred):
    """
    Implement permutation invariant training loss.
    
    Args:
        y_true: Ground truth sources [batch, max_sources, time, channels]
        y_pred: Predicted sources [batch, max_sources, time, channels]
        
    Returns:
        The minimum MSE loss across all possible permutations
    """
    batch_size = tf.shape(y_true)[0]
    num_sources = tf.shape(y_true)[1]
    
    # Calculate pairwise MSE for all combinations of true and predicted sources
    # Expand dims to create tensors of shape [batch, num_true, num_pred, time, channels]
    y_true_expanded = tf.expand_dims(y_true, axis=2)  # [batch, num_true, 1, time, channels]
    y_pred_expanded = tf.expand_dims(y_pred, axis=1)  # [batch, 1, num_pred, time, channels]
    
    # Calculate MSE for each pair
    pairwise_mse = tf.reduce_mean(tf.square(y_true_expanded - y_pred_expanded), axis=[3, 4])
    
    # Compute total loss for each possible permutation
    min_loss = tf.ones((batch_size,), dtype=tf.float32) * float('inf')
    
    # Limited number of sources (2-4), so we can enumerate all permutations
    all_perms = list(permutations(range(4)))
    
    for perm in all_perms:
        perm_tensor = tf.convert_to_tensor(perm, dtype=tf.int32)
        
        # For each sample in batch, calculate loss for this permutation
        batch_losses = []
        for b in range(8):
            loss = 0.0
            for i, p in enumerate(perm):
                loss += pairwise_mse[b, i, p]
            batch_losses.append(loss / tf.cast(num_sources, tf.float32))
        
        batch_losses = tf.stack(batch_losses)
        min_loss = tf.minimum(min_loss, batch_losses)
    
    return tf.reduce_mean(min_loss)


# Faster implementation for 2-4 sources
def fast_pit_loss(y_true, y_pred):
    """
    A faster implementation of PIT loss for a small, fixed number of sources.
    
    Args:
        y_true: Ground truth sources [batch, max_sources, time, channels]
        y_pred: Predicted sources [batch, max_sources, time, channels]
        
    Returns:
        The minimum MSE loss across all possible permutations
    """
    batch_size = tf.shape(y_true)[0]
    num_sources = tf.shape(y_true)[1]
    
    # Calculate all pairwise losses
    # [batch, true_sources, pred_sources]
    pairwise_loss = tf.reduce_mean(tf.square(tf.expand_dims(y_true, 2) - 
                                             tf.expand_dims(y_pred, 1)), 
                                   axis=[3, 4])
    
    # For 2 sources (2 permutations)
    if num_sources == 2:
        # Original order: [0,1]
        perm1_loss = (pairwise_loss[:, 0, 0] + pairwise_loss[:, 1, 1]) / 2.0
        # Swapped order: [1,0]
        perm2_loss = (pairwise_loss[:, 0, 1] + pairwise_loss[:, 1, 0]) / 2.0
        
        # Take minimum of the two permutations
        min_loss = tf.minimum(perm1_loss, perm2_loss)
        
    # For 3 sources (6 permutations)
    elif num_sources == 3:
        # Define the 6 possible permutations
        perms = [
            [0, 1, 2],  # Original
            [0, 2, 1],  # Swap 2nd and 3rd
            [1, 0, 2],  # Swap 1st and 2nd
            [1, 2, 0],  # Rotate left
            [2, 0, 1],  # Rotate right
            [2, 1, 0]   # Swap 1st and 3rd, then 2nd and 3rd
        ]
        
        # Calculate loss for each permutation
        perm_losses = []
        for perm in perms:
            loss = (pairwise_loss[:, 0, perm[0]] + 
                    pairwise_loss[:, 1, perm[1]] + 
                    pairwise_loss[:, 2, perm[2]]) / 3.0
            perm_losses.append(loss)
        
        # Stack and find minimum
        perm_losses = tf.stack(perm_losses, axis=1)
        min_loss = tf.reduce_min(perm_losses, axis=1)
        
    # For 4 sources (24 permutations)
    elif num_sources == 4:
        # Get all 24 permutations
        all_perms = list(permutations(range(4)))
        
        # Calculate loss for each permutation
        perm_losses = []
        for perm in all_perms:
            loss = (pairwise_loss[:, 0, perm[0]] + 
                    pairwise_loss[:, 1, perm[1]] + 
                    pairwise_loss[:, 2, perm[2]] + 
                    pairwise_loss[:, 3, perm[3]]) / 4.0
            perm_losses.append(loss)
        
        # Stack and find minimum
        perm_losses = tf.stack(perm_losses, axis=1)
        min_loss = tf.reduce_min(perm_losses, axis=1)
        
    else:
        raise ValueError(f"Unsupported number of sources: {num_sources}")
    
    return tf.reduce_mean(min_loss)


# Data Generator for TensorFlow
class AudioMixtureDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, source_combinations, batch_size=8, max_sources=4, shuffle=True,
                 workers=4, use_multiprocessing=True):
        self.source_combinations = source_combinations
        self.batch_size = batch_size
        self.max_sources = max_sources
        self.shuffle = shuffle
        self.workers = workers
        self.use_multiprocessing = use_multiprocessing
        self.indices = np.arange(len(self.source_combinations))
        if self.shuffle:
            np.random.shuffle(self.indices)

    
    def __len__(self):
        return len(self.source_combinations) // self.batch_size
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_combinations = [self.source_combinations[i] for i in batch_indices]
        
        # Initialize batch arrays
        X_batch = np.zeros((self.batch_size, config.SEGMENT_LENGTH, 1), dtype=np.float32)
        y_batch = np.zeros((self.batch_size, self.max_sources, config.SEGMENT_LENGTH, 1), dtype=np.float32)
        
        for i, source_paths in enumerate(batch_combinations):
            # Load sources
            sources = []
            for path in source_paths:
                source = np.load(path).astype(np.float32)
                # Reshape for channel dimension
                source = source.reshape(-1, 1)
                sources.append(source)
            
            # Create mixture by summing sources
            mixture = np.zeros((config.SEGMENT_LENGTH, 1), dtype=np.float32)
            for source in sources:
                mixture += source
            
            # Normalize mixture
            if np.max(np.abs(mixture)) > 0:
                mixture = mixture / np.max(np.abs(mixture))
            
            # Store mixture in X_batch
            X_batch[i] = mixture
            
            # Zero-pad sources if needed and store in y_batch
            sources_array = np.array(sources)
            if len(sources) < self.max_sources:
                # Create zero-padding for missing sources
                zero_pad = np.zeros((self.max_sources - len(sources), config.SEGMENT_LENGTH, 1), dtype=np.float32)
                sources_array = np.vstack([sources_array, zero_pad])
            
            y_batch[i] = sources_array
        
        return X_batch, y_batch
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


# Training Function
def train_model():
    # Load and preprocess data
    segments = load_and_preprocess_audio_files(config.DATA_DIR)
    source_combinations = create_source_combinations(segments, config.NUM_EXAMPLES)
    
    # Split into training and validation sets
    val_size = int(len(source_combinations) * config.VAL_SPLIT)
    train_combinations = source_combinations[:-val_size]
    val_combinations = source_combinations[-val_size:]
    
    print(f"Training on {len(train_combinations)} combinations")
    print(f"Validating on {len(val_combinations)} combinations")
    
    # Create data generators
    train_generator = AudioMixtureDataGenerator(
        train_combinations, 
        batch_size=config.BATCH_SIZE,
        max_sources=config.MAX_SOURCES,
        shuffle=True,
        workers=4,
        use_multiprocessing=True
    )
    
    val_generator = AudioMixtureDataGenerator(
        val_combinations, 
        batch_size=config.BATCH_SIZE,
        max_sources=config.MAX_SOURCES,
        shuffle=False,
        workers=4,
        use_multiprocessing=True
    )
    
    # Create the model
    model = ImprovedWaveletUNet(
        num_coeffs=config.NUM_COEFFS,
        wavelet_depth=config.WAVELET_DEPTH,
        batch_size=config.BATCH_SIZE,
        channels=config.CHANNELS,
        num_layers=config.NUM_LAYERS,
        num_init_filters=config.NUM_INIT_FILTERS,
        filter_size=config.FILTER_SIZE,
        merge_filter_size=config.MERGE_FILTER_SIZE,
        l1_reg=config.L1_REG,
        l2_reg=config.L2_REG,
        max_sources=config.MAX_SOURCES,
        wavelet_family=config.WAVELET_FAMILY,
        
    )
    
    # Compile the model with PIT loss
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
    dummy_input = tf.zeros((config.BATCH_SIZE, config.SEGMENT_LENGTH, 1))
    _ = model(dummy_input)
    model.compile(
        optimizer=optimizer,
        loss=pit_loss,
        metrics=['mse']  # For monitoring purposes
    )
    
    model.summary()
    
    # Set up callbacks
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(config.CHECKPOINT_DIR, 'improved_wavelet_unet_{epoch:02d}_{val_loss:.4f}.keras'),
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            save_weights_only=False,  # Save the entire model
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(config.CHECKPOINT_DIR, 'logs'),
            histogram_freq=1,
            update_freq='epoch'
        )
    ]
    
    # Train the model
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=config.EPOCHS,
        callbacks=callbacks,
        # workers=4,
        # use_multiprocessing=True
    )
    
    # Save the final model
    model_save_path = os.path.join(config.CHECKPOINT_DIR, 'improved_wavelet_unet_final.keras')
    
    # Special saving technique to ensure proper serialization
    model_json = model.to_json()
    with open(os.path.join(config.CHECKPOINT_DIR, 'model_architecture.json'), 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(os.path.join(config.CHECKPOINT_DIR, 'model_weights.h5'))
    
    # Also save in TensorFlow SavedModel format
    try:
        saved_model_path = os.path.join(config.CHECKPOINT_DIR, 'improved_wavelet_unet_savedmodel')
        tf.saved_model.save(model, saved_model_path)
        print(f"Model successfully saved to {saved_model_path}")
    except Exception as e:
        print(f"Error saving in SavedModel format: {e}")
        print("Falling back to H5 format only")
    
    try:
        # Save with h5 format
        model.save(model_save_path, save_format='h5')
        print(f"Model successfully saved to {model_save_path}")
    except Exception as e:
        print(f"Error saving in H5 format: {e}")
        print("Model is saved as architecture + weights only")
    
    return model, history


# Load model function with multiple fallbacks
def load_model_with_fallbacks(model_dir):
    """Load the model with multiple fallback options."""
    print(f"Attempting to load model from {model_dir}")
    
    # Try loading SavedModel format
    saved_model_path = os.path.join(model_dir, 'improved_wavelet_unet_savedmodel')
    if os.path.exists(saved_model_path):
        try:
            print("Attempting to load SavedModel format...")
            custom_objects = {
                'ImprovedWaveletUNet': ImprovedWaveletUNet,
                'DaubechiesWaveletLayer': DaubechiesWaveletLayer,
                'InverseDaubechiesWaveletLayer': InverseDaubechiesWaveletLayer,
                'EnhancedDownsamplingLayer': EnhancedDownsamplingLayer,
                'EnhancedUpsamplingLayer': EnhancedUpsamplingLayer,
                'EnhancedSkipConnection': EnhancedSkipConnection,
                'fast_pit_loss': fast_pit_loss,
                'gelu': gelu
            }
            
            model = tf.keras.models.load_model(
                saved_model_path,
                custom_objects=custom_objects
            )
            print("Successfully loaded model from SavedModel format")
            return model
        except Exception as e:
            print(f"Error loading SavedModel: {e}")
    
    # Try loading H5 format
    h5_path = os.path.join(model_dir, 'improved_wavelet_unet_final.h5')
    if os.path.exists(h5_path):
        try:
            print("Attempting to load H5 format...")
            custom_objects = {
                'ImprovedWaveletUNet': ImprovedWaveletUNet,
                'DaubechiesWaveletLayer': DaubechiesWaveletLayer,
                'InverseDaubechiesWaveletLayer': InverseDaubechiesWaveletLayer,
                'EnhancedDownsamplingLayer': EnhancedDownsamplingLayer,
                'EnhancedUpsamplingLayer': EnhancedUpsamplingLayer,
                'EnhancedSkipConnection': EnhancedSkipConnection,
                'fast_pit_loss': fast_pit_loss,
                'gelu': gelu
            }
            
            model = tf.keras.models.load_model(
                h5_path,
                custom_objects=custom_objects
            )
            print("Successfully loaded model from H5 format")
            return model
        except Exception as e:
            print(f"Error loading H5 model: {e}")
    
    # Try loading from architecture + weights
    arch_path = os.path.join(model_dir, 'model_architecture.json')
    weights_path = os.path.join(model_dir, 'model_weights.h5')
    
    if os.path.exists(arch_path) and os.path.exists(weights_path):
        try:
            print("Attempting to load from architecture + weights...")
            with open(arch_path, 'r') as json_file:
                model_json = json_file.read()
            
            custom_objects = {
                'ImprovedWaveletUNet': ImprovedWaveletUNet,
                'DaubechiesWaveletLayer': DaubechiesWaveletLayer,
                'InverseDaubechiesWaveletLayer': InverseDaubechiesWaveletLayer,
                'EnhancedDownsamplingLayer': EnhancedDownsamplingLayer,
                'EnhancedUpsamplingLayer': EnhancedUpsamplingLayer,
                'EnhancedSkipConnection': EnhancedSkipConnection
            }
            
            model = tf.keras.models.model_from_json(
                model_json,
                custom_objects=custom_objects
            )
            
            # Load weights
            model.load_weights(weights_path)
            
            # Recompile the model
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
                loss=fast_pit_loss,
                metrics=['mse']
            )
            
            print("Successfully loaded model from architecture + weights")
            return model
        except Exception as e:
            print(f"Error loading from architecture + weights: {e}")
    
    print("Failed to load model with any method")
    return None


# Evaluate the model
def evaluate_model(model, test_generator, num_examples=5):
    """Evaluate the model and visualize separation results."""
    # Get samples from the test generator
    X_test, y_test = test_generator.__getitem__(0)
    
    # Select a few examples
    indices = np.random.choice(X_test.shape[0], num_examples, replace=False)
    
    # Make predictions
    y_pred = model.predict(X_test[indices])
    
    # Calculate SDR
    sdrs = []
    
    for i in range(num_examples):
        # For each example, calculate SDR for active sources
        example_sdrs = []
        
        for j in range(config.MAX_SOURCES):
            # Check if this is an active source (not zero-padded)
            source_energy = np.sum(np.abs(y_test[indices[i], j]))
            
            if source_energy > 1e-6:  # Only evaluate active sources
                target = y_test[indices[i], j, :, 0]
                estimate = y_pred[i, j, :, 0]
                
                # Calculate SDR
                target_energy = np.sum(target**2)
                error = target - estimate
                error_energy = np.sum(error**2)
                
                sdr = 10 * np.log10(target_energy / (error_energy + 1e-10))
                example_sdrs.append(sdr)
                print(f"Example {i+1}, Source {j+1}: SDR = {sdr:.2f} dB")
        
        if example_sdrs:
            avg_example_sdr = np.mean(example_sdrs)
            sdrs.append(avg_example_sdr)
            print(f"Example {i+1} Average SDR: {avg_example_sdr:.2f} dB")
    
    # Calculate overall average SDR
    if sdrs:
        avg_sdr = np.mean(sdrs)
        print(f"Overall Average SDR: {avg_sdr:.2f} dB")
    
    # Plotting
    plt.figure(figsize=(20, 4 * num_examples))
    
    for i in range(num_examples):
        # Plot mixture
        plt.subplot(num_examples, config.MAX_SOURCES + 2, i * (config.MAX_SOURCES + 2) + 1)
        plt.plot(X_test[indices[i], :, 0])
        plt.title(f"Example {i+1} - Mixture")
        plt.ylim([-1, 1])
        
        # Plot true and predicted sources
        for j in range(config.MAX_SOURCES):
            # True source
            plt.subplot(num_examples, config.MAX_SOURCES + 2, i * (config.MAX_SOURCES + 2) + j + 2)
            plt.plot(y_test[indices[i], j, :, 0])
            plt.title(f"True Source {j+1}")
            plt.ylim([-1, 1])
            
            # Predicted source
            plt.subplot(num_examples, config.MAX_SOURCES + 2, i * (config.MAX_SOURCES + 2) + config.MAX_SOURCES + 2)
            plt.plot(y_pred[i, j, :, 0])
            plt.title(f"Pred Source {j+1}")
            plt.ylim([-1, 1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.CHECKPOINT_DIR, 'evaluation_results.png'))
    plt.show()
    
    return sdrs


if __name__ == "__main__":
    # Train the model
    model, history = train_model()
    
    # Create test generator
    segments = glob("content/preprocessed/segment_*.npy")
    test_combinations = create_source_combinations(segments[:100], num_examples=100)
    
    test_generator = AudioMixtureDataGenerator(
        test_combinations,
        batch_size=config.BATCH_SIZE,
        max_sources=config.MAX_SOURCES,
        shuffle=False
    )
    
    # Evaluate the model
    evaluate_model(model, test_generator)
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mse'])
    plt.plot(history.history['val_mse'])
    plt.title('Model MSE')
    plt.ylabel('MSE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.CHECKPOINT_DIR, 'training_history.png'))
    plt.show()
    
    print("Improved Wavelet U-Net pipeline completed successfully!")