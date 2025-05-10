import os
import numpy as np
import tensorflow as tf
import random
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from itertools import combinations, permutations
import soundfile as sf
import librosa
import pywt
from utils.Pipeline import (
    create_tf_dataset,
    create_tf_dataset_from_tfrecords,
)
from utils.config import Config

config = Config()

# GELU Activation Function
def gelu(x):
    """Gaussian Error Linear Unit activation function"""
    return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))

# Wavelet Transform Functions
@tf.keras.utils.register_keras_serializable()
class DWTLayer(tf.keras.layers.Layer):
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
class IDWTLayer(tf.keras.layers.Layer):
    def __init__(self, wavelet_family='db4',batch_size=16, mode='periodization', name=None, **kwargs):
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

    @tf.function(jit_compile=True, reduce_retracing=True)
    def _upsample(self, x):
        """Vectorized upsampling without loops"""
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        channels = tf.shape(x)[2]
        output = tf.zeros([batch_size, seq_len*2, channels], dtype=x.dtype)
        batch_indices = tf.repeat(tf.range(batch_size), seq_len)
        seq_indices = tf.tile(tf.range(0, seq_len*2, 2), [batch_size])
        indices = tf.stack([batch_indices, seq_indices], axis=1)
        updates = tf.reshape(x, [-1, channels])

        return tf.tensor_scatter_nd_update(output, indices, updates)

    
    def get_config(self):
        config = super().get_config()
        config.update({
            'wavelet_family': self.wavelet_family,
            'mode': self.mode
        })
        return config


@tf.keras.utils.register_keras_serializable()
class DownsamplingLayer(tf.keras.layers.Layer):
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
        
            
        # Main path
        x = self.conv(inputs)
        x = self.layer_norm(x)
        x = gelu(x)
        
        # Add residual
        return x 
    
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
class UpsamplingLayer(tf.keras.layers.Layer):
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
        # half_channels = self.input_channels // 2
        self.approx_conv = tf.keras.layers.Conv1D(
            self.input_channels,
            self.filter_size,
            padding='same',
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg),
            name=f'approx_conv_{self.name}'
        )
        
        # self.detail_conv = tf.keras.layers.Conv1D(
        #     self.input_channels,
        #     self.filter_size,
        #     padding='same',
        #     kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg),
        #     name=f'detail_conv_{self.name}'
        # )
        
        # # Gating mechanisms
        # self.approx_gate = tf.keras.layers.Conv1D(
        #     self.input_channels,
        #     1,
        #     activation='sigmoid',
        #     padding='same',
        #     name=f'approx_gate_{self.name}'
        # )
        
        # self.detail_gate = tf.keras.layers.Conv1D(
        #     self.input_channels,
        #     1,
        #     activation='sigmoid',
        #     padding='same',
        #     name=f'detail_gate_{self.name}'
        # )
        
        # Layer normalization
        self.approx_norm = tf.keras.layers.LayerNormalization(name=f'approx_norm_{self.name}')
        # self.detail_norm = tf.keras.layers.LayerNormalization(name=f'detail_norm_{self.name}')
        
        # Final convolution after recombination
        # self.output_conv = tf.keras.layers.Conv1D(
        #     self.num_filters,
        #     1,
        #     padding='same',
        #     kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg),
        #     name=f'output_conv_{self.name}'
        # )
        
        # self.output_norm = tf.keras.layers.LayerNormalization(name=f'output_norm_{self.name}')
        
        super().build(input_shape)

    def call(self, inputs):
        # Split into approximation and detail coefficients
        
        
        # Process each separately
        approx_processed = self.approx_conv(inputs)
        approx_processed = self.approx_norm(approx_processed)
        approx_processed = gelu(approx_processed)
        
        
        
        
        # Recombine
        
        
        
        
        # Final processing
        # output = self.output_conv(combined)
        # output = self.output_norm(output)
        # output = gelu(output)
        
        return approx_processed 
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_filters': self.num_filters,
            'filter_size': self.filter_size,
            'l1_reg': self.l1_reg,
            'l2_reg': self.l2_reg
        })
        return config


# Gated Skip Connection
@tf.keras.utils.register_keras_serializable()
class GatedSkipConnection(tf.keras.layers.Layer):
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


#  Wavelet U-Net Model
@tf.keras.utils.register_keras_serializable()
class WaveletUNet(tf.keras.Model):
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
        # self.initial_conv = tf.keras.layers.Conv1D(
        #     self.num_init_filters,
        #     self.filter_size,
        #     padding='same',
        #     kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg),
        #     name='initial_conv'
        # )
        # self.initial_norm = tf.keras.layers.LayerNormalization(name='initial_norm')
        
        # Create enhanced downsampling blocks
        self.downsampling_blocks = {}
        self.dwt_layers = {}
        self.down_process_blocks = {}
        
        for i in range(self.num_layers):
            block_name = f'{i+1}'
            num_filters = self.num_init_filters + (self.num_init_filters * i)
            
            # Main downsampling block
            self.downsampling_blocks[block_name] = DownsamplingLayer(
                num_filters, 
                self.filter_size, 
                l1_reg=self.l1_reg, 
                l2_reg=self.l2_reg,
                name=f'ds_block_{block_name}'
            )
            
            # DWT layer
            self.dwt_layers[block_name] = DWTLayer(
                wavelet_family=self.wavelet_family,
                name=f'dwt_{block_name}'
            )
            
            # Post-DWT processing
            self.down_process_blocks[block_name] = UpsamplingLayer(
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
            self.idwt_layers[block_name] = IDWTLayer(
                wavelet_family=self.wavelet_family,
                name=f'idwt_{block_name}'
            )
            
            # Pre-skip connection processing
            self.up_process_blocks[block_name] = UpsamplingLayer(
                num_filters,
                self.filter_size,
                l1_reg=self.l1_reg,
                l2_reg=self.l2_reg,
                name=f'up_process_{block_name}'
            )
            
            # Gated skip connection
            self.skip_connections[block_name] = GatedSkipConnection(
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
        
        # self.source_projection = tf.keras.layers.Conv1D(
        #     filters=1,  # Match the number of channels in mixture (typically 1 or 2)
        #     kernel_size=1,         # 1x1 convolution for projection
        #     padding='same',
        #     activation=None,        # Linear projection
        #     name='source_projection'
        # )
        
        # Output layer for each source
        # self.output_convs = []
        # for i in range(self.max_sources):
        #     self.output_convs.append(
        #         tf.keras.layers.Conv1D(
        #             1,
        #             1,
        #             activation='tanh',
        #             padding='same',
        #             name=f'output_conv_{i}'
        #         )
        #     )
            
        self.output_conv = tf.keras.layers.Conv1D(
            self.max_sources,
            1,
            activation='tanh',
            padding='same',
            name='output_conv'
        )
            
        super().build(input_shape)
        self.summary()

    def call(self, inputs, training=True):
        # Initial processing
        # Store the input for skip connection to final layer
        full_mix = tf.reduce_sum(inputs, axis=-1, keepdims=True)
        current_layer = inputs
        # current_layer = self.initial_conv(inputs)
        # current_layer = self.initial_norm(current_layer)
        # current_layer = gelu(current_layer)
        
        
        
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

        current_layer = tf.concat([full_mix, current_layer], axis=-1)
        
        # Generate separate outputs for each source using residual connections
        outputs = self.output_conv(current_layer)
        
        # outputs = tf.expand_dims(outputs, axis=-1)  # Add a new axis for channels[0 bs, 1 features, 2 chan/sources, 3 new axis]
        outputs = tf.transpose(outputs, [0, 2, 1])  # Transpose to [batch, sources, time]
        # outputs = tf.transpose(outputs, [0, 2, 1, 3])

        
        # outputs = tf.reshape(outputs, [self.batch_size, self.max_sources, self.num_coeffs, -1])
    
        # Stack outputs along a new axis
        return outputs


# Permutation Invariant Training Loss
@tf.keras.utils.register_keras_serializable()
def pit_loss(y_true, y_pred):
    """
    Permutation Invariant Training loss for audio source separation.
    Handles the specific case where:
    - y_true has shape [batch, sources, time] (e.g., [batch, 2, 16000])
    - y_pred has shape [batch, time, sources] (e.g., [batch, 16000, 2])
    
    Args:
        y_true: Ground truth with shape [batch, sources, time]
        y_pred: Model prediction with shape [batch, time, sources]
        
    Returns:
        The PIT loss value (scalar)
    """
    # Print shapes for debugging
    # print(f"y_true shape: {y_true.shape}")
    # print(f"y_pred shape: {y_pred.shape}")
    
    # Transpose y_pred from [batch, time, sources] to [batch, sources, time]
    # to match y_true's dimensions
    # y_pred_transposed = tf.transpose(y_pred, [0, 2, 1])
    # print(f"Transposed y_pred shape: {y_pred_transposed.shape}")
    
    # Now both tensors should have shape [batch, sources, time]
    # Extract sources (assuming 2 sources)
    y_true_s1 = y_true[:, 0, :]  # [batch, time]
    y_true_s2 = y_true[:, 1, :]  # [batch, time]
    y_pred_s1 = y_pred[:, 0, :]  # [batch, time]
    y_pred_s2 = y_pred[:, 1, :]  # [batch, time]
    
    # Calculate MSE for both permutations
    # Permutation 1: (true_s1, pred_s1), (true_s2, pred_s2)
    mse_1_1 = tf.reduce_mean(tf.square(y_true_s1 - y_pred_s1), axis=1)  # [batch]
    mse_2_2 = tf.reduce_mean(tf.square(y_true_s2 - y_pred_s2), axis=1)  # [batch]
    loss_perm1 = (mse_1_1 + mse_2_2) / 2.0  # [batch]
    
    # Permutation 2: (true_s1, pred_s2), (true_s2, pred_s1)
    mse_1_2 = tf.reduce_mean(tf.square(y_true_s1 - y_pred_s2), axis=1)  # [batch]
    mse_2_1 = tf.reduce_mean(tf.square(y_true_s2 - y_pred_s1), axis=1)  # [batch]
    loss_perm2 = (mse_1_2 + mse_2_1) / 2.0  # [batch]
    
    # Choose the minimum loss between the two permutations
    min_loss = tf.minimum(loss_perm1, loss_perm2)  # [batch]
    
    # Mixture consistency constraint
    true_sum = y_true_s1 + y_true_s2  # [batch, time]
    pred_sum = y_pred_s1 + y_pred_s2  # [batch, time]
    sum_loss = tf.reduce_mean(tf.square(true_sum - pred_sum), axis=1)  # [batch]
    
    # Combine losses
    alpha = 0.4  # Weight for the mixture consistency constraint
    combined_loss = min_loss + alpha * sum_loss  # [batch]
    
    return tf.reduce_mean(combined_loss) 


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