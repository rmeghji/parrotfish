import json
import os
import random
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from scipy.signal import windows
import soundfile as sf
import tensorflow as tf
from tqdm import tqdm
import pywt
import glob
import time
import gc

class Waveform:
    """Class to store waveform data with associated metadata"""
    def __init__(self, waveform, source_id=None, source_ids=None):
        self.waveform = waveform
        self.source_id = source_id
        self.source_ids = source_ids if source_ids else []
        self.tensor_coeffs = None

def getWaveletTransform(wave_dict, wave_id, wavelet_level):
    """Perform wavelet transform on a waveform in the wave dictionary"""
    if wave_dict[wave_id].tensor_coeffs is not None:
        return wave_dict[wave_id].tensor_coeffs
    
    # Apply wavelet transform
    coeffs = pywt.wavedec(wave_dict[wave_id].waveform, 'db4', level=wavelet_level)
    
    # Convert coefficients to tensors
    tensor_coeffs = tf.convert_to_tensor(coeffs, dtype=tf.float32)
    
    # Store tensor coefficients
    wave_dict[wave_id].tensor_coeffs = tensor_coeffs
    
    return tensor_coeffs

def makeWaveDictBatch(mixed_waveforms, source_waveforms, source_ids_list, mixed_ids):
    """Create dictionary containing both mixed and source waveforms"""
    wave_dict = {}
    
    # Add mixed waveforms
    for i, (mixed_waveform, mix_id) in enumerate(zip(mixed_waveforms, mixed_ids)):
        wave_dict[mix_id] = Waveform(mixed_waveform, source_id=mix_id, source_ids=source_ids_list[i])
    
    # Add source waveforms
    for i, source_list in enumerate(source_ids_list):
        for j, source_id in enumerate(source_list):
            if source_id and source_id not in wave_dict:  # Skip empty sources
                wave_dict[source_id] = Waveform(source_waveforms[i][j], source_id=source_id)
    
    return wave_dict

class AudioProcessor:
    def __init__(self, clip_duration_seconds=1.0, window_overlap_ratio=0.1):
        """Initialize audio processor with clip settings"""
        self.clip_duration_seconds = clip_duration_seconds
        self.window_overlap_ratio = window_overlap_ratio
        self.samples_per_clip = int(clip_duration_seconds * 16000)
        self.overlap_samples = int(self.samples_per_clip * window_overlap_ratio)
        self.window = windows.hann(self.samples_per_clip)
    
    def load_and_normalize_audio(self, file_path):
        """Load audio file and normalize it"""
        try:
            # Read audio file
            if str(file_path).lower().endswith('.wav'):
                sample_rate, audio_array = wavfile.read(str(file_path))
                if sample_rate != 16000:
                    print(f"Warning: Audio file {file_path} sample rate is {sample_rate}Hz (expected 16kHz)")
                    # Implement resampling if needed
            else:
                # Use soundfile for other formats
                audio_array, sample_rate = sf.read(str(file_path))
                if sample_rate != 16000:
                    print(f"Warning: Audio file {file_path} sample rate is {sample_rate}Hz (expected 16kHz)")
                    # Implement resampling if needed
            
            # Convert to float32 if needed
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)
                if audio_array.max() > 1.0:
                    audio_array = audio_array / 32768.0  # Normalize 16-bit integer
            
            # Convert to mono if stereo
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)
            
            # Normalize
            max_val = np.max(np.abs(audio_array))
            if max_val > 0:
                audio_array = audio_array / max_val
            
            return audio_array
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None
    
    def split_into_clips(self, audio):
        """Split audio into non-overlapping 1-second clips with light Hann windowing"""
        # Create a very light Hann window (mostly 1s with slight tapering at edges)
        window = windows.hann(self.samples_per_clip)
        # Make the window mostly flat with just slight edge tapering
        window = 0.1 * window + 0.9  # This makes the window range from 0.9 to 1.0
        
        # Calculate number of complete clips
        num_clips = len(audio) // self.samples_per_clip
        clips = []
        
        for i in range(num_clips):
            start = i * self.samples_per_clip
            end = start + self.samples_per_clip
            clip = audio[start:end]
            # Apply light windowing
            clip = clip * window
            clips.append(clip)
        
        # Handle remaining audio if any
        if len(audio) % self.samples_per_clip > 0:
            start = num_clips * self.samples_per_clip
            remaining = audio[start:]
            # Pad with zeros to reach full duration
            clip = np.pad(remaining, (0, self.samples_per_clip - len(remaining)))
            # Apply light windowing
            clip = clip * window
            clips.append(clip)
        
        return clips if clips else [np.zeros(self.samples_per_clip)]
    
    def segment_audio(self, input_file, output_dir):
        """Process a single audio file and save clips"""
        # Load and normalize audio
        audio = self.load_and_normalize_audio(input_file)
        if audio is None:
            return
        
        # Split into clips
        clips = self.split_into_clips(audio)
        
        # Create base output directory if it doesn't exist
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save clips
        base_name = Path(input_file).stem
        for i, clip in enumerate(clips):
            # Generate clip name
            clip_name = f"{base_name}_clip_{i:03d}.wav"
            
            # Hash the clip name to determine subfolder (0-499)
            hash_value = hash(clip_name) % 500
            subfolder = output_dir / f"{hash_value:03d}"
            
            # Create subfolder if it doesn't exist
            subfolder.mkdir(exist_ok=True)
            
            # Save clip in the appropriate subfolder
            output_path = subfolder / clip_name
            sf.write(str(output_path), clip, 16000, format='WAV')

    def serialize_audio(self, audio_path):
        """Convert a single audio file to TFRecord format features."""
        audio_binary = tf.io.read_file(audio_path)
        return {
            'audio_binary': tf.train.Feature(bytes_list=tf.train.BytesList(value=[audio_binary.numpy()])),
            'path': tf.train.Feature(bytes_list=tf.train.BytesList(value=[audio_path.encode()]))
        }

    def create_tf_example(self, audio_path):
        """Create a complete TF Example from an audio file."""
        feature_dict = self.serialize_audio(audio_path)
        example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
        return example

    def convert_dataset_to_tfrecords_in_batches(self, base_input_dir, base_output_dir, batch_size=50, pause_duration=5):
        """Convert audio files from GCS to TFRecords in GCS using batches"""
        
        # Ensure output directory exists
        if not tf.io.gfile.exists(base_output_dir):
            tf.io.gfile.makedirs(base_output_dir)
        
        # List all subdirectories in GCS
        all_subdirs = tf.io.gfile.glob(f"{base_input_dir}/*")
        all_subdirs = [d for d in all_subdirs if tf.io.gfile.isdir(d)]
        
        print(f"Found {len(all_subdirs)} subdirectories to process")
        
        # Check which subdirectories have already been processed
        existing_tfrecords = tf.io.gfile.glob(f"{base_output_dir}/*.tfrecord")
        processed_subdirs = set()
        
        for tfrecord_path in existing_tfrecords:
            subdir_name = os.path.basename(tfrecord_path).replace('.tfrecord', '')
            processed_subdirs.add(subdir_name)
        
        # Filter out already processed subdirectories
        subdirs_to_process = []
        for subdir in all_subdirs:
            subdir_name = os.path.basename(subdir)
            if subdir_name not in processed_subdirs:
                subdirs_to_process.append(subdir)
        
        print(f"Skipping {len(all_subdirs) - len(subdirs_to_process)} already processed subdirectories")
        print(f"Will process {len(subdirs_to_process)} remaining subdirectories")
        
        # Process in batches
        for batch_idx in range(0, len(subdirs_to_process), batch_size):
            batch_start_time = time.time()
            
            batch_subdirs = subdirs_to_process[batch_idx:batch_idx + batch_size]
            print(f"\nProcessing batch {batch_idx//batch_size + 1}/{(len(subdirs_to_process)-1)//batch_size + 1} with {len(batch_subdirs)} subdirectories")
            
            # Process each subdirectory in this batch
            for subdir in batch_subdirs:
                subdir_name = os.path.basename(subdir)
                output_path = f"{base_output_dir}/{subdir_name}.tfrecord"
                
                # List audio files in this subdirectory
                audio_files = []
                for ext in ['.wav', '.mp3', '.flac', '.ogg']:
                    audio_files.extend(tf.io.gfile.glob(f"{subdir}/*{ext}"))
                
                print(f"  - {subdir_name}: Found {len(audio_files)} audio files")
                
                # Create TFRecord directly in GCS
                with tf.io.TFRecordWriter(output_path) as writer:
                    for audio_path in audio_files:
                        # Read file directly from GCS
                        audio_binary = tf.io.read_file(audio_path)
                        
                        # Create example
                        feature = {
                            'audio_binary': tf.train.Feature(bytes_list=tf.train.BytesList(value=[audio_binary.numpy()])),
                            'path': tf.train.Feature(bytes_list=tf.train.BytesList(value=[audio_path.encode()]))
                        }
                        example = tf.train.Example(features=tf.train.Features(feature=feature))
                        
                        # Write to GCS
                        writer.write(example.SerializeToString())
            
            # Calculate batch processing time
            batch_duration = time.time() - batch_start_time
            print(f"Batch processed in {batch_duration:.2f} seconds")
            
            # Add short pause between batches
            if batch_idx + batch_size < len(subdirs_to_process):
                print(f"Waiting {pause_duration} seconds before starting next batch...")
                time.sleep(pause_duration)

def process_audio_files(base_folder, output_folder, clip_duration_seconds=1.0, window_overlap_ratio=0.1, batch_size=100):
    """Process all audio files in a folder and save clips to output folder with subfolder distribution"""
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Initialize processor
    processor = AudioProcessor(
        clip_duration_seconds=clip_duration_seconds,
        window_overlap_ratio=window_overlap_ratio
    )
    
    # Get all audio files recursively
    audio_files = []
    for dirpath, dirnames, filenames in os.walk(base_folder):
        for filename in filenames:
            if filename.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a')):
                audio_files.append(os.path.join(dirpath, filename))
    
    print(f"Found {len(audio_files)} audio files to process")
    
    # Process files in batches
    total_clips = 0
    successful_files = 0
    batch_start = 0
    
    while batch_start < len(audio_files):
        batch_end = min(batch_start + batch_size, len(audio_files))
        batch = audio_files[batch_start:batch_end]
        
        print(f"\nProcessing batch {batch_start//batch_size + 1} ({batch_start}-{batch_end} of {len(audio_files)})")
        
        for audio_file in tqdm(batch, desc="Processing audio files"):
            try:
                # Process the audio file using segment_audio
                processor.segment_audio(audio_file, output_folder)
                successful_files += 1
                
                # Count clips for statistics
                audio = processor.load_and_normalize_audio(audio_file)
                if audio is not None:
                    total_clips += len(processor.split_into_clips(audio))
                
            except Exception as e:
                print(f"Error processing {audio_file}: {str(e)}")
        
        batch_start = batch_end
    
    print(f"\nAudio segmentation complete!")
    print(f"Successfully processed {successful_files}/{len(audio_files)} files")
    print(f"Generated approximately {total_clips} clips distributed across subfolders in {output_folder}")

class AudioMixerGenerator:
    def __init__(self, base_dir, clips_dir, num_speakers=4, batch_size=1000):
        """
        Initialize the AudioMixerGenerator for mixing pre-processed 1-second clips.
        
        Args:
            base_dir: Base directory containing the data
            clips_dir: Directory containing the pre-processed 1-second audio clips
            num_speakers: Number of speakers to mix together
            batch_size: Number of files to process at once when selecting random files
        """
        self.base_dir = Path(base_dir)
        self.clips_dir = Path(clips_dir)
        self.input_dir = self.clips_dir
        self.num_speakers = num_speakers
        self.batch_size = batch_size
        self.samples_per_clip = 16000  # 1-second clips at 16kHz
        
    def load_clip(self, file_path):
        """Load a pre-processed 1-second clip"""
        try:
            sample_rate, audio_array = wavfile.read(str(file_path))
            if sample_rate != 16000:
                raise ValueError(f"Clip {file_path} must be 16kHz (found {sample_rate}Hz)")
            
            if len(audio_array) != self.samples_per_clip:
                raise ValueError(f"Clip {file_path} must be exactly 1 second (found {len(audio_array)/16000:.2f}s)")
            
            # Convert to float32 if needed
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)
                if audio_array.max() > 1.0:
                    audio_array = audio_array / 32768.0  # Normalize 16-bit integer
            
            return audio_array
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None
    
    def get_random_files(self):
        """Get random files without loading entire directory"""
        all_files = []
        max_attempts = 10  # Prevent infinite loops
        attempts = 0
        
        while len(all_files) < self.num_speakers and attempts < max_attempts:
            # Randomly select a subfolder (000-499)
            subfolder = f"{random.randint(0, 499):03d}"
            subfolder_path = self.clips_dir / subfolder
            
            if not subfolder_path.exists():
                attempts += 1
                continue
            
            # Get all WAV files in this subfolder
            files = [f for f in os.listdir(subfolder_path) if f.endswith('.wav')]
            if not files:
                attempts += 1
                continue
            
            # Add random files from this subfolder
            needed = self.num_speakers - len(all_files)
            batch_samples = random.sample(files, min(needed, len(files)))
            # Store as subfolder/filename for later use
            all_files.extend(f"{subfolder}/{f}" for f in batch_samples)
        
        if len(all_files) < self.num_speakers:
            print(f"Warning: Could only find {len(all_files)} valid files")
        
        return all_files
    
    def generate_sample(self):
        """Mix multiple 1-second clips together
        
        Returns:
            tuple: (mixed_audio, input_clips) where:
                - mixed_audio: tensor of shape (1, samples_per_clip) containing the mixed audio
                - input_clips: tensor of shape (num_speakers, samples_per_clip) containing the individual clips
        """
        # Randomly select input files using batch processing
        selected_files = self.get_random_files()
        
        # Initialize arrays for mixed and individual clips
        mixed = np.zeros(self.samples_per_clip)
        input_clips = []
        source_ids = []
        
        for rel_path in selected_files:
            file_path = self.input_dir / rel_path
            audio = self.load_clip(file_path)
            
            if audio is not None:
                mixed += audio
                input_clips.append(audio)
                source_ids.append(rel_path)
        
        if not input_clips:
            # Return zeros if no valid audio was found
            return (
                tf.zeros((1, self.samples_per_clip), dtype=tf.float32),
                tf.zeros((self.num_speakers, self.samples_per_clip), dtype=tf.float32),
                []
            )
        
        # Normalize the mixed audio
        max_val = np.max(np.abs(mixed))
        if max_val > 0:
            mixed = mixed / max_val
        
        # Convert to tensors
        mixed_tensor = tf.convert_to_tensor([mixed], dtype=tf.float32)
        
        # Pad input_clips if we got fewer clips than num_speakers
        while len(input_clips) < self.num_speakers:
            input_clips.append(np.zeros(self.samples_per_clip))
            source_ids.append('')
        
        sources_tensor = tf.convert_to_tensor(input_clips, dtype=tf.float32)
        
        return mixed_tensor, sources_tensor, source_ids

    def batch_training_data(self, batch_size=32, wavelet_level=5):
        """Batch the training data
        
        params:
        - batch_size: int, batch size
        - wavelet_level: int, wavelet level
        
        return:
        - X: tf.Tensor, mixed waveforms
        - Y: tf.Tensor, source waveforms
        """
        mixed_waveforms = []
        all_source_waveforms = []
        all_source_ids = []
        
        for _ in range(batch_size):
            mixed_tensor, sources_tensor, source_ids = self.generate_sample()
            mixed_waveforms.append(mixed_tensor.numpy()[0])
            all_source_waveforms.append(sources_tensor.numpy())
            all_source_ids.append(source_ids)
        
        mixed_ids = ["_".join(ids) for ids in all_source_ids]
        wave_dict = makeWaveDictBatch(mixed_waveforms, all_source_waveforms, all_source_ids, mixed_ids)

        X = []
        Y = []

        # Comment out the wavelet transform code for now
        # We'll use direct waveforms instead of wavelet coefficients
        for mix_id in mixed_ids:
            X.append(wave_dict[mix_id].waveform)
        
        for source_ids in all_source_ids:
            source_list = []
            for source_id in source_ids:
                if source_id:  # Skip empty sources
                    source_list.append(wave_dict[source_id].waveform)
                else:
                    source_list.append(np.zeros(self.samples_per_clip))
            Y.append(source_list)

        X = tf.convert_to_tensor(X)
        Y = tf.convert_to_tensor(Y)
        
        X = tf.expand_dims(X, axis=-1)
        Y = tf.expand_dims(Y, axis=-1)

        return X, Y

def create_tf_dataset(base_dir, clips_dir, num_speakers, batch_size=32, wavelet_level=5):
    """Create a TensorFlow dataset that generates audio samples on-the-fly"""
    mixer = AudioMixerGenerator(
        base_dir,
        clips_dir,
        num_speakers=num_speakers
    )
    
    def generator_fn():
        while True:
            X, Y = mixer.batch_training_data(batch_size, wavelet_level)
            yield X, Y

    # Create output signature based on direct waveforms
    dataset = tf.data.Dataset.from_generator(
        generator_fn,
        output_signature=(
            tf.TensorSpec(shape=(batch_size, mixer.samples_per_clip, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(batch_size, num_speakers, mixer.samples_per_clip, 1), dtype=tf.float32)
        )
    )

    return dataset.prefetch(tf.data.AUTOTUNE)

# DWT-based network layers
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
        self.filter_length = len(wavelet.dec_lo)
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
    def __init__(self, wavelet_family='db4', batch_size=16, mode='periodization', name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.wavelet_family = wavelet_family
        self.mode = mode
        self.batch_size = batch_size   
        
        # Get Daubechies wavelet filter coefficients
        wavelet = pywt.Wavelet(wavelet_family)
        self.rec_lo = tf.constant(wavelet.rec_lo, dtype=tf.float32)
        self.rec_hi = tf.constant(wavelet.rec_hi, dtype=tf.float32)
    
    def build(self, input_shape):
        wavelet = pywt.Wavelet(self.wavelet_family)
        self.filter_length = len(wavelet.rec_lo)
        self.in_channels = input_shape[-1] // 2  # Half for approx, half for detail
        
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
        batch_size = tf.shape(x)[0]
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
            'mode': self.mode,
            'batch_size': self.batch_size
        })
        return config

# Main training function for testing
def test_pipeline(base_dir, clips_dir, num_speakers=4, batch_size=16):
    """Test the entire pipeline by creating a dataset and checking sample outputs"""
    print("Testing audio source separation pipeline...")
    
    # Create dataset
    dataset = create_tf_dataset(base_dir, clips_dir, num_speakers, batch_size)
    
    # Test fetching a batch
    for x_batch, y_batch in dataset.take(1):
        print(f"Input shape: {x_batch.shape}")
        print(f"Output shape: {y_batch.shape}")
        
        # Test DWT layer
        print("Testing DWT layer...")
        dwt_layer = DWTLayer(wavelet_family='db4')
        dwt_output = dwt_layer(x_batch)
        print(f"DWT output shape: {dwt_output.shape}")
        
        # Test IDWT layer
        print("Testing IDWT layer...")
        idwt_layer = IDWTLayer(wavelet_family='db4', batch_size=batch_size)
        idwt_output = idwt_layer(dwt_output)
        print(f"IDWT output shape: {idwt_output.shape}")
        
        # Calculate reconstruction error
        input_slice = x_batch[:, :idwt_output.shape[1], :]
        recon_error = tf.reduce_mean(tf.square(input_slice - idwt_output))
        print(f"Reconstruction error: {recon_error}")
        
        return True
    
    return False

if __name__ == "__main__":
    # Example usage
    base_dir = "/path/to/data"
    clips_dir = "/path/to/clips"
    
    # Process audio files
    # process_audio_files(base_dir, clips_dir)
    
    # Test pipeline
    success = test_pipeline(base_dir, clips_dir)
    if success:
        print("Pipeline tested successfully!")
    else:
        print("Pipeline test failed!")