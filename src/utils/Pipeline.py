import json
import os
import random
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from scipy.signal import windows
import soundfile as sf
import tensorflow as tf
import pywt
import glob
import time
import gc
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm

class Waveform:
    """Class to store waveform data with associated metadata"""
    def __init__(self, waveform, source_id=None, source_ids=None):
        self.waveform = waveform
        self.source_id = source_id
        self.source_ids = source_ids if source_ids else []
        self.tensor_coeffs = None

def makeWaveDictBatch(mixed_waveforms, source_waveforms, source_ids_list, mixed_ids):
    """Create dictionary containing both mixed and source waveforms"""
    wave_dict = {}
    
    for i, (mixed_waveform, mix_id) in enumerate(zip(mixed_waveforms, mixed_ids)):
        wave_dict[mix_id] = Waveform(mixed_waveform, source_id=mix_id, source_ids=source_ids_list[i])
    
    for i, source_list in enumerate(source_ids_list):
        for j, source_id in enumerate(source_list):
            if source_id and source_id not in wave_dict:
                wave_dict[source_id] = Waveform(source_waveforms[i][j], source_id=source_id)
    
    return wave_dict

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
        self.samples_per_clip = 16000
        
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
                    audio_array = audio_array / 32768.0
            
            return audio_array
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None
    
    def get_random_files(self):
        """Get random files without loading entire directory"""
        all_files = []
        max_attempts = 10
        attempts = 0
        
        while len(all_files) < self.num_speakers and attempts < max_attempts:
            subfolder = f"{random.randint(0, 499):03d}"
            subfolder_path = self.clips_dir / subfolder
            
            if not subfolder_path.exists():
                attempts += 1
                continue
            
            files = [f for f in os.listdir(subfolder_path) if f.endswith('.wav')]
            if not files:
                attempts += 1
                continue
            
            needed = self.num_speakers - len(all_files)
            batch_samples = random.sample(files, min(needed, len(files)))
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
        selected_files = self.get_random_files()
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
            return (
                tf.zeros((1, self.samples_per_clip), dtype=tf.float32),
                tf.zeros((self.num_speakers, self.samples_per_clip), dtype=tf.float32),
                []
            )
        
        max_val = np.max(np.abs(mixed))
        if max_val > 0:
            mixed = mixed / max_val
        
        mixed_tensor = tf.convert_to_tensor([mixed], dtype=tf.float32)
        
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

        for mix_id in mixed_ids:
            X.append(wave_dict[mix_id].waveform)
        
        for source_ids in all_source_ids:
            source_list = []
            for source_id in source_ids:
                if source_id:
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

    dataset = tf.data.Dataset.from_generator(
        generator_fn,
        output_signature=(
            tf.TensorSpec(shape=(batch_size, mixer.samples_per_clip, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(batch_size, num_speakers, mixer.samples_per_clip, 1), dtype=tf.float32)
        )
    )

    return dataset.prefetch(tf.data.AUTOTUNE)

def create_tf_dataset_from_tfrecords(tfrecords_dir, num_speakers, batch_size=32, wavelet_level=5):
    """Create a TensorFlow dataset from TFRecord files containing audio data
    
    Args:
        tfrecords_dir: Directory containing TFRecord files
        num_speakers: Number of speakers to mix
        batch_size: Batch size for training
        wavelet_level: Wavelet decomposition level
        
    Returns:
        tf.data.Dataset: Dataset that yields (mixed_audio, separated_audio) pairs
    """
    tfrecord_files = tf.io.gfile.glob(f"{tfrecords_dir}/*.tfrecord")
    if not tfrecord_files:
        raise ValueError(f"No TFRecord files found in {tfrecords_dir}")
    samples_per_clip = 16000
    
    def _parse_tfrecord(example_proto):
        feature_description = {
            'audio_binary': tf.io.FixedLenFeature([], tf.string),
            'path': tf.io.FixedLenFeature([], tf.string)
        }
        parsed_features = tf.io.parse_single_example(example_proto, feature_description)
        audio_tensor = tf.audio.decode_wav(parsed_features['audio_binary'])
        waveform = audio_tensor.audio
        current_length = tf.shape(waveform)[0]
        
        if current_length < samples_per_clip:
            padding = [[0, samples_per_clip - current_length], [0, 0]]
            waveform = tf.pad(waveform, padding)
        else:
            waveform = waveform[:samples_per_clip]
        
        waveform = tf.reshape(waveform, (samples_per_clip, 1))
        
        return waveform

    def _prepare_batch(waveforms):
        batch_size = tf.shape(waveforms)[0]
        
        mixed_audio = tf.zeros((batch_size, samples_per_clip, 1), dtype=tf.float32)
        separated_audio = tf.zeros((batch_size, num_speakers, samples_per_clip, 1), dtype=tf.float32)
        
        weights = tf.random.uniform((batch_size, num_speakers), minval=0.5, maxval=1.5)
        weights = weights / tf.reduce_sum(weights, axis=1, keepdims=True)
        
        for i in range(num_speakers):
            indices = tf.random.uniform((batch_size,), 0, batch_size, dtype=tf.int32)
            selected_waveforms = tf.gather(waveforms, indices)
            
            batch_weights = tf.reshape(weights[:, i], (batch_size, 1, 1))
            weighted_waveforms = selected_waveforms * batch_weights
            
            mixed_audio += weighted_waveforms
            separated_audio = tf.tensor_scatter_nd_update(
                separated_audio,
                tf.stack([tf.range(batch_size), tf.fill((batch_size,), i)], axis=1),
                weighted_waveforms
            )
        
        return mixed_audio, separated_audio

    dataset = tf.data.TFRecordDataset(tfrecord_files, compression_type='GZIP')
    dataset = dataset.map(_parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(_prepare_batch, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset