import json
import os
import random
from pathlib import Path
import numpy as np
# from moviepy.editor import VideoFileClip
from scipy.io import wavfile
from scipy.signal import windows
from tqdm import tqdm
import soundfile as sf
# from google.colab import drive
import io
import tensorflow as tf
import time
from Wavelets import getWaveletTransform, makeWaveDictBatch

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
        print("Getting random files...")
        
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
        mixed = mixed / np.max(np.abs(mixed))
        
        # Convert to tensors
        mixed_tensor = tf.convert_to_tensor([mixed], dtype=tf.float32)
        
        # Pad input_clips if we got fewer clips than num_speakers
        while len(input_clips) < self.num_speakers:
            input_clips.append(np.zeros(self.samples_per_clip))
            source_ids.append('')
        
        sources_tensor = tf.convert_to_tensor(input_clips, dtype=tf.float32)
        
        return mixed_tensor, sources_tensor, source_ids

    def batch_training_data(self, batch_size: int = 32, wavelet_level: int = 5):
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

        # for mix_id in mixed_ids:
            # getWaveletTransform(wave_dict, mix_id, wavelet_level)
            
            # for source_id in wave_dict[mix_id].source_ids:
            #     getWaveletTransform(wave_dict, source_id, wavelet_level)
            
            # mixed_coeffs = wave_dict[mix_id].tensor_coeffs
            
            # source_coeffs = []
            # for source_id in wave_dict[mix_id].source_ids:
            #     source_coeffs.append(wave_dict[source_id].tensor_coeffs)
                
            # X.append(mixed_coeffs)
            # Y.append(source_coeffs)        

        for mix_id in mixed_ids:
            X.append(wave_dict[mix_id].waveform)
        
        for source_ids in all_source_ids:
            source_list = []
            for source_id in source_ids:
                source_list.append(wave_dict[source_id].waveform)
            Y.append(source_list)

        X = tf.convert_to_tensor(X)
        Y = tf.convert_to_tensor(Y)

        return X, Y

def create_tf_dataset(base_dir, clips_dir, num_speakers,
                     batch_size=32):
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

    wavelet_level = 5
    approx_feature_size = mixer.samples_per_clip // (2 ** wavelet_level)

    # for wavelet version
    # dataset = tf.data.Dataset.from_generator(
    #     generator_fn,
    #     output_signature=(
    #         tf.TensorSpec(shape=(batch_size, wavelet_level + 1, approx_feature_size), dtype=tf.float32),
    #         tf.TensorSpec(shape=(batch_size, num_speakers, wavelet_level + 1, approx_feature_size), dtype=tf.float32)
    #     )
    # )

    dataset = tf.data.Dataset.from_generator(
        generator_fn,
        output_signature=(
            tf.TensorSpec(shape=(batch_size, mixer.samples_per_clip), dtype=tf.float32),
            tf.TensorSpec(shape=(batch_size, num_speakers, mixer.samples_per_clip), dtype=tf.float32)
        )
    )

    return dataset.prefetch(tf.data.AUTOTUNE)

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
            sample_rate, audio_array = wavfile.read(str(file_path))
            if sample_rate != 16000:
                raise ValueError(f"Audio file {file_path} must be 16kHz (found {sample_rate}Hz)")
            
            # Convert to float32 if needed
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)
                if audio_array.max() > 1.0:
                    audio_array = audio_array / 32768.0  # Normalize 16-bit integer
            
            # Convert to mono if stereo
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)
            
            # Normalize
            audio_array = audio_array / np.max(np.abs(audio_array))
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

def process_drive_audio(base_folder, output_folder, clip_duration_seconds=1.0, window_overlap_ratio=0.1, batch_size=100):
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