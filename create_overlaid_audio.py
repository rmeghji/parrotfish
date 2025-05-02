import json
import os
import random
from pathlib import Path
import numpy as np
from moviepy.editor import VideoFileClip
from scipy.io import wavfile
from scipy.signal import windows
from tqdm import tqdm
import soundfile as sf
from google.colab import drive
import io
import tensorflow as tf
import time

class AudioMixerGenerator:
    def __init__(self, base_dir, clips_dir, num_speakers=2, batch_size=1000):
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
        import itertools
        
        all_files = []
        start = 0
        max_attempts = 10  # Prevent infinite loops
        attempts = 0
        
        while len(all_files) < self.num_speakers and attempts < max_attempts:
            # List only a subset of files
            files = list(itertools.islice(
                (f for f in os.listdir(self.clips_dir) if f.endswith('.wav')), 
                start, start + self.batch_size
            ))
            
            if not files:
                start = 0  # Reset if we hit the end
                attempts += 1
                continue
                
            # Add random files from this batch
            needed = self.num_speakers - len(all_files)
            batch_samples = random.sample(files, min(needed, len(files)))
            all_files.extend(batch_samples)
            start += self.batch_size
        
        if len(all_files) < self.num_speakers:
            print(f"Warning: Could only find {len(all_files)} valid files")
        
        return all_files
    
    def generate_sample(self):
        """Mix multiple 1-second clips together"""
        # Randomly select input files using batch processing
        selected_files = self.get_random_files()
        
        # Mix the clips
        mixed = np.zeros(self.samples_per_clip)
        count = 0
        
        for filename in selected_files:
            file_path = self.input_dir / filename
            audio = self.load_clip(file_path)
            
            if audio is not None:
                mixed += audio
                count += 1
        
        if count == 0:
            # Return zeros if no valid audio was found
            return tf.zeros((1, self.samples_per_clip), dtype=tf.float32)
        
        # Normalize the mixed audio
        mixed = mixed / np.max(np.abs(mixed))
        
        return tf.convert_to_tensor([mixed], dtype=tf.float32)

def create_tf_dataset(base_dir, clips_dir, num_speakers,
                     batch_size=32, buffer_size=1000):
    """Create a TensorFlow dataset that generates audio samples on-the-fly"""
    mixer = AudioMixerGenerator(
        base_dir,
        clips_dir,
        num_speakers=num_speakers
    )
    
    def generator_fn():
        while True:
            clips = mixer.generate_sample()
            for clip in clips:
                yield clip
    
    dataset = tf.data.Dataset.from_generator(
        generator_fn,
        output_signature=tf.TensorSpec(shape=(mixer.samples_per_clip,), dtype=tf.float32)
    )
    
    # Shuffle, batch, and prefetch for better performance
    return dataset.shuffle(buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)

class AudioOverlayGenerator:
    def __init__(self, base_dir, clips_dir):
        self.base_dir = Path(base_dir)
        self.clips_dir = Path(clips_dir)
    
    def save_to_drive(self, audio_clips, filename_prefix, mount_point="/content/drive/MyDrive/parrotfish/"):
        """Save audio clips directly to Google Drive"""
        drive.mount('/content/drive')
        os.makedirs(mount_point, exist_ok=True)
        
        # Save each clip with an index
        for i, clip in enumerate(audio_clips):
            output_path = Path(mount_point) / f"{filename_prefix}_clip_{i:03d}.wav"
            sf.write(str(output_path), clip.numpy(), 16000, format='WAV')
    
    def create_sample_dataset(self, num_clips_per_category=10):
        """Create a small sample dataset and save to Google Drive"""
        print("Generating sample dataset...")
        
        for num_speakers in [2, 3, 4]:
            generator = AudioMixerGenerator(
                self.base_dir,
                self.clips_dir,
                num_speakers=num_speakers
            )
            
            for i in tqdm(range(num_clips_per_category)):
                audio_clips = generator.generate_sample()
                self.save_to_drive(
                    audio_clips,
                    f"{num_speakers}_speakers_{i:04d}"
                )

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
        
        # Save clips
        base_name = Path(input_file).stem
        for i, clip in enumerate(clips):
            output_path = Path(output_dir) / f"{base_name}_clip_{i:03d}.wav"
            # sf.write(str(output_path), clip, 16000, format='OGG', subtype='VORBIS')
            sf.write(str(output_path), clip, 16000, format='WAV')

def process_drive_audio(base_folder, output_folder, clip_duration_seconds=1.0, window_overlap_ratio=0.1, batch_size=100):
    """Process all audio files in Google Drive folder and save all clips to a single output folder"""
    # Mount Google Drive if not already mounted
    if not os.path.exists('/content/drive'):
        drive.mount('/content/drive')
    
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
    
    def save_clip_with_retry(clip, output_path, max_retries=3, delay=1):
        """Try to save a clip with retries on failure"""
        for attempt in range(max_retries):
            try:
                sf.write(str(output_path), clip, 16000, format='WAV')
                return True
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(delay)
                    continue
                else:
                    print(f"Failed to save {output_path} after {max_retries} attempts: {str(e)}")
                    return False
    
    # Process files in batches to avoid overwhelming Google Drive
    total_clips = 0
    successful_files = 0
    batch_start = 0
    
    while batch_start < len(audio_files):
        batch_end = min(batch_start + batch_size, len(audio_files))
        batch = audio_files[batch_start:batch_end]
        
        print(f"\nProcessing batch {batch_start//batch_size + 1} ({batch_start}-{batch_end} of {len(audio_files)})")
        
        for audio_file in tqdm(batch, desc="Processing audio files"):
            try:
                # Load and normalize audio
                audio = processor.load_and_normalize_audio(audio_file)
                if audio is None:
                    continue
                
                # Split into clips
                clips = processor.split_into_clips(audio)
                
                # Save clips with a unique prefix based on original file
                file_prefix = Path(audio_file).stem
                
                # Track if all clips for this file were saved successfully
                file_success = True
                
                # Save all clips in the single output folder
                for i, clip in enumerate(clips):
                    output_path = Path(output_folder) / f"{file_prefix}_clip_{i:03d}.wav"
                    if save_clip_with_retry(clip, output_path):
                        total_clips += 1
                    else:
                        file_success = False
                
                if file_success:
                    successful_files += 1
                
            except Exception as e:
                print(f"Error processing {audio_file}: {str(e)}")
        
        # Add a small delay between batches to let Google Drive catch up
        if batch_end < len(audio_files):
            print("Waiting 5 seconds before next batch...")
            time.sleep(5)
        
        batch_start = batch_end
    
    print(f"\nAudio segmentation complete!")
    print(f"Successfully processed {successful_files}/{len(audio_files)} files")
    print(f"Generated {total_clips} clips in {output_folder}")

def main():
    base_dir = Path(__file__).parent
    clips_dir = base_dir / "casualconversations" / "clips"
    
    # Example 1: Process existing audio files in Google Drive
    base_folder = '/content/drive/MyDrive/parrotfish/data/casual_conversations'
    output_folder = '/content/drive/MyDrive/parrotfish/data/casual_conversations/clips'
    process_drive_audio(base_folder, output_folder, clip_duration_seconds=1.0, window_overlap_ratio=0.1)
    
    # Example 2: Generate and save new mixed audio clips
    mixer = AudioMixerGenerator(
        base_dir, 
        clips_dir, 
        num_speakers=2
    )
    audio_clips = mixer.generate_sample()
    
    generator = AudioOverlayGenerator(
        base_dir, 
        clips_dir
    )
    generator.save_to_drive(audio_clips, "example_2speakers")

if __name__ == "__main__":
    main()
