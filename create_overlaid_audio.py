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
    def __init__(self, clips_dir, num_speakers=2, clip_duration_seconds=1.0, window_overlap_ratio=0.1):
        """Initialize the generator with a directory of clips"""
        self.clips_dir = Path(clips_dir)
        if not self.clips_dir.exists():
            raise ValueError(f"Clips directory {clips_dir} does not exist")
        
        self.num_speakers = num_speakers
        self.clip_duration_seconds = clip_duration_seconds
        self.window_overlap_ratio = window_overlap_ratio
        self.samples_per_clip = int(clip_duration_seconds * 16000)
        self.overlap_samples = int(self.samples_per_clip * window_overlap_ratio)
        
        # Get all WAV files in the directory
        self.clip_files = []
        for file in self.clips_dir.glob("*.wav"):
            self.clip_files.append(file)
        
        if not self.clip_files:
            raise ValueError(f"No WAV files found in {clips_dir}")
        
        print(f"Found {len(self.clip_files)} clips in {clips_dir}")
    
    def load_audio(self, file_path):
        """Load and normalize a WAV file"""
        try:
            sample_rate, audio = wavfile.read(str(file_path))
            if sample_rate != 16000:
                raise ValueError(f"Audio file {file_path} must be 16kHz (found {sample_rate}Hz)")
            
            # Convert to float32 if needed
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
                if audio.max() > 1.0:
                    audio = audio / 32768.0  # Normalize 16-bit integer
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            # Normalize
            audio = audio / np.max(np.abs(audio))
            return audio
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
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
    
    def generate_sample(self):
        """Generate a mixed audio sample by overlaying random clips"""
        # Randomly select clips to mix
        selected_clips = np.random.choice(self.clip_files, size=self.num_speakers, replace=False)
        
        # Initialize mixed audio
        mixed = np.zeros(self.samples_per_clip)
        count = 0
        
        # Load and mix clips
        for clip_file in selected_clips:
            audio = self.load_audio(clip_file)
            if audio is None:
                continue
            
            # If clip is longer than needed, take a random segment
            if len(audio) > self.samples_per_clip:
                start = np.random.randint(0, len(audio) - self.samples_per_clip)
                audio = audio[start:start + self.samples_per_clip]
            else:
                # Pad shorter clips with zeros
                audio = np.pad(audio, (0, max(0, self.samples_per_clip - len(audio))))
            
            mixed += audio
            count += 1
        
        if count == 0:
            # Return zeros if no valid audio was found
            return tf.zeros((1, self.samples_per_clip), dtype=tf.float32)
        
        # Normalize the mixed audio
        mixed = mixed / np.max(np.abs(mixed))
        
        # Split into clips
        clips = self.split_into_clips(mixed)
        return tf.convert_to_tensor(clips, dtype=tf.float32)

def create_tf_dataset(clips_dir, num_speakers, batch_size=32, buffer_size=1000, 
                     clip_duration_seconds=1.0, window_overlap_ratio=0.1):
    """Create a TensorFlow dataset that generates mixed audio samples on-the-fly"""
    generator = AudioMixerGenerator(
        clips_dir,
        num_speakers=num_speakers,
        clip_duration_seconds=clip_duration_seconds,
        window_overlap_ratio=window_overlap_ratio
    )
    
    def generator_fn():
        while True:
            clips = generator.generate_sample()
            for clip in clips:
                yield clip
    
    dataset = tf.data.Dataset.from_generator(
        generator_fn,
        output_signature=tf.TensorSpec(shape=(generator.samples_per_clip,), dtype=tf.float32)
    )
    
    # Shuffle, batch, and prefetch for better performance
    return dataset.shuffle(buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)

class AudioOverlayGenerator:
    def __init__(self, clips_dir):
        self.clips_dir = Path(clips_dir)
    
    def save_to_drive(self, audio_clips, filename_prefix, mount_point="/content/drive/MyDrive/parrotfish/"):
        """Save audio clips directly to Google Drive"""
        drive.mount('/content/drive')
        os.makedirs(mount_point, exist_ok=True)
        
        # Save each clip with an index
        for i, clip in enumerate(audio_clips):
            output_path = Path(mount_point) / f"{filename_prefix}_clip_{i:03d}.wav"
            sf.write(str(output_path), clip.numpy(), 16000, format='WAV')
    
def main():
    # Example usage with directory of clips
    clips_dir = '/content/drive/MyDrive/parrotfish/data/casual_conversations/clips'
    
    # Create generator and mix some clips
    mixer = AudioMixerGenerator(clips_dir, num_speakers=2)
    mixed_clips = mixer.generate_sample()
    
    # Create a TensorFlow dataset
    dataset = create_tf_dataset(clips_dir, num_speakers=2, batch_size=32)
    
    # Save some example mixed clips
    generator = AudioOverlayGenerator(clips_dir)
    generator.save_to_drive(mixed_clips, "example_2speakers")

if __name__ == "__main__":
    main()
