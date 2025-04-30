import json
import os
import random
from pathlib import Path
import numpy as np
from moviepy.editor import VideoFileClip
from scipy.io import wavfile
from tqdm import tqdm
import soundfile as sf
from google.colab import drive
import io
import tensorflow as tf

class AudioMixerGenerator:
    def __init__(self, base_dir, annotations_file, activity_file, num_speakers=2, max_duration_seconds=10):
        self.base_dir = Path(base_dir)
        self.video_dir = self.base_dir / "videos"
        self.num_speakers = num_speakers
        self.max_duration_seconds = max_duration_seconds
        self.max_samples = int(max_duration_seconds * 16000)
        
        # Load annotations
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        with open(activity_file, 'r') as f:
            self.activities = json.load(f)
            
        self.video_files = [item["video_name"] for item in self.activities]
    
    def video_to_audio(self, video_path):
        """Convert video to 16kHz audio and return as numpy array"""
        try:
            video = VideoFileClip(str(video_path))
            audio = video.audio
            if audio is None:
                return None
            
            # Extract audio data and resample to 16kHz
            audio_array = audio.to_soundarray(fps=16000)
            
            # Convert to mono if stereo
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)
            
            # Normalize
            audio_array = audio_array / np.max(np.abs(audio_array))
            
            video.close()
            return audio_array
            
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            return None
    
    def generate_sample(self):
        """Generate a single mixed audio sample"""
        # Randomly select videos
        selected_videos = random.sample(self.video_files, self.num_speakers)
        
        # Convert videos to audio and mix
        mixed = np.zeros(self.max_samples)
        count = 0
        
        for video_name in selected_videos:
            video_path = self.video_dir / video_name
            audio = self.video_to_audio(video_path)
            
            if audio is not None:
                # Trim or pad to max duration
                if len(audio) > self.max_samples:
                    start = random.randint(0, len(audio) - self.max_samples)
                    audio = audio[start:start + self.max_samples]
                else:
                    audio = np.pad(audio, (0, self.max_samples - len(audio)))
                
                mixed += audio
                count += 1
        
        if count == 0:
            # Return zeros if no valid audio was found
            return tf.zeros(self.max_samples, dtype=tf.float32)
        
        # Normalize the mixed audio
        mixed = mixed / np.max(np.abs(mixed))
        return tf.convert_to_tensor(mixed, dtype=tf.float32)

def create_tf_dataset(base_dir, annotations_file, activity_file, num_speakers,
                     batch_size=32, max_duration_seconds=10, buffer_size=1000):
    """Create a TensorFlow dataset that generates audio samples on-the-fly"""
    generator = AudioMixerGenerator(
        base_dir,
        annotations_file,
        activity_file,
        num_speakers=num_speakers,
        max_duration_seconds=max_duration_seconds
    )
    
    def generator_fn():
        while True:
            yield generator.generate_sample()
    
    dataset = tf.data.Dataset.from_generator(
        generator_fn,
        output_signature=tf.TensorSpec(shape=(generator.max_samples,), dtype=tf.float32)
    )
    
    # Shuffle, batch, and prefetch for better performance
    return dataset.shuffle(buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)

class AudioOverlayGenerator:
    def __init__(self, base_dir, annotations_file, activity_file):
        self.base_dir = Path(base_dir)
        self.video_dir = self.base_dir / "videos"
        self.annotations_file = annotations_file
        self.activity_file = activity_file
    
    def save_to_drive(self, audio_data, filename, mount_point="/content/drive/MyDrive/audio_data"):
        """Save audio data directly to Google Drive"""
        drive.mount('/content/drive')
        os.makedirs(mount_point, exist_ok=True)
        
        # Save as compressed OGG file
        output_path = Path(mount_point) / filename.replace('.wav', '.ogg')
        sf.write(str(output_path), audio_data.numpy(), 16000, format='OGG', subtype='VORBIS')
    
    def create_sample_dataset(self, num_clips_per_category=10, max_duration_seconds=10):
        """Create a small sample dataset and save to Google Drive"""
        print("Generating sample dataset...")
        
        for num_speakers in [2, 3, 4]:
            generator = AudioMixerGenerator(
                self.base_dir,
                self.annotations_file,
                self.activity_file,
                num_speakers=num_speakers,
                max_duration_seconds=max_duration_seconds
            )
            
            for i in tqdm(range(num_clips_per_category)):
                audio = generator.generate_sample()
                self.save_to_drive(
                    audio,
                    f"{num_speakers}_speakers_{i:04d}.ogg"
                )

def main():
    base_dir = Path(__file__).parent
    annotations_file = base_dir / "casualconversations" / "CasualConversationsV2.json"
    activity_file = base_dir / "casualconversations" / "CasualConversationsV2_activity.json"
    
    # Example: Create a small sample dataset in Google Drive
    generator = AudioOverlayGenerator(base_dir, annotations_file, activity_file)
    generator.create_sample_dataset(num_clips_per_category=5)
    
    # Example: Create a streaming dataset for training
    dataset = create_tf_dataset(
        base_dir,
        annotations_file,
        activity_file,
        num_speakers=2,
        batch_size=32
    )
    
    # You can use this dataset in your training loop
    for batch in dataset.take(1):  # Example: take 1 batch
        # batch is a tensor of shape [batch_size, max_samples]
        # Use this for training
        print(f"Batch shape: {batch.shape}")

if __name__ == "__main__":
    main()
