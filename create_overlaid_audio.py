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

class AudioMixerGenerator:
    def __init__(self, base_dir, annotations_file, activity_file, num_speakers=2, 
                 max_duration_seconds=10, input_type='video', clip_duration_seconds=1.0,
                 window_overlap_ratio=0.1):
        """
        Initialize the AudioMixerGenerator.
        
        Args:
            base_dir: Base directory containing the data
            annotations_file: Path to annotations JSON file
            activity_file: Path to activity JSON file
            num_speakers: Number of speakers to mix together
            max_duration_seconds: Maximum duration of audio clips
            input_type: Type of input files ('video' or 'wav')
            clip_duration_seconds: Duration of each output clip in seconds
            window_overlap_ratio: Ratio of overlap between adjacent windows (0.0 to 0.5)
        """
        self.base_dir = Path(base_dir)
        self.input_type = input_type.lower()
        if self.input_type not in ['video', 'wav']:
            raise ValueError("input_type must be either 'video' or 'wav'")
            
        # Set input directory based on type
        self.input_dir = self.base_dir / ('videos' if self.input_type == 'video' else 'wavs')
        self.num_speakers = num_speakers
        self.max_duration_seconds = max_duration_seconds
        self.max_samples = int(max_duration_seconds * 16000)
        
        # Clip and window parameters
        self.clip_duration_seconds = clip_duration_seconds
        self.samples_per_clip = int(clip_duration_seconds * 16000)
        self.window_overlap_ratio = window_overlap_ratio
        self.overlap_samples = int(self.samples_per_clip * window_overlap_ratio)
        
        # Create Hann window for smooth transitions
        self.window = windows.hann(self.samples_per_clip)
        
        # Load annotations
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        with open(activity_file, 'r') as f:
            self.activities = json.load(f)
            
        # Get input files based on type
        if self.input_type == 'video':
            self.input_files = [item["video_name"] for item in self.activities]
        else:
            # For WAV files, replace video extensions with .wav
            self.input_files = [item["video_name"].rsplit('.', 1)[0] + '.wav' 
                              for item in self.activities]
    
    def load_audio(self, file_path):
        """Load audio from either video or WAV file"""
        try:
            if self.input_type == 'video':
                video = VideoFileClip(str(file_path))
                audio = video.audio
                if audio is None:
                    return None
                
                # Extract audio data and resample to 16kHz
                audio_array = audio.to_soundarray(fps=16000)
                
                # Convert to mono if stereo
                if len(audio_array.shape) > 1:
                    audio_array = audio_array.mean(axis=1)
                    
                video.close()
                
            else:  # WAV file
                # Read WAV file (assuming it's already 16kHz)
                sample_rate, audio_array = wavfile.read(str(file_path))
                if sample_rate != 16000:
                    raise ValueError(f"WAV file {file_path} must be 16kHz (found {sample_rate}Hz)")
                
                # Convert to float32 if needed
                if audio_array.dtype != np.float32:
                    audio_array = audio_array.astype(np.float32)
                    if audio_array.max() > 1.0:
                        audio_array = audio_array / 32768.0  # Normalize 16-bit integer
            
            # Normalize
            audio_array = audio_array / np.max(np.abs(audio_array))
            return audio_array
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None
    
    def split_into_clips(self, audio):
        """Split audio into overlapping clips with windowing"""
        # Calculate number of clips with overlap
        hop_size = self.samples_per_clip - self.overlap_samples
        num_clips = max(1, (len(audio) - self.samples_per_clip) // hop_size + 1)
        clips = []
        
        for i in range(num_clips):
            start = i * hop_size
            end = start + self.samples_per_clip
            
            if end > len(audio):
                # Pad the last clip if needed
                clip = np.pad(audio[start:], (0, end - len(audio)))
            else:
                clip = audio[start:end]
            
            # Apply windowing
            clip = clip * self.window
            clips.append(clip)
        
        return np.stack(clips) if clips else np.zeros((1, self.samples_per_clip))
    
    def generate_sample(self):
        """Generate a single mixed audio sample and split into clips"""
        # Randomly select input files
        selected_files = random.sample(self.input_files, self.num_speakers)
        
        # Convert to audio and mix
        mixed = np.zeros(self.max_samples)
        count = 0
        
        for filename in selected_files:
            file_path = self.input_dir / filename
            audio = self.load_audio(file_path)
            
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
            return tf.zeros((1, self.samples_per_clip), dtype=tf.float32)
        
        # Normalize the mixed audio
        mixed = mixed / np.max(np.abs(mixed))
        
        # Split into clips with windowing
        clips = self.split_into_clips(mixed)
        return tf.convert_to_tensor(clips, dtype=tf.float32)

def create_tf_dataset(base_dir, annotations_file, activity_file, num_speakers,
                     batch_size=32, max_duration_seconds=10, buffer_size=1000, 
                     input_type='video', clip_duration_seconds=1.0, window_overlap_ratio=0.1):
    """Create a TensorFlow dataset that generates audio samples on-the-fly"""
    generator = AudioMixerGenerator(
        base_dir,
        annotations_file,
        activity_file,
        num_speakers=num_speakers,
        max_duration_seconds=max_duration_seconds,
        input_type=input_type,
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
    def __init__(self, base_dir, annotations_file, activity_file, input_type='video',
                 clip_duration_seconds=1.0, window_overlap_ratio=0.1):
        self.base_dir = Path(base_dir)
        self.input_type = input_type.lower()
        if self.input_type not in ['video', 'wav']:
            raise ValueError("input_type must be either 'video' or 'wav'")
        self.input_dir = self.base_dir / ('videos' if self.input_type == 'video' else 'wavs')
        self.annotations_file = annotations_file
        self.activity_file = activity_file
        self.clip_duration_seconds = clip_duration_seconds
        self.window_overlap_ratio = window_overlap_ratio
    
    def save_to_drive(self, audio_clips, filename_prefix, mount_point="/content/drive/MyDrive/parrotfish/"):
        """Save audio clips directly to Google Drive"""
        drive.mount('/content/drive')
        os.makedirs(mount_point, exist_ok=True)
        
        # Save each clip with an index
        for i, clip in enumerate(audio_clips):
            output_path = Path(mount_point) / f"{filename_prefix}_clip_{i:03d}.ogg"
            sf.write(str(output_path), clip.numpy(), 16000, format='OGG', subtype='VORBIS')
    
    def create_sample_dataset(self, num_clips_per_category=10, max_duration_seconds=10):
        """Create a small sample dataset and save to Google Drive"""
        print("Generating sample dataset...")
        
        for num_speakers in [2, 3, 4]:
            generator = AudioMixerGenerator(
                self.base_dir,
                self.annotations_file,
                self.activity_file,
                num_speakers=num_speakers,
                max_duration_seconds=max_duration_seconds,
                input_type=self.input_type,
                clip_duration_seconds=self.clip_duration_seconds,
                window_overlap_ratio=self.window_overlap_ratio
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
        """Split audio into overlapping clips with windowing"""
        # Calculate number of clips with overlap
        hop_size = self.samples_per_clip - self.overlap_samples
        num_clips = max(1, (len(audio) - self.samples_per_clip) // hop_size + 1)
        clips = []
        
        for i in range(num_clips):
            start = i * hop_size
            end = start + self.samples_per_clip
            
            if end > len(audio):
                # Pad the last clip if needed
                clip = np.pad(audio[start:], (0, end - len(audio)))
            else:
                clip = audio[start:end]
            
            # Apply windowing
            clip = clip * self.window
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
            output_path = Path(output_dir) / f"{base_name}_clip_{i:03d}.ogg"
            sf.write(str(output_path), clip, 16000, format='OGG', subtype='VORBIS')

def process_drive_audio(base_folder, output_folder, clip_duration_seconds=1.0, window_overlap_ratio=0.1):
    """Process all audio files in Google Drive folder"""
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
    
    # Get all audio files
    audio_files = []
    for dirpath, dirnames, filenames in os.walk(base_folder):
        for filename in filenames:
            if filename.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a')):
                audio_files.append(os.path.join(dirpath, filename))
    
    print(f"Found {len(audio_files)} audio files to process")
    
    # Process each audio file
    for audio_file in tqdm(audio_files, desc="Processing audio files"):
        try:
            # Get relative path to maintain folder structure
            rel_path = os.path.relpath(audio_file, base_folder)
            file_output_dir = os.path.join(output_folder, os.path.dirname(rel_path))
            
            # Create output directory for this file
            os.makedirs(file_output_dir, exist_ok=True)
            
            # Process the file
            processor.segment_audio(audio_file, file_output_dir)
            
        except Exception as e:
            print(f"Error processing {audio_file}: {str(e)}")
    
    print("Audio segmentation complete!")

def main():
    base_dir = Path(__file__).parent
    annotations_file = base_dir / "casualconversations" / "CasualConversationsV2.json"
    activity_file = base_dir / "casualconversations" / "CasualConversationsV2_activity.json"
    
    # Example 1: Process existing audio files in Google Drive
    base_folder = '/content/drive/MyDrive/parrotfish/data/casual_conversations'
    output_folder = '/content/drive/MyDrive/parrotfish/data/casual_conversations/clips'
    process_drive_audio(base_folder, output_folder, clip_duration_seconds=1.0, window_overlap_ratio=0.1)
    
    # Example 2: Generate and save new mixed audio clips
    input_type = 'wav'
    mixer = AudioMixerGenerator(
        base_dir, 
        annotations_file, 
        activity_file, 
        num_speakers=2, 
        input_type=input_type,
        clip_duration_seconds=1.0,
        window_overlap_ratio=0.1
    )
    audio_clips = mixer.generate_sample()
    
    generator = AudioOverlayGenerator(
        base_dir, 
        annotations_file, 
        activity_file, 
        input_type=input_type,
        clip_duration_seconds=1.0,
        window_overlap_ratio=0.1
    )
    generator.save_to_drive(audio_clips, "example_2speakers")

if __name__ == "__main__":
    main()
