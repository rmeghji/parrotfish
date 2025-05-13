import os
import numpy as np
from scipy.io import wavfile
import tensorflow as tf
import random
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from itertools import combinations, permutations
import soundfile as sf
import librosa
import pywt
from scipy.signal import windows
from scipy import optimize

from utils.Pipeline import (
    create_tf_dataset,
    create_tf_dataset_from_tfrecords,
    generate_sample_from_clips
)
from utils.config import Config, RetrainConfig
from model import (
    WaveletUNet,
    pit_loss,
    gelu,
    DWTLayer,
    IDWTLayer,
    DownsamplingLayer,
    UpsamplingLayer,
    GatedSkipConnection,
)
from utils.data_preparation import process_audio_for_prediction, reconstruct_audio_from_clips
from pydub import AudioSegment

config = RetrainConfig()

def load_saved_model(model_dir, filename):
    """Load the model with multiple fallback options."""
    print(f"Attempting to load model from {model_dir}")

    custom_objects = {
        'WaveletUNet': WaveletUNet,
        'DWTLayer': DWTLayer,
        'IDWTLayer': IDWTLayer,
        'DownsamplingLayer': DownsamplingLayer,
        'UpsamplingLayer': UpsamplingLayer,
        'GatedSkipConnection': GatedSkipConnection,
        'pit_loss': pit_loss,
        'gelu': gelu
    }

    loaded_model = tf.keras.models.load_model(os.path.join(model_dir, f"{filename}.keras"), custom_objects=custom_objects)
    loaded_model(tf.random.normal(shape=(config.BATCH_SIZE, config.SEGMENT_LENGTH, 1))) # dummy input to build model
    loaded_model.load_weights(os.path.join(model_dir, f"{filename}_weightsonly.weights.h5"))

    return loaded_model

def evaluate_model(model, test_generator, num_examples=5):
    """Evaluate the model and visualize separation results."""
    X_test, y_test = test_generator.__getitem__(0)
    indices = np.random.choice(X_test.shape[0], num_examples, replace=False)
    y_pred = model.predict(X_test[indices])
    sdrs = []
    
    for i in range(num_examples):
        example_sdrs = []
        for j in range(config.MAX_SOURCES):
            source_energy = np.sum(np.abs(y_test[indices[i], j]))
            
            if source_energy > 1e-6:
                target = y_test[indices[i], j, :, 0]
                estimate = y_pred[i, j, :, 0]
                
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
    
    if sdrs:
        avg_sdr = np.mean(sdrs)
        print(f"Overall Average SDR: {avg_sdr:.2f} dB")
    
    plt.figure(figsize=(20, 4 * num_examples))
    
    for i in range(num_examples):
        plt.subplot(num_examples, config.MAX_SOURCES + 2, i * (config.MAX_SOURCES + 2) + 1)
        plt.plot(X_test[indices[i], :, 0])
        plt.title(f"Example {i+1} - Mixture")
        plt.ylim([-1, 1])
        
        for j in range(config.MAX_SOURCES):
            plt.subplot(num_examples, config.MAX_SOURCES + 2, i * (config.MAX_SOURCES + 2) + j + 2)
            plt.plot(y_test[indices[i], j, :, 0])
            plt.title(f"True Source {j+1}")
            plt.ylim([-1, 1])
            
            plt.subplot(num_examples, config.MAX_SOURCES + 2, i * (config.MAX_SOURCES + 2) + config.MAX_SOURCES + 2)
            plt.plot(y_pred[i, j, :, 0])
            plt.title(f"Pred Source {j+1}")
            plt.ylim([-1, 1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.CHECKPOINT_DIR, 'evaluation_results.png'))
    plt.show()
    
    return sdrs

def save_model(model, config, save_directory):
    print("Saving trained model...")
    model_save_path = os.path.join(config.CHECKPOINT_DIR, 'wavelet_unet_final.keras')
    model_save_path2 = os.path.join(save_directory, 'wavelet_unet_final.keras')
    
    model_json = model.to_json()
    with open(os.path.join(config.CHECKPOINT_DIR, 'model_architecture.json'), 'w') as json_file:
        json_file.write(model_json)
    
    with open(os.path.join(save_directory, 'model_architecture.json'), 'w') as json_file:
        json_file.write(model_json)
    
    model.save_weights(os.path.join(config.CHECKPOINT_DIR, 'model.weights.h5'))
    model.save_weights(os.path.join(save_directory, 'model.weights.h5'))
    
    try:
        saved_model_path = os.path.join(config.CHECKPOINT_DIR, 'wavelet_unet_savedmodel')
        tf.saved_model.save(model, saved_model_path)
        saved_model_path2 = os.path.join(save_directory, 'wavelet_unet_savedmodel')
        tf.saved_model.save(model, saved_model_path2)
        print(f"Model successfully saved to {saved_model_path}")
        print(f"Model successfully saved to {saved_model_path2}")
    except Exception as e:
        print(f"Error saving in SavedModel format: {e}")
        print("Falling back to H5 format only")
    
    try:
        model.save(model_save_path, save_format='h5')
        model.save(model_save_path2, save_format='h5')
        print(f"Model successfully saved to {model_save_path}")
        print(f"Model successfully saved to {model_save_path2}")
    except Exception as e:
        print(f"Error saving in H5 format: {e}")
        print("Model is saved as architecture + weights only")

def plot_model(history, config, save_directory):
    print("Plotting training history...")
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
    plt.savefig(os.path.join(save_directory, 'training_history.png'))
    
    print("Wavelet U-Net pipeline completed successfully!")

def test_separation(model, audio_file, output_dir="separated"):
    """Test the model on a new audio file"""
    os.makedirs(output_dir, exist_ok=True)
    audio, sample_rate = sf.read(audio_file)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    if sample_rate != 16000:
        print(f"Warning: Audio file sample rate is {sample_rate}Hz (expected 16kHz)")
    audio = audio / np.max(np.abs(audio))
    chunk_size = 16000
    num_chunks = len(audio) // chunk_size
    
    if len(audio) < chunk_size:
        audio = np.pad(audio, (0, chunk_size - len(audio)))
        num_chunks = 1
        print(f"Audio file is shorter than 1 second, padding with zeros")
    
    separated_sources = []
    for i in range(Config.MAX_SOURCES):
        separated_sources.append(np.zeros(len(audio)))
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        chunk = audio[start_idx:end_idx]
        chunk_input = chunk.reshape(1, chunk_size, 1)
        predictions = model.predict(chunk_input)
        for j in range(Config.MAX_SOURCES):
            source_chunk = predictions[0, j, :]
            separated_sources[j][start_idx:end_idx] = source_chunk
    
    for i, source in enumerate(separated_sources):
        output_path = os.path.join(output_dir, f"source_{i+1}.wav")
        sf.write(output_path, source, 16000)
    
    print(f"Separated sources saved to {output_dir}")

def separate_audio(model, audio, clip_duration_seconds=1.0, window_overlap_ratio=0.25):
    """Separate audio into sources using overlapping windows for better reconstruction.
    
    Args:
        model: The trained separation model
        audio: Input audio array
        clip_duration_seconds: Duration of each clip in seconds
        window_overlap_ratio: Overlap ratio between consecutive windows
        
    Returns:
        list: List of separated source arrays
    """
    clips, _ = process_audio_for_prediction(audio, clip_duration_seconds, window_overlap_ratio)
    separated_sources = [np.zeros((len(clips), clips.shape[1])) for _ in range(Config.MAX_SOURCES)]
    
    for i, clip in enumerate(clips):
        clip_input = clip.reshape(1, -1, 1)
        predictions = model.predict(clip_input)
        for j in range(Config.MAX_SOURCES):
            separated_sources[j][i] = predictions[0, j, :]
    
    reconstructed_sources = [
        reconstruct_audio_from_clips(source, clip_duration_seconds, window_overlap_ratio)
        for source in separated_sources
    ]
    
    return reconstructed_sources

def correlation_score(x, y, mode='pearson'):
    """
    Calculate correlation between two signals.
    
    Args:
        x: First signal
        y: Second signal
        mode: Type of correlation ('pearson', 'energy', 'cosine')
        
    Returns:
        float: Correlation score
    """
    # Ensure input arrays are 1D
    x = x.flatten()
    y = y.flatten()
    
    if mode == 'pearson':
        # Pearson correlation coefficient
        x_centered = x - np.mean(x)
        y_centered = y - np.mean(y)
        numerator = np.sum(x_centered * y_centered)
        denominator = np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2))
        if denominator < 1e-10:
            return 0
        return numerator / denominator
    
    elif mode == 'energy':
        # Energy-based correlation (useful for audio with silence)
        energy_x = np.sum(x**2)
        energy_y = np.sum(y**2)
        if energy_x < 1e-10 or energy_y < 1e-10:
            return 0
        return np.sum(x * y) / np.sqrt(energy_x * energy_y)
    
    elif mode == 'cosine':
        # Cosine similarity
        norm_x = np.linalg.norm(x)
        norm_y = np.linalg.norm(y)
        if norm_x < 1e-10 or norm_y < 1e-10:
            return 0
        return np.dot(x, y) / (norm_x * norm_y)
    
    else:
        raise ValueError(f"Unknown correlation mode: {mode}")

def find_optimal_source_assignment(current_sources, next_sources, overlap_size):
    """
    Find the optimal assignment of sources in the next frame to match the current frame sources.
    
    Args:
        current_sources: numpy array of shape (num_sources, samples_per_clip)
        next_sources: numpy array of shape (num_sources, samples_per_clip)
        overlap_size: Number of samples in the overlap region
        
    Returns:
        list: Optimal permutation of source indices
    """
    num_sources = current_sources.shape[0]
    
    # Calculate correlation matrix
    correlation_matrix = np.zeros((num_sources, num_sources))
    
    for i in range(num_sources):
        for j in range(num_sources):
            # Calculate correlation in the overlap region
            current_overlap = current_sources[i, -overlap_size:]
            next_overlap = next_sources[j, :overlap_size]
                  
            correlation_matrix[i, j] = correlation_score(current_overlap, next_overlap, mode='energy')
    
    # Hungarian algorithm to find optimal assignment that maximizes correlation
    row_ind, col_ind = optimize.linear_sum_assignment(-correlation_matrix)
    
    # Return the optimal permutation
    return col_ind

def separate_audio_with_consistent_tracking(model, audio, clip_duration_seconds=1.0, 
                                          window_overlap_ratio=0.25, sample_rate=22050):
    """
    Separate audio into sources using overlapping windows with consistent source tracking.
    
    Args:
        model: The trained separation model
        audio: Input audio array or path
        clip_duration_seconds: Duration of each clip in seconds
        window_overlap_ratio: Overlap ratio between consecutive windows
        sample_rate: Audio sample rate
        
    Returns:
        list: List of separated source arrays
    """
    # Process audio into overlapping clips
    clips, audio_data = process_audio_for_prediction(
        audio, clip_duration_seconds, window_overlap_ratio, sample_rate
    )
    
    num_clips = clips.shape[0]
    samples_per_clip = clips.shape[1]
    step_size = int(samples_per_clip * (1 - window_overlap_ratio))
    overlap_size = samples_per_clip - step_size
    
    # Predict sources for all clips
    all_predictions = []
    for i, clip in enumerate(clips):
        clip_input = clip.reshape(1, -1, 1)  # Adjust shape based on your model input requirements
        predictions = model.predict(clip_input, verbose=0)
        
        # Extract predictions - adjust based on your model's output format
        if isinstance(predictions, list):
            # Multiple output model
            predictions = predictions[0]  # Assuming first output is the source separation
        
        # Reshape if needed - adjust based on your model's output format
        if len(predictions.shape) == 3:
            # Shape: (batch, time, sources)
            predictions = np.transpose(predictions[0], (1, 0))  # to (sources, time)
        elif len(predictions.shape) == 2:
            # Shape: (batch, time*sources)
            predictions = predictions[0].reshape(Config.MAX_SOURCES, -1)
        
        all_predictions.append(predictions)
    
    # Initialize source order mapping for consistent tracking
    source_order = np.arange(Config.MAX_SOURCES)
    
    # Apply consistent source tracking
    aligned_predictions = []
    aligned_predictions.append(all_predictions[0][source_order])
    
    for i in range(1, num_clips):
        # Find optimal mapping between current and next frame sources
        optimal_perm = find_optimal_source_assignment(
            aligned_predictions[-1], all_predictions[i], overlap_size
        )
        
        # Update source order based on optimal permutation
        source_order = optimal_perm
        
        # Apply the source order mapping
        aligned_predictions.append(all_predictions[i][source_order])
    
    # Reconstruct full audio for each source
    reconstructed_sources = []
    print(aligned_predictions.shape)    
    
    
    # Create window function for smooth blending
    window = windows.hann(samples_per_clip)
    window = window_overlap_ratio * window + (1 - window_overlap_ratio)
    
    for source_idx in range(Config.MAX_SOURCES):
        # Calculate total length of output
        total_length = (num_clips - 1) * step_size + samples_per_clip
        output = np.zeros(total_length)
        normalization = np.zeros(total_length)
        
        for i in range(num_clips):
            start = i * step_size
            end = start + samples_per_clip
            
            # Apply window for smooth blending
            print(np.array(aligned_predictions[i][source_idx]).shape)
            
            output[start:end] += aligned_predictions[i][source_idx] * window
            normalization[start:end] += window
        
        # Normalize to avoid amplitude changes due to overlapping
        mask = normalization > 1e-10
        output[mask] /= normalization[mask]
        
        reconstructed_sources.append(output)
    
    return reconstructed_sources
    
def generate_prediction(model_dir, model_filename, audio_dir, audio_filename, clip_duration_seconds=1.0, window_overlap_ratio=0.25):
    audio_file = os.path.join(audio_dir, audio_filename)

    # audio, _ = process_audio_for_prediction(audio_file)
    output_dir = os.path.join(audio_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    model = load_saved_model(model_dir, model_filename)
    # separated_sources = separate_audio(model, audio_file, clip_duration_seconds=1.0, window_overlap_ratio=0.25)
    
    
    separated_sources = separate_audio_with_consistent_tracking(
        model, audio_file, clip_duration_seconds, window_overlap_ratio, sample_rate=16000
    )
    
    for i, source in enumerate(separated_sources):
        sf.write(os.path.join(output_dir, f"source_{i+1}.wav"), source, 16000)

def separate_mp4(video_dir, video_filename, audio_filename, start_time, length):
    video_file = os.path.join(video_dir, video_filename)
    print(f"Separating audio from {video_file}...")
    audio = AudioSegment.from_file(video_file, "mp4")#[start_time * 1000:(start_time + length) * 1000]
    audio = audio.set_frame_rate(16000)
    audio.export(os.path.join(video_dir, audio_filename), format="wav")
    return os.path.join(video_dir, audio_filename)



if __name__ == "__main__":
    # for testing with arbitrary model on test mix clip (already processed 1s clip)
    # model = load_saved_model("models", "wavelet_unet_22_0.0002")
    # test_separation(model, "data/test_mix.wav", "data/output")
    
    # audio_path = separate_mp4(video_dir="data/joe", video_filename="joe.mp4", audio_filename="joe.wav", start_time=7, length=10)
    generate_prediction(model_dir="models/arbitrary", model_filename="wavelet_unet_22_0.0002", audio_dir="data/joe", audio_filename="joe.wav")

    # ex1 = wavfile.read("data/ex2/true1.wav")[1]
    # ex2 = wavfile.read("data/ex2/true2.wav")[1]
    # sample = generate_sample_from_clips(ex1, ex2)
    # sf.write("data/ex2/mixed.wav", sample.numpy()[0] , 16000)
    # generate_prediction(model_dir="models/arbitrary", model_filename="wavelet_unet_22_0.0002", audio_dir="data/ex2", audio_filename="mixed.wav")