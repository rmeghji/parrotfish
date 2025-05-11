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

config = Config()

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

def subtract_wav(source1, source2, mixed):
    mixed_, msr = librosa.load(mixed,mono=False)
    print(f"Mixed shape: {msr}")
    source1_, s1sr = librosa.load(source1,mono=False)
    print(f"Source 1 shape: {s1sr}")
    source2_, s2sr = librosa.load(source2,mono=False)
    print(f"Source 2 shape: {s2sr}")

    subtracted_1 = mixed_ - source1_ 
    subtracted_2 = mixed_ - source2_ 
    subtracted_pair = source1_ - source2_
    
    return subtracted_1, subtracted_2, subtracted_pair
    

if __name__ == "__main__":
    # subtracted_1, subtracted_2, subtracted_pair = subtract_wav("data/output/source_1.wav", "data/output/source_2.wav", "data/test_mix.wav")
    # sf.write("data/output/subtracted_1.wav", subtracted_1, 22050)
    # sf.write("data/output/subtracted_2.wav", subtracted_2, 22050)
    # sf.write("data/output/subtracted_pair.wav", subtracted_pair, 22050)
    # Run the main pipeline
    # clips_dir = "data/clips"
    # model, history = main(clips_dir)


    model = load_saved_model("models/arbitrary", "wavelet_unet_37_0.0003")
    test_separation(model, "data/test_mix.wav", "data/output")