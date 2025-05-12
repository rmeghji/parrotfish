import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf

# Fix the file paths and loading
try:
    # Woman voice - true and predicted
    woman_true = librosa.load('ex2/woman_voice_true.wav', sr=16000, mono=True)[0]
    woman_pred = librosa.load('ex2/output/woman_voice.wav', sr=16000, mono=True)[0]
    
    # Man voice - true and predicted (fixing the typo in the path)
    man_true = librosa.load('ex2/man_voice_true.wav', sr=16000, mono=True)[0]
    man_pred = librosa.load('ex2/output/man_voice.wav', sr=16000, mono=True)[0]
    
    # Silence - true and predicted
    silence_true = librosa.load('ex2/mixed.wav', sr=16000, mono=True)[0]
    silence_pred = librosa.load('ex2/output/silence.wav', sr=16000, mono=True)[0]
    
    
    
    # Compute STFTs
    def compute_stft(audio, n_fft=2048, hop_length=256):
        return librosa.amplitude_to_db(np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)), ref=np.max)
    
    woman_true_stft = compute_stft(woman_true)
    woman_pred_stft = compute_stft(woman_pred)
    man_true_stft = compute_stft(man_true)
    man_pred_stft = compute_stft(man_pred)
    silence_true_stft = compute_stft(silence_true)
    silence_pred_stft = compute_stft(silence_pred)
    
    # Create figure with two columns and three rows
    plt.figure(figsize=(15, 12))
    
    # Parameters for spectrogram display
    sr = 16000
    hop_length = 512
    
    # Row 1: Woman voice
    plt.subplot(3, 2, 1)
    librosa.display.specshow(woman_true_stft, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log' )
    plt.title('Feminine Voice (True)')
    plt.colorbar(format='%+2.0f dB')
    
    plt.subplot(3, 2, 2)
    librosa.display.specshow(woman_pred_stft, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log' )
    plt.title('Feminine Voice (Predicted)')
    plt.colorbar(format='%+2.0f dB')
    
    # Row 2: Man voice
    plt.subplot(3, 2, 3)
    librosa.display.specshow(man_true_stft, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log' )
    plt.title('Masculine Voice (True)')
    plt.colorbar(format='%+2.0f dB')
    
    plt.subplot(3, 2, 4)
    librosa.display.specshow(man_pred_stft, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log' )
    plt.title('Masculine Voice (Predicted)')
    plt.colorbar(format='%+2.0f dB')
    
    # Row 3: Silence
    plt.subplot(3, 2, 5)
    librosa.display.specshow(silence_true_stft, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log' )
    plt.title('Mixture Audio')
    plt.colorbar(format='%+2.0f dB')
    
    plt.subplot(3, 2, 6)
    librosa.display.specshow(silence_pred_stft, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log' )
    plt.title('Silence (Predicted)')
    plt.colorbar(format='%+2.0f dB')
    
    plt.tight_layout()
    plt.savefig('voice_separation_spectrograms.png', dpi=300)
    plt.show()
    
except FileNotFoundError as e:
    # Handle file not found errors
    print(f"Error: {e}")
    print("Please check the file paths and make sure the audio files exist in the specified locations.")
    
    # Create a simplified demonstration with random data
    print("Creating a demonstration with synthetic data instead...")
    
    # Create synthetic audio data
    duration = 1  # seconds
    sr = 16000
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # Synthetic woman voice (higher frequency)
    woman_true = 0.5 * np.sin(2 * np.pi * 280 * t) + 0.3 * np.sin(2 * np.pi * 560 * t)
    woman_pred = woman_true + 0.1 * np.random.randn(len(woman_true))
    
    # Synthetic man voice (lower frequency)
    man_true = 0.5 * np.sin(2 * np.pi * 120 * t) + 0.3 * np.sin(2 * np.pi * 240 * t)
    man_pred = man_true + 0.1 * np.random.randn(len(man_true))
    
    # Silence
    silence_true = np.zeros_like(t)
    silence_pred = 0.05 * np.random.randn(len(silence_true))
    
    # Compute STFTs
    def compute_stft(audio, n_fft=2048, hop_length=128):
        return librosa.amplitude_to_db(np.abs(librosa.stft(audio / np.max(abs(audio)), n_fft=n_fft, hop_length=hop_length)), ref=np.max)
    
    woman_true_stft = compute_stft(woman_true)
    woman_pred_stft = compute_stft(woman_pred)
    man_true_stft = compute_stft(man_true)
    man_pred_stft = compute_stft(man_pred)
    silence_true_stft = compute_stft(silence_true)
    silence_pred_stft = compute_stft(silence_pred)
    
    # Create figure with two columns and three rows
    plt.figure(figsize=(15, 12))
    
    # Parameters for spectrogram display
    hop_length = 512
    
    # Row 1: Woman voice
    plt.subplot(3, 2, 1)
    librosa.display.specshow(woman_true_stft, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log' )
    plt.title('Woman Voice (True) - SYNTHETIC DATA')
    plt.colorbar(format='%+2.0f dB')
    
    plt.subplot(3, 2, 2)
    librosa.display.specshow(woman_pred_stft, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log' )
    plt.title('Woman Voice (Predicted) - SYNTHETIC DATA')
    plt.colorbar(format='%+2.0f dB')
    
    # Row 2: Man voice
    plt.subplot(3, 2, 3)
    librosa.display.specshow(man_true_stft, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log' )
    plt.title('Man Voice (True) - SYNTHETIC DATA')
    plt.colorbar(format='%+2.0f dB')
    
    plt.subplot(3, 2, 4)
    librosa.display.specshow(man_pred_stft, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log' )
    plt.title('Man Voice (Predicted) - SYNTHETIC DATA')
    plt.colorbar(format='%+2.0f dB')
    
    # Row 3: Silence
    plt.subplot(3, 2, 5)
    librosa.display.specshow(silence_true_stft, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log' )
    plt.title('Silence (True) - SYNTHETIC DATA')
    plt.colorbar(format='%+2.0f dB')
    
    plt.subplot(3, 2, 6)
    librosa.display.specshow(silence_pred_stft, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log' )
    plt.title('Silence (Predicted) - SYNTHETIC DATA')
    plt.colorbar(format='%+2.0f dB')
    
    plt.tight_layout()
    plt.savefig('voice_separation_spectrograms_synthetic.png', dpi=300)
    plt.show()