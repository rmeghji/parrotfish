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

def load_and_preprocess_audio_files(data_dir):
    """Load all audio files and preprocess them for training."""
    print("Loading and preprocessing audio files...")
    
    # Find all audio files
    audio_files = glob(os.path.join(data_dir, "**/*.wav"), recursive=True)
    if not audio_files:
        raise ValueError(f"No audio files found in {data_dir}")
    
    print(f"Found {len(audio_files)} audio files")
    
    # Create a directory for preprocessed segments
    # os.makedirs("/content/preprocessed", exist_ok=True)
    
    # Process each file into segments
    segments = []
    for file_path in tqdm(audio_files):
        try:
            # Load audio
            audio, _ = tf.audio.decode_wav(
                tf.io.read_file(file_path), 
                desired_channels=config.CHANNELS
            )
            
            # Convert to numpy for easier handling
            audio = audio.numpy().reshape(-1)
            
            # Split into segments
            num_segments = len(audio) // config.SEGMENT_LENGTH
            for i in range(num_segments):
                start_idx = i * config.SEGMENT_LENGTH
                end_idx = start_idx + config.SEGMENT_LENGTH
                segment = audio[start_idx:end_idx]
                
                # Normalize segment to [-1, 1]
                if np.max(np.abs(segment)) > 0:
                    segment = segment / np.max(np.abs(segment))
                
                # Save segment
                segment_path = f"content/preprocessed/segment_{len(segments)}.npy"
                np.save(segment_path, segment)
                segments.append(segment_path)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    print(f"Created {len(segments)} segments")
    return segments

def create_source_combinations(segments, num_examples=31000):
    """Create a balanced set of source combinations for mixtures."""
    print("Creating source combinations...")
    
    num_segments = len(segments)
    combinations_per_count = num_examples // 3  # Even split between 2, 3, and 4 sources
    
    all_combinations = []
    
    # Generate combinations for each source count
    for source_count in range(config.MIN_SOURCES, config.MAX_SOURCES + 1):
        print(f"Generating {combinations_per_count} combinations with {source_count} sources")
        
        if source_count > num_segments:
            raise ValueError(f"Not enough segments ({num_segments}) to create combinations of {source_count} sources")
        
        # Calculate how many unique combinations are possible
        max_possible = min(combinations_per_count, 
                           int(np.math.comb(num_segments, source_count)))
        
        # Randomly sample combinations without replacement if possible
        if max_possible < combinations_per_count:
            print(f"Warning: Only {max_possible} unique combinations possible for {source_count} sources")
            
            # Generate all possible combinations indices
            indices = list(range(num_segments))
            combinations_list = []
            
            # Create combinations until we reach the target or exhaust possibilities
            count = 0
            while count < max_possible:
                # Randomly sample source_count indices
                combo = sorted(random.sample(indices, source_count))
                combo_key = tuple(combo)
                
                # Only add if this exact combination hasn't been seen before
                if combo_key not in combinations_list:
                    combinations_list.append(combo_key)
                    all_combinations.append([segments[i] for i in combo])
                    count += 1
        else:
            # When we have enough segments, directly sample random combinations
            for _ in range(combinations_per_count):
                source_indices = random.sample(range(num_segments), source_count)
                all_combinations.append([segments[i] for i in source_indices])
    
    print(f"Created {len(all_combinations)} source combinations")
    return all_combinations

def zero_pad_sources(sources, max_sources):
    """Zero pad sources to have the same number of sources."""
    num_sources = len(sources)
    if num_sources < max_sources:
        # Create zero arrays for padding
        zero_pad = np.zeros((max_sources - num_sources, config.SEGMENT_LENGTH), dtype=np.float32)
        return np.vstack([sources, zero_pad])
    return sources

class AudioMixtureDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, source_combinations, batch_size=16, max_sources=4, shuffle=True,
                 workers=4, use_multiprocessing=True):
        self.source_combinations = source_combinations
        self.batch_size = batch_size
        self.max_sources = max_sources
        self.shuffle = shuffle
        self.workers = workers
        self.use_multiprocessing = use_multiprocessing
        self.indices = np.arange(len(self.source_combinations))
        if self.shuffle:
            np.random.shuffle(self.indices)

    
    def __len__(self):
        return len(self.source_combinations) // self.batch_size
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_combinations = [self.source_combinations[i] for i in batch_indices]
        
        # Initialize batch arrays
        X_batch = np.zeros((self.batch_size, config.SEGMENT_LENGTH, 1), dtype=np.float32)
        y_batch = np.zeros((self.batch_size, self.max_sources, config.SEGMENT_LENGTH, 1), dtype=np.float32)
        
        for i, source_paths in enumerate(batch_combinations):
            # Load sources
            sources = []
            for path in source_paths:
                source = np.load(path).astype(np.float32)
                # Reshape for channel dimension
                source = source.reshape(-1, 1)
                sources.append(source)
            
            # Create mixture by summing sources
            mixture = np.zeros((config.SEGMENT_LENGTH, 1), dtype=np.float32)
            for source in sources:
                mixture += source
            
            # Normalize mixture
            if np.max(np.abs(mixture)) > 0:
                mixture = mixture / np.max(np.abs(mixture))
            
            # Store mixture in X_batch
            X_batch[i] = mixture
            
            # Zero-pad sources if needed and store in y_batch
            sources_array = np.array(sources)
            if len(sources) < self.max_sources:
                # Create zero-padding for missing sources
                zero_pad = np.zeros((self.max_sources - len(sources), config.SEGMENT_LENGTH, 1), dtype=np.float32)
                sources_array = np.vstack([sources_array, zero_pad])
            
            y_batch[i] = sources_array
        
        return X_batch, y_batch
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

def train_model_local():
    # Load and preprocess data
    segments = load_and_preprocess_audio_files(config.DATA_DIR)
    source_combinations = create_source_combinations(segments, config.NUM_EXAMPLES)
    
    # Split into training and validation sets
    val_size = int(len(source_combinations) * config.VAL_SPLIT)
    train_combinations = source_combinations[:-val_size]
    val_combinations = source_combinations[-val_size:]
    
    print(f"Training on {len(train_combinations)} combinations")
    print(f"Validating on {len(val_combinations)} combinations")
    
    train_generator = AudioMixtureDataGenerator(
        train_combinations, 
        batch_size=config.BATCH_SIZE,
        max_sources=config.MAX_SOURCES,
        shuffle=True,
        workers=4,
        use_multiprocessing=True
    )
    
    val_generator = AudioMixtureDataGenerator(
        val_combinations, 
        batch_size=config.BATCH_SIZE,
        max_sources=config.MAX_SOURCES,
        shuffle=False,
        workers=4,
        use_multiprocessing=True
    )
    
    model = WaveletUNet(
        num_coeffs=config.NUM_COEFFS,
        wavelet_depth=config.WAVELET_DEPTH,
        batch_size=config.BATCH_SIZE,
        channels=config.CHANNELS,
        num_layers=config.NUM_LAYERS,
        num_init_filters=config.NUM_INIT_FILTERS,
        filter_size=config.FILTER_SIZE,
        merge_filter_size=config.MERGE_FILTER_SIZE,
        l1_reg=config.L1_REG,
        l2_reg=config.L2_REG,
        max_sources=config.MAX_SOURCES,
        wavelet_family=config.WAVELET_FAMILY,
        
    )
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
    dummy_input = tf.zeros((config.BATCH_SIZE, config.SEGMENT_LENGTH, 1))
    _ = model(dummy_input)
    model.compile(
        optimizer=optimizer,
        loss=pit_loss,
        metrics=['mse']
    )
    
    model.summary()
    
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(config.CHECKPOINT_DIR, 'improved_wavelet_unet_{epoch:02d}_{val_loss:.4f}.keras'),
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            save_weights_only=False,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(config.CHECKPOINT_DIR, 'logs'),
            histogram_freq=1,
            update_freq='epoch'
        )
    ]
    
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=config.EPOCHS,
        callbacks=callbacks,
    )
    
    model_save_path = os.path.join(config.CHECKPOINT_DIR, 'model.keras')
    
    model_json = model.to_json()
    with open(os.path.join(config.CHECKPOINT_DIR, 'model_architecture.json'), 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(os.path.join(config.CHECKPOINT_DIR, 'model.weights.h5'))
    
    try:
        saved_model_path = os.path.join(config.CHECKPOINT_DIR, 'model.savedmodel')
        tf.saved_model.save(model, saved_model_path)
        print(f"Model successfully saved to {saved_model_path}")
    except Exception as e:
        print(f"Error saving in SavedModel format: {e}")
        print("Falling back to H5 format only")
    
    # save weights in h5
    try:
        model.save(model_save_path, save_format='h5')
        print(f"Model successfully saved to {model_save_path}")
    except Exception as e:
        print(f"Error saving in H5 format: {e}")
        print("Model is saved as architecture + weights only")
    
    return model, history
