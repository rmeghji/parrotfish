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

def get_callbacks(save_directory):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(config.CHECKPOINT_DIR, 'wavelet_unet_{epoch:02d}_{val_loss:.4f}.keras'),
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            save_weights_only=False,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(save_directory, 'wavelet_unet_{epoch:02d}_{val_loss:.4f}.keras'),
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            save_weights_only=False,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(config.CHECKPOINT_DIR, 'wavelet_unet_{epoch:02d}_{val_loss:.4f}_weightsonly.weights.h5'),
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            save_weights_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(save_directory, 'wavelet_unet_{epoch:02d}_{val_loss:.4f}_weightsonly.weights.h5'),
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            save_weights_only=True,
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

def train_model(clips_dir=None, tfrecords_dir=None, save_directory=None, num_speakers=config.MAX_SOURCES):
    """Main function to run the audio source separation pipeline"""    
    print("Starting audio source separation pipeline...")

    if clips_dir and not tfrecords_dir:
        print("Creating TensorFlow dataset for training from WAV files...")
        dataset = create_tf_dataset(
            base_dir=config.DATA_DIR,
            clips_dir=clips_dir,
            num_speakers=num_speakers,
            batch_size=config.BATCH_SIZE
        )
    elif tfrecords_dir and not clips_dir:
        print("Creating TensorFlow dataset for training from TFRecords...")
        dataset = create_tf_dataset_from_tfrecords(
            tfrecords_dir=tfrecords_dir,
            num_speakers=num_speakers,
            batch_size=config.BATCH_SIZE
        )

    train_size = int(config.NUM_EXAMPLES * (1 - config.VAL_SPLIT))
    val_size = int(config.NUM_EXAMPLES * config.VAL_SPLIT)
    
    train_steps = int(train_size / config.BATCH_SIZE)
    val_steps = int(val_size / config.BATCH_SIZE)
    
    # train_dataset = dataset.take(train_size).repeat()
    # val_dataset = dataset.skip(train_size).take(val_size).repeat()

    train_dataset = dataset.repeat()
    val_dataset = dataset.repeat()
    
    print(f"Training dataset created with {train_size} examples")
    print(f"Validation dataset created with {int(config.NUM_EXAMPLES * config.VAL_SPLIT)} examples")
    
    print("Creating Wavelet U-Net model...")
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
        max_sources=num_speakers,
        wavelet_family=config.WAVELET_FAMILY
    )
    
    print("Compiling model with PIT loss...")
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
    dummy_input = tf.zeros((config.BATCH_SIZE, config.SEGMENT_LENGTH, 1))
    _ = model(dummy_input)
    model.compile(
        optimizer=optimizer,
        loss=pit_loss,
        metrics=['mse']
    )
    
    model.summary()
    
    print("Setting up training callbacks...")
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    callbacks = get_callbacks(save_directory)

    print(f"Training model for {config.EPOCHS} epochs...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.EPOCHS,
        callbacks=callbacks,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
    )
    
    save_model(model, config, save_directory)
    plot_model(history, config, save_directory)
    
    print("Wavelet U-Net pipeline completed successfully!")
    return model, history

if __name__ == "__main__":
    train_model(tfrecords_dir="data", save_directory="checkpoints")
