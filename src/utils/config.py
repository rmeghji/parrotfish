class Config:
    """Configuration class for parrotfish""" 
    # Data settings
    DATA_DIR = "data/audio"
    CLIPS_DIR = "data/clips"
    SAMPLE_RATE = 16000
    SEGMENT_LENGTH = 16000  # 1 second at 16kHz
    
    # Model settings
    NUM_COEFFS = 16000  # 1 second at 16kHz
    WAVELET_DEPTH = 5
    BATCH_SIZE = 32 # 16-32
    CHANNELS = 1  # Mono audio
    NUM_LAYERS = 11 # 10-12
    NUM_INIT_FILTERS = 32 ## was 24
    FILTER_SIZE = 32 # was 16 should be 16
    MERGE_FILTER_SIZE = 32 # was 5 should be like 8
    L1_REG = 1e-6
    L2_REG = 1e-6
    
    # Training settings
    LEARNING_RATE = 1e-4
    EPOCHS = 50
    VAL_SPLIT = 0.1
    CHECKPOINT_DIR = "checkpoints"
    MAX_WORKERS = 4
    
    # Mixture generation
    MIN_SOURCES = 2
    MAX_SOURCES = 2 # first curriculum learning
    NUM_EXAMPLES = BATCH_SIZE * 5000
    
    # Wavelet settings
    WAVELET_FAMILY = 'db4'  # Daubechies wavelet with 4 vanishing moments