"""Training and model configuration parameters"""

# Data parameters
PATCH_SIZE = 9  # Size of NDVI patches (must be odd)
SEQUENCE_LENGTH = 10  # Number of timesteps to use for prediction
DATA_SAMPLE_PERCENTAGE = 0.5  # Percentage of data to use for training and validation

# Training parameters
BATCH_SIZE = 16384  # Increased batch size for better GPU utilization
EPOCHS = 30  # Number of epochs to train
WARMUP_EPOCHS = 5  # Number of epochs to keep learning rate constant
LEARNING_RATE = 2e-3  # Increased learning rate to compensate for larger batch size
DECAY_GAMMA = 0.95  # Decay rate for learning rate scheduler
GRADIENT_ACCUMULATION_STEPS = 1

# DataLoader parameters
WORKERS = 24  # Reduced number of workers to prevent CPU bottleneck
PREFETCH_FACTOR = 4  # Reduced prefetch factor to prevent memory issues
LOG_INTERVAL = 10  # Reduced logging interval for better progress tracking

# Model parameters
MODEL_D_MODEL = 256  # Model dimension
MODEL_NUM_HEADS = 8  # Number of attention heads
MODEL_NUM_LAYERS = 3  # Number of transformer layers
MODEL_D_FF = 1024  # Feed-forward hidden dimension
MODEL_DROPOUT = 0.1  # Dropout probability
MODEL_NDVI_EMBED_DIM = 32  # NDVI embedding dimension
MODEL_YEAR_EMBED_DIM = 8  # Year embedding dimension
MODEL_LATLON_EMBED_DIM = 8  # Lat/lon embedding dimension
MODEL_MAX_SEQ_LEN = 5000  # Maximum sequence length for positional encoding
MODEL_FIRST_YEAR = 1984  # Reference year for year encoding

# Training/validation split years
TRAIN_YEARS = (1984, 2014)
VAL_YEARS = (2015, 2024)
