"""Trading configuration parameters"""

# Trading Parameters
INITIAL_BALANCE = 10000  # Initial balance in USD
MAX_POSITIONS = 5        # Maximum number of positions
MAX_LOTS = 1.0          # Maximum lot size per trade
RISK_PER_TRADE = 0.02   # 2% risk per trade

# Evolution Strategy Parameters
POPULATION_SIZE = 15
SIGMA = 0.1
LEARNING_RATE = 0.03
ITERATIONS = 500
CHECKPOINT_INTERVAL = 10

# Model Parameters
WINDOW_SIZE = 30
LAYER_SIZE = 500
OUTPUT_SIZE = 3

# Data Parameters
TIMEFRAME = 'H1'        # 1 hour timeframe
LOOKBACK_PERIOD = 1000  # Number of historical candles to fetch