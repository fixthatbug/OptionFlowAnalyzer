# alpha_config.py
"""
Configuration for alpha extraction parameters
"""

# Signal Detection Thresholds
ALPHA_BLOCK_SIZE_THRESHOLD = 50  # Minimum contracts for block trade
ALPHA_BLOCK_NOTIONAL_THRESHOLD = 25000  # Minimum notional for block
ALPHA_ULTRA_BLOCK_SIZE = 500  # Ultra-large block threshold
ALPHA_ULTRA_BLOCK_NOTIONAL = 100000  # Ultra-large notional threshold

# Sweep Detection
ALPHA_SWEEP_TIME_WINDOW_MS = 5000  # Time window for sweep detection
ALPHA_SWEEP_MIN_LEGS = 3  # Minimum legs for sweep
ALPHA_SWEEP_MIN_EXCHANGES = 2  # Minimum exchanges for sweep
ALPHA_SWEEP_MIN_TOTAL_QTY = 50  # Minimum total quantity for sweep

# Smart Money Detection
ALPHA_SMART_MONEY_MIN_PREMIUM = 10000  # Minimum premium for smart money
ALPHA_SMART_MONEY_VOLUME_RATIO = 2.0  # Buy/sell ratio for direction
ALPHA_SMART_MONEY_CONCENTRATION_THRESHOLD = 0.3  # 30% of volume in one strike

# Unusual Activity
ALPHA_UNUSUAL_VOLUME_MULTIPLIER = 3.0  # Multiple of average for unusual
ALPHA_UNUSUAL_IV_DEVIATION = 2.0  # Standard deviations for IV spike

# Microstructure Thresholds
ALPHA_HIGH_VPIN_THRESHOLD = 0.7  # High toxicity threshold
ALPHA_WIDE_SPREAD_MULTIPLIER = 1.5  # Wide effective spread
ALPHA_MM_PRESSURE_THRESHOLD = 0.7  # Market maker inventory pressure
ALPHA_HIDDEN_LIQUIDITY_THRESHOLD = 0.3  # Hidden order detection

# Signal Quality Thresholds
ALPHA_MIN_CONFIDENCE = 60  # Minimum confidence to show signal
ALPHA_MIN_URGENCY = 50  # Minimum urgency score
ALPHA_MIN_SMART_MONEY_SCORE = 60  # Minimum smart money score
ALPHA_SIGNAL_STRENGTH_THRESHOLD = 2  # Minimum strength (MODERATE)

# Real-time Monitoring
ALPHA_BUFFER_PROCESS_INTERVAL = 5  # Seconds between buffer processing
ALPHA_URGENT_SIGNAL_THRESHOLD = 85  # Urgency score for immediate alert
ALPHA_SMART_MONEY_ALERT_THRESHOLD = 80  # Smart money score for alert
ALPHA_PERFORMANCE_TRACK_DAYS = 5  # Days to track signal performance
ALPHA_MAX_SIGNALS_PER_TICKER = 50  # Maximum signals to keep in memory

# Risk Management
ALPHA_DEFAULT_STOP_LOSS = 0.3  # 30% stop loss for options
ALPHA_DEFAULT_TARGET = 0.5  # 50% profit target
ALPHA_POSITION_SIZE_FACTOR = 0.02  # 2% of account per trade
ALPHA_MAX_RISK_PER_TRADE = 0.05  # 5% max risk

# Signal Ranking Weights
ALPHA_RANKING_WEIGHTS = {
    'signal_strength': 0.25,
    'confidence': 0.25,
    'urgency': 0.20,
    'smart_money': 0.20,
    'notional_size': 0.10
}

# Alert Configuration
ALPHA_ALERT_METHODS = ['ui', 'sound', 'file']  # Available alert methods
ALPHA_ALERT_SOUND_FILE = 'alert.wav'  # Sound file for alerts
ALPHA_ALERT_COOLDOWN = 60  # Seconds between same signal alerts

# Performance Metrics
ALPHA_WIN_RATE_TARGET = 0.55  # Target win rate
ALPHA_PROFIT_FACTOR_TARGET = 1.5  # Target profit factor
ALPHA_SHARPE_RATIO_TARGET = 1.0  # Target Sharpe ratio