# rtd_usage_example.py
"""
Complete example of how to use the RTD system with signal detection
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

# Import the RTD system components
from rtd_handler import RTDDataHandler, RTDSignalProcessor
from rtd_integration import RTDAnalysisIntegrator
from alpha_extractor import AlphaExtractor
from analysis_engine import AnalysisEngine

def example_rtd_workflow():
    """Complete example workflow using RTD with signal detection"""
    
    print("=== RTD SYSTEM USAGE EXAMPLE ===\n")
    
    # Step 1: Simulate some cleaned options data
    print("1. Creating sample options data...")
    sample_data = create_sample_options_data()
    print(f"   Created {len(sample_data)} sample trades\n")
    
    # Step 2: Run analysis to identify signals
    print("2. Running analysis to identify alpha signals...")
    analysis_engine = AnalysisEngine("SPY")
    analysis_results = analysis_engine.run_holistic_analysis(sample_data, "SPY")
    
    alpha_signals = analysis_results.get('alpha_signals', [])
    print(f"   Found {len(alpha_signals)} alpha signals")
    
    if alpha_signals:
        print("   Top signals:")
        for i, signal in enumerate(alpha_signals[:3], 1):
            print(f"     {i}. {signal.signal_type.value} - {signal.option_symbol} ({signal.confidence:.0f}% confidence)")
    print()
    
    # Step 3: Set up RTD integration
    print("3. Setting up RTD integration...")
    
    def log_callback(message, ticker=None, is_error=False):
        prefix = "ERROR" if is_error else "INFO"
        print(f"   [{prefix}] {message}")
    
    # Initialize RTD integrator
    integrator = RTDAnalysisIntegrator("SPY", log_callback)
    
    # Initialize integration (in real usage, this would connect to Excel/TOS)
    print("   Initializing RTD connection...")
    if integrator.initialize_integration(analysis_results, sample_data):
        print("   ✓ RTD integration initialized successfully")
    else:
        print("   ✗ RTD integration failed (this is expected in demo mode)")
        print("   Note: In real usage, this would connect to ThinkorSwim via Excel RTD")
        return
    
    # Step 4: Demonstrate signal enhancement
    print("\n4. Demonstrating RTD signal enhancement...")
    demonstrate_signal_enhancement()
    
    # Step 5: Show monitoring workflow
    print("\n5. Demonstrating real-time monitoring workflow...")
    demonstrate_monitoring_workflow()
    
    print("\n=== RTD EXAMPLE COMPLETE ===")


def create_sample_options_data():
    """Create sample options data for demonstration"""
    
    # Create sample option symbols
    base_symbols = [
        "SPY240315C00500000",  # SPY Call
        "SPY240315P00480000",  # SPY Put
        "SPY240315C00505000",  # SPY Call OTM
        "SPY240315P00475000",  # SPY Put OTM
        "SPY240322C00500000",  # SPY Call next week
    ]
    
    # Generate sample trades
    trades = []
    base_time = datetime.now().replace(second=0, microsecond=0)
    
    for i in range(50):
        symbol = np.random.choice(base_symbols)
        
        # Simulate different trade characteristics
        if "C00500" in symbol:  # ATM calls - more activity
            quantity = np.random.choice([25, 50, 75, 100, 150, 200], p=[0.3, 0.3, 0.2, 0.1, 0.05, 0.05])
        elif "P00480" in symbol:  # OTM puts - block trades
            quantity = np.random.choice([50, 100, 200, 500], p=[0.4, 0.3, 0.2, 0.1])
        else:
            quantity = np.random.choice([10, 25, 50, 100], p=[0.4, 0.3, 0.2, 0.1])
        
        # Price based on moneyness
        if "C00500" in symbol:
            base_price = 2.50
        elif "P00480" in symbol:
            base_price = 1.80
        elif "C00505" in symbol:
            base_price = 1.20
        else:
            base_price = 0.80
        
        price = base_price + np.random.normal(0, 0.20)
        price = max(0.05, price)  # Minimum price
        
        # Determine aggressor
        aggressor = np.random.choice(["Buy", "Sell"], p=[0.6, 0.4])
        
        # Generate trade time (spread over last hour)
        trade_time = base_time - timedelta(minutes=np.random.randint(0, 60))