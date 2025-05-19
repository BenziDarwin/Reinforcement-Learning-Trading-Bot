"""Main script to run the XAUUSD trading bot"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import os
import pickle

from models.neural_model import Model
from agents.evolution_agent import TradingAgent
from config.trading_config import *
import fickling

def fetch_xauusd_data(timeframe: str = TIMEFRAME, periods: int = LOOKBACK_PERIOD):
    """Fetch XAUUSD historical data from MetaTrader5"""
    if not mt5.initialize():
        print("MetaTrader5 initialization failed")
        return None
    
    rates = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_H1, 0, periods)
    mt5.shutdown()
    
    if rates is None:
        return None
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def prepare_data(df: pd.DataFrame) -> np.ndarray:
    """Prepare data for training"""
    # Use close prices for training
    prices = df['close'].values
    return prices

def save_model(agent: TradingAgent, filename: str = 'xauusd_model.pkl'):
    """Save the trained model"""
    model_dir = 'models/saved'
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, filename)
    with open(model_path, 'wb') as f:
        pickle.dump(agent.model.get_weights(), f)
    print(f"Model saved to {model_path}")

def load_model(agent: TradingAgent, filename: str = 'xauusd_model.pkl'):
    """Load a trained model"""
    model_path = os.path.join('models/saved', filename)
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            weights = fickling.load(f)
            agent.model.set_weights(weights)
        print(f"Model loaded from {model_path}")
        return True
    return False

def evaluate_model(agent: TradingAgent, test_prices: np.ndarray):
    """Evaluate model performance"""
    final_balance, trade_history = agent.trade(test_prices, plot=True)
    
    # Calculate performance metrics
    total_trades = len(trade_history)
    profitable_trades = sum(1 for trade in trade_history if 
                          trade.get('profit', 0) > 0)
    
    if total_trades > 0:
        win_rate = (profitable_trades / total_trades) * 100
    else:
        win_rate = 0
    
    roi = ((final_balance - INITIAL_BALANCE) / INITIAL_BALANCE) * 100
    
    print("\nPerformance Metrics:")
    print(f"Final Balance: ${final_balance:.2f}")
    print(f"Return on Investment: {roi:.2f}%")
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.2f}%")
    
    return final_balance, trade_history

def main():
    # Fetch historical data
    print("Fetching XAUUSD data...")
    df = fetch_xauusd_data()
    if df is None:
        print("Failed to fetch data. Exiting...")
        return
    
    # Prepare data
    prices = prepare_data(df)
    train_size = int(len(prices) * 0.8)
    train_prices = prices[:train_size]
    test_prices = prices[train_size:]
    
    # Initialize model and agent
    model = Model(WINDOW_SIZE, LAYER_SIZE, OUTPUT_SIZE)
    agent = TradingAgent(model, INITIAL_BALANCE, MAX_BUY, MAX_BUY)  # Provide `max_sell`

    # Check for existing model
    if not load_model(agent):
        print("\nTraining new model...")
        print(f"Training on {len(train_prices)} price points...")
        agent.train(train_prices, iterations=ITERATIONS, checkpoint=CHECKPOINT_INTERVAL)  # Added checkpoint
        save_model(agent)
    
    # Evaluate on test data
    print("\nEvaluating model on test data...")
    final_balance, trade_history = evaluate_model(agent, test_prices)
    
    # Live trading setup (commented out for safety)
    """
    print("\nSetting up live trading...")
    while True:
        current_data = fetch_xauusd_data(periods=WINDOW_SIZE+1)
        if current_data is not None:
            current_prices = prepare_data(current_data)
            state = get_state(current_prices, 0, WINDOW_SIZE + 1)
            action, position_size = agent.act(state)
            
            if action == 1:  # Buy signal
                print(f"Buy signal detected. Suggested position size: {position_size}")
            elif action == 2:  # Sell signal
                print(f"Sell signal detected")
            
            # Add your live trading logic here
            
        time.sleep(3600)  # Wait for 1 hour before next check
    """

if __name__ == "__main__":
    main()

