"""Utility functions for handling market state data"""

import numpy as np
from typing import List, Union

def get_state(data: Union[List[float], np.ndarray], t: int, n: int) -> np.ndarray:
    """
    Create a state representation from price data
    
    Args:
        data: List or array of close prices
        t: Current time index 
        n: Number of previous prices to include
        
    Returns:
        numpy array containing price changes
    """
    # Convert data to numpy array if it's a list
    if isinstance(data, list):
        data = np.array(data)
        
    # Calculate the starting index
    d = t - n + 1
    
    # Create the price block
    if d >= 0:
        # If we have enough historical data, use it directly
        block = data[d:t + 1]
    else:
        # If we need padding, create it properly
        padding = np.repeat(data[0], -d)
        history = data[0:t + 1]
        block = np.concatenate([padding, history])
    
    # Calculate price changes
    res = []
    for i in range(len(block) - 1):
        res.append(block[i + 1] - block[i])
        
    # Ensure we have the correct number of changes
    res = np.array(res)
    if len(res) < n - 1:
        # Pad with zeros if needed
        padding_needed = n - 1 - len(res)
        res = np.pad(res, (0, padding_needed), 'constant')
        
    return res.reshape(1, -1)

def normalize_data(data: np.ndarray) -> np.ndarray:
    """
    Normalize price data to range [0,1]
    
    Args:
        data: Price data to normalize
        
    Returns:
        Normalized data array
    """
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val == min_val:
        return np.zeros_like(data)
    return (data - min_val) / (max_val - min_val)

def calculate_position_size(
    balance: float,
    risk_per_trade: float,
    stop_loss_pips: float,
    max_position: float = 1.0
) -> float:
    """
    Calculate position size based on risk management rules
    
    Args:
        balance: Current account balance
        risk_per_trade: Risk percentage per trade (0-1)
        stop_loss_pips: Stop loss distance in pips
        max_position: Maximum allowed position size
        
    Returns:
        Position size in lots
    """
    if stop_loss_pips <= 0:
        return 0
        
    risk_amount = balance * risk_per_trade
    position_size = risk_amount / (stop_loss_pips * 10)  # For XAUUSD, 1 pip = $10
    return min(position_size, max_position)

def calculate_price_changes(prices: np.ndarray, period: int = 1) -> np.ndarray:
    """
    Calculate price changes over a given period
    
    Args:
        prices: Array of prices
        period: Number of periods to calculate change over
        
    Returns:
        Array of price changes
    """
    if len(prices) < period + 1:
        return np.zeros(len(prices))
    
    changes = np.zeros(len(prices))
    changes[period:] = prices[period:] - prices[:-period]
    return changes