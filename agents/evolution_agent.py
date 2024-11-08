import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import MetaTrader5 as mt5

from models.deep_evolution import Deep_Evolution_Strategy
from utils.state_utils import get_state
from config.trading_config import POPULATION_SIZE, SIGMA, LEARNING_RATE, WINDOW_SIZE

class TradingAgent:
    def __init__(self, model, initial_balance: float, max_buy: int, max_sell: int, skip: int = 1, short_window: int = 5, long_window: int = 20):
        """
        Initialize the trading agent with MT5 for XAUUSD.
        
        Args:
            model: Neural network model for decision making
            initial_balance: Starting balance
            max_buy: Maximum number of units to buy
            max_sell: Maximum number of units to sell
        """
        self.model = model
        self.initial_balance = initial_balance
        self.max_buy = max_buy
        self.max_sell = max_sell
        self.skip = skip
        self.short_window = short_window  # Short-term moving average window size
        self.long_window = long_window    # Long-term moving average window size
        
        self.es = Deep_Evolution_Strategy(
            self.model.get_weights(),
            self.get_reward,
            POPULATION_SIZE,
            SIGMA,
            LEARNING_RATE,
        )

    def act(self, state: np.ndarray) -> Tuple[int, float]:
        """Choose action based on current state."""
        decision, buy = self.model.predict(np.array(state))
        return np.argmax(decision[0]), int(buy[0])

    def get_reward(self, weights: List[np.ndarray]) -> float:
        """Calculate reward based on given weights."""
        initial_money = self.initial_balance
        starting_money = initial_money
        self.model.weights = weights
        
        state = get_state(self.prices, 0, WINDOW_SIZE + 1)
        inventory = []
        quantity = 0

        for t in range(len(self.prices) - 1):
            action, buy_units = self.act(state)
            next_state = get_state(self.prices, t + 1, WINDOW_SIZE + 1)
            current_price = self.prices[t]
            
            # Buy action
            if action == 1 and initial_money >= current_price:
                buy_units = max(1, min(buy_units, self.max_buy))
                total_buy = buy_units * current_price
                initial_money -= total_buy
                inventory.append(total_buy)
                quantity += buy_units

            # Sell action
            elif action == 2 and len(inventory) > 0:
                sell_units = min(quantity, self.max_sell)
                total_sell = sell_units * current_price
                initial_money += total_sell
                inventory = []
                quantity = 0
            
            state = next_state

        return ((initial_money - starting_money) / starting_money) * 100

    def train(self, prices: List[float], iterations: int, checkpoint: int):
        """Train the trading agent."""
        self.prices = prices
        self.es.train(iterations, print_every=checkpoint)

    def trade(self, prices: List[float], plot: bool = True) -> Tuple[float, List[dict]]:
        """Simulate trading activity with improved strategy."""
        initial_money = self.initial_balance
        starting_money = initial_money
        state = get_state(prices, 0, WINDOW_SIZE + 1)
        states_buy = []
        states_sell = []
        inventory = []
        quantity = 0
        trade_history = []
        skip = 3  # Increase skip to reduce trade frequency
        min_profit_percent = 0.1  # Lower minimum profit threshold to allow more trades
        
        for t in range(0, len(prices) - 1, skip):
            action, buy_units = self.act(state)
            next_state = get_state(prices, t + 1, WINDOW_SIZE + 1)
            current_price = prices[t]
            
            # Calculate moving averages (Optional: Relax or remove trend conditions)
            if t >= self.long_window:
                short_avg = np.mean(prices[t - self.short_window + 1 : t + 1])
                long_avg = np.mean(prices[t - self.long_window + 1 : t + 1])
                downtrend = current_price < long_avg  # Optional condition for trend

            # Buy condition
            if action == 1 and initial_money >= current_price:
                buy_units = max(1, min(buy_units, self.max_buy))
                total_buy = buy_units * current_price
                initial_money -= total_buy
                inventory.append(total_buy)
                quantity += buy_units
                states_buy.append(t)
                trade_history.append({
                    'type': 'buy',
                    'price': current_price,
                    'units': buy_units,
                    'balance': initial_money
                })
                print(f"Day {t}: Buy {buy_units} units at price {current_price}, Total Balance: {initial_money}")

            # Sell condition
            elif action == 2 and len(inventory) > 0:
                bought_price = inventory.pop(0)
                sell_units = min(quantity, self.max_sell)
                if sell_units < 1:
                    continue
                
                # Check for profit threshold (Optional: Lower threshold or remove it)
                profit_percent = ((current_price - bought_price) / bought_price) * 100
                if profit_percent < min_profit_percent:
                    continue  # Skip trades with low profit (can remove this condition)

                quantity -= sell_units
                total_sell = sell_units * current_price
                initial_money += total_sell
                states_sell.append(t)
                trade_history.append({
                    'type': 'sell',
                    'price': current_price,
                    'units': sell_units,
                    'balance': initial_money,
                    'investment_return': profit_percent
                })
                print(
                    f"Day {t}: Sell {sell_units} units at price {current_price}, "
                    f"Investment Return: {profit_percent:.2f}%, Total Balance: {initial_money}"
                )

            state = next_state

        # Calculate and print total investment gain/loss
        total_investment_return = ((initial_money - starting_money) / starting_money) * 100
        print(f"\nTotal Gained: {initial_money - starting_money}, Total Investment Return: {total_investment_return:.2f}%")

        # Plot trades if specified
        if plot:
            self._plot_trades(prices, states_buy, states_sell)
        
        return initial_money, trade_history

    def _plot_trades(self, prices: List[float], states_buy: List[int], states_sell: List[int]):
        """Plot the buy and sell trades on price chart."""
        plt.figure(figsize=(20, 10))
        plt.plot(prices, label='XAUUSD Price', color='g')
        plt.plot(prices, 'X', label='Buy', markevery=states_buy, color='b')
        plt.plot(prices, 'o', label='Sell', markevery=states_sell, color='r')
        plt.title('Trading Activity on XAUUSD')
        plt.legend()
        plt.grid(True)
        plt.show()

