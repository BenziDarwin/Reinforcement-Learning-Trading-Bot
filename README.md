# XAUUSD (Gold) Trading Bot

A deep evolution strategy-based trading bot for XAUUSD (Gold) using MetaTrader 5. The bot uses neural networks optimized through evolutionary strategies to make trading decisions.

## Project Structure

```
xauusd_trading_bot/
│
├── data/               # Data handling and storage
│   └── __init__.py
│
├── models/            # Neural network and evolution strategy implementations
│   ├── __init__.py
│   ├── deep_evolution.py
│   └── neural_model.py
│
├── agents/            # Trading agent implementation
│   ├── __init__.py
│   └── evolution_agent.py
│
├── utils/             # Utility functions
│   ├── __init__.py
│   └── state_utils.py
│
├── config/            # Configuration settings
│   ├── __init__.py
│   └── trading_config.py
│
├── requirements.txt   # Project dependencies
├── main.py           # Main execution script
└── README.md         # This file
```

## Features

- Deep Evolution Strategy for model optimization
- Neural network-based decision making
- Risk management and position sizing
- Historical data analysis and backtesting
- Real-time trading capabilities through MetaTrader 5
- Performance metrics tracking
- Model saving and loading functionality

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd xauusd_trading_bot
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Install MetaTrader 5 and set up your account

## Configuration

Edit `config/trading_config.py` to adjust:
- Initial balance
- Maximum positions and lot sizes
- Risk per trade
- Evolution strategy parameters
- Neural network architecture
- Trading timeframe settings

## Usage

1. Training the model:
```bash
python main.py
```

This will:
- Fetch historical XAUUSD data
- Train the model using deep evolution strategy
- Save the trained model
- Display performance metrics

2. Running in live mode:
- Uncomment the live trading section in `main.py`
- Configure your MetaTrader 5 credentials
- Run the script

## Warning

⚠️ **Use at your own risk. This is an experimental trading bot and should not be used with real money without thorough testing and understanding of the risks involved.**

## Performance Metrics

The bot tracks:
- Final balance
- Return on Investment (ROI)
- Total number of trades
- Win rate
- Trade history

## Requirements

- Python 3.8+
- MetaTrader 5
- Dependencies listed in requirements.txt:
  - numpy==1.26.4
  - pandas==2.2.3
  - matplotlib==3.9.2
  - seaborn==0.13.2
  - MetaTrader5==5.0.45
  - python-dotenv==1.0.0

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is for educational purposes only. Do not risk money which you are afraid to lose. USE THE SOFTWARE AT YOUR OWN RISK. THE AUTHORS AND ALL AFFILIATES ASSUME NO RESPONSIBILITY FOR YOUR TRADING RESULTS.