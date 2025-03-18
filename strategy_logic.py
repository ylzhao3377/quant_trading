import numpy as np
import pandas as pd
from datetime import datetime


class StrategyLogic:
    """
    Centralized strategy logic that can be shared between portfolio_strategy.py and trade_advisor.py

    This class contains the core signal generation logic for different trading strategies,
    allowing for consistent strategy implementation across the system.
    """

    @staticmethod
    def compute_trend_strength(data, params):
        """
        Calculate trend strength based on technical indicators.

        Args:
            data (pd.Series or dict): Data point with technical indicators
            params (dict): Strategy parameters

        Returns:
            float: Trend strength (0.0 to 1.0)
        """
        price = data['close']
        adx = data['ADX']
        atr = data['ATR']
        atr_pct = atr / price if price > 0 else 0

        volume_spike = data['volume'] > data['Volume_MA'] * params.get("vol_threshold", 1.5)
        breakout_confirmed = (adx > params.get("adx_threshold", 30)) or volume_spike
        trend_up = (data['EMA50'] > data['EMA200']) and (adx > 20)

        if trend_up and (price > data['Upper']) and breakout_confirmed and (
                atr_pct > params.get("atr_pct_threshold", 0.005)):
            return 1.0

        return 0.0

    @staticmethod
    def compute_range_strength(data, params):
        """
        Calculate range strength based on price deviation from mean.

        Args:
            data (pd.Series or dict): Data point with technical indicators
            params (dict): Strategy parameters

        Returns:
            float: Range strength (0.0 to 1.0)
        """
        price = data['close']
        middle = data['Middle']
        std = data['STD']

        if std <= 0:
            return 0.0

        z = (price - middle) / std
        k, c = 3, 2
        strength = 1 / (1 + np.exp(-k * (abs(z) - c)))

        return strength

    @staticmethod
    def generate_trend_signal(data, params, current_position=0, max_profit_price=None):
        """
        Generate signal for trend following strategy.

        Args:
            data (pd.Series or dict): Data point with technical indicators
            params (dict): Strategy parameters
            current_position (float): Current position size
            max_profit_price (float, optional): Maximum profit price for trailing stop

        Returns:
            tuple: (signal, reason, details)
        """
        price = data['close']
        adx = data['ADX']
        atr = data['ATR']
        atr_pct = atr / price if price > 0 else 0
        volume_spike = data['volume'] > data['Volume_MA'] * params.get("vol_threshold", 1.5)
        breakout_confirmed = (adx > params.get("adx_threshold", 30)) or volume_spike
        trend_up = (data['EMA50'] > data['EMA200']) and (adx > 20)

        # BUY signal
        if current_position == 0:
            if (trend_up and (price > data['Upper']) and
                    breakout_confirmed and
                    (atr_pct > params.get("atr_pct_threshold", 0.005))):
                return "BUY", f"Trend breakout confirmed (ADX: {adx:.1f}, Vol spike: {volume_spike})", {
                    "strength": 1.0,
                    "adx": adx,
                    "atr_pct": atr_pct,
                    "volume_spike": volume_spike
                }
            return "HOLD", "No trend breakout condition met", {"strength": 0.0}

        # SELL signals (if we have a position)
        trailing_stop_pct = params.get("trailing_stop_pct", 0.95)

        # If max_profit_price is provided, check trailing stop
        if max_profit_price is not None:
            trailing_stop_price = max_profit_price * trailing_stop_pct
            if price <= trailing_stop_price:
                return "SELL", f"Trailing stop triggered ({trailing_stop_pct:.0%} of ${max_profit_price:.2f})", {
                    "strength": 0.8,
                    "stop_type": "trailing"
                }

        # Other sell conditions
        if price < data['Middle']:
            return "SELL", "Price below Bollinger Middle Band (Take Profit)", {
                "strength": 0.7,
                "stop_type": "take_profit"
            }
        elif price < data['EMA200']:
            return "SELL", "Price below EMA200 (Stop Loss)", {
                "strength": 0.9,
                "stop_type": "stop_loss"
            }
        else:
            return "HOLD", f"Maintaining trend position (Price: {price:.2f})", {
                "strength": 0.0,
                "current_price": price
            }

    @staticmethod
    def generate_range_signal(data, params, current_position=0, last_buy_price=None, last_sell_price=None):
        """
        Generate signal for range trading strategy.

        Args:
            data (pd.Series or dict): Data point with technical indicators
            params (dict): Strategy parameters
            current_position (float): Current position size
            last_buy_price (float, optional): Last buy price
            last_sell_price (float, optional): Last sell price

        Returns:
            tuple: (signal, reason, details)
        """
        price = data['close']
        lower_band = data['Lower']
        upper_band = data['Upper']
        middle_band = data['Middle']
        std = data['STD']

        # Calculate Z-score and breakout strength
        z_score = (price - middle_band) / std if std > 0 else 0
        k, c = 3, 2
        breakout_strength = 1 / (1 + np.exp(-k * (abs(z_score) - c)))
        min_strength = params.get("min_strength", 0.3)
        min_drop_pct = params.get("min_drop_pct", 0.01)
        min_rise_pct = params.get("min_rise_pct", 0.01)

        # BUY signal - price below lower band with sufficient strength
        if price < lower_band and breakout_strength >= min_strength:
            # Check price drop if last buy price is available
            if last_buy_price is None or price < last_buy_price * (1 - min_drop_pct):
                buy_reason = f"Price below lower band (Z-score: {z_score:.2f}, Strength: {breakout_strength:.2f})"
                return "BUY", buy_reason, {
                    "strength": breakout_strength,
                    "z_score": z_score,
                    "band": "lower"
                }

        # SELL signal - price above upper band with sufficient strength and we have a position
        if price > upper_band and breakout_strength >= min_strength and current_position > 0:
            # Check price rise if last sell price is available
            if last_sell_price is None or price > last_sell_price * (1 + min_rise_pct):
                sell_reason = f"Price above upper band (Z-score: {z_score:.2f}, Strength: {breakout_strength:.2f})"
                return "SELL", sell_reason, {
                    "strength": breakout_strength,
                    "z_score": z_score,
                    "band": "upper"
                }

        # HOLD signal - price within bands or insufficient strength for breakout
        return "HOLD", f"Price within range bands (Current Z-score: {z_score:.2f})", {
            "strength": 0.0,
            "z_score": z_score,
            "band": "middle" if (price >= lower_band and price <= upper_band) else "outside"
        }