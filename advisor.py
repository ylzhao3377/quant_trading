import pandas as pd
import numpy as np
import os
import ast
from datetime import datetime, timedelta
import smtplib
from email.message import EmailMessage
import pytz

# Import shared components from strategy_logic.py (renamed from simple_strategy.py)
from strategy_logic import (
    fetch_stock_data,
    compute_indicators,
    compute_trend_strength,
    compute_range_strength,
    run_trend_sell_logic,
    run_range_sell_logic,
    get_available_funds
)
from portfolio_strategy import EmailNotifier


def convert_to_pst(dt):
    """
    Convert a UTC datetime to PST (Pacific Standard Time).

    Args:
        dt: datetime object or timestamp string

    Returns:
        datetime: PST datetime object
    """
    if isinstance(dt, str):
        dt = pd.to_datetime(dt)

    # If datetime is naive (no timezone), assume it's UTC
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=pytz.UTC)

    # Convert to PST
    pacific = pytz.timezone('US/Pacific')
    return dt.astimezone(pacific)


class PortfolioAdvisor:
    """
    Advises on portfolio actions based on latest data and current portfolio status.
    """

    def __init__(self, strategy_results_file, combined_log_file, snapshot_log_file):
        """
        Initialize the portfolio advisor.

        Args:
            strategy_results_file (str): Path to strategy results CSV file
            combined_log_file (str): Path to combined trade log CSV file
            snapshot_log_file (str): Path to snapshot log CSV file
        """
        self.strategy_results_file = strategy_results_file
        self.combined_log_file = combined_log_file
        self.snapshot_log_file = snapshot_log_file

        # Load data
        self.strategy_results = pd.read_csv(strategy_results_file)
        self.combined_log = pd.read_csv(combined_log_file)
        self.snapshot_log = pd.read_csv(snapshot_log_file)

        # Convert date columns - ensure we have both UTC and PST
        self.combined_log['Date'] = pd.to_datetime(self.combined_log['Date'])
        self.snapshot_log['Date'] = pd.to_datetime(self.snapshot_log['Date'])

        # Check if PST columns already exist; if not, create them
        if 'Date_PST' not in self.combined_log.columns:
            self.combined_log['Date_PST'] = self.combined_log['Date'].apply(convert_to_pst)
        else:
            # Ensure Date_PST is a datetime column
            self.combined_log['Date_PST'] = pd.to_datetime(self.combined_log['Date_PST'])

        if 'Date_PST' not in self.snapshot_log.columns:
            self.snapshot_log['Date_PST'] = self.snapshot_log['Date'].apply(convert_to_pst)
        else:
            # Ensure Date_PST is a datetime column
            self.snapshot_log['Date_PST'] = pd.to_datetime(self.snapshot_log['Date_PST'])

        # Format PST dates as strings if not already present
        if 'Date_PST_Str' not in self.combined_log.columns:
            self.combined_log['Date_PST_Str'] = self.combined_log['Date_PST'].dt.strftime('%Y-%m-%d %H:%M:%S %Z')

        if 'Date_PST_Str' not in self.snapshot_log.columns:
            self.snapshot_log['Date_PST_Str'] = self.snapshot_log['Date_PST'].dt.strftime('%Y-%m-%d %H:%M:%S %Z')

        # Initialize portfolio state
        self.current_positions = self._get_current_positions()
        self.total_balance = self._get_total_balance()
        self.recommendations = []

    def _get_current_positions(self):
        """
        Extract current positions from the latest entries in the combined log.

        Returns:
            dict: Current positions by stock ticker
        """
        positions = {}
        # Group by Stock and get the latest entry for each
        for stock, group in self.combined_log.groupby('Stock'):
            latest = group.sort_values('Date').iloc[-1]
            positions[stock] = {
                'position': latest['Position'],
                'strategy_type': self._get_strategy_type(stock),
                'params': self._get_strategy_params(stock),
                'in_position': latest['Position'] > 0,
                'entry_price': self._get_entry_price(stock, group),
                'max_profit_price': self._get_max_profit_price(stock, group),
                'realized_profit': latest['Realized_Profit'],
                'balance': latest['Balance']
            }

            # Add range-specific attributes if needed
            if positions[stock]['strategy_type'] == 'range':
                positions[stock].update({
                    'base_position': latest['Position'],  # Assuming Position includes both base and float
                    'float_position': 0,  # We don't have this info directly, assuming 0
                    'base_entry_prices': [self._get_entry_price(stock, group)] * int(latest['Position']),
                    'last_buy_price': self._get_last_buy_price(stock, group),
                    'last_sell_price': self._get_last_sell_price(stock, group)
                })

        return positions

    def _get_entry_price(self, stock, stock_log):
        """Find the entry price for current position"""
        # Filter for BUY actions that haven't been fully sold
        buy_actions = stock_log[(stock_log['Action'] == 'BUY')].sort_values('Date')
        if buy_actions.empty:
            return 0

        latest_sells = stock_log[(stock_log['Action'] == 'SELL')].sort_values('Date')
        if latest_sells.empty:
            # If no sells, return the latest buy price
            return buy_actions.iloc[-1]['Price']

        latest_buy_date = buy_actions.iloc[-1]['Date']
        latest_sell_date = latest_sells.iloc[-1]['Date']

        # If latest action is a sell, and there are no buys after that, no position
        if latest_sell_date > latest_buy_date:
            return 0

        # Return the price of the latest buy
        return buy_actions.iloc[-1]['Price']

    def _get_max_profit_price(self, stock, stock_log):
        """Find the maximum price since entry for trailing stop calculation"""
        entry_price = self._get_entry_price(stock, stock_log)
        if entry_price == 0:
            return 0

        # Find the entry date
        buy_actions = stock_log[(stock_log['Action'] == 'BUY')].sort_values('Date')
        if buy_actions.empty:
            return 0

        entry_date = buy_actions.iloc[-1]['Date']

        # Get all prices since entry
        prices_since_entry = stock_log[stock_log['Date'] >= entry_date]['Price']
        if prices_since_entry.empty:
            return entry_price

        return max(prices_since_entry.max(), entry_price)

    def _get_last_buy_price(self, stock, stock_log):
        """Get the last buy price for a stock (for range trading)"""
        buy_actions = stock_log[(stock_log['Action'] == 'BUY')].sort_values('Date')
        if buy_actions.empty:
            return None
        return buy_actions.iloc[-1]['Price']

    def _get_last_sell_price(self, stock, stock_log):
        """Get the last sell price for a stock (for range trading)"""
        sell_actions = stock_log[(stock_log['Action'] == 'SELL')].sort_values('Date')
        if sell_actions.empty:
            return None
        return sell_actions.iloc[-1]['Price']

    def _get_total_balance(self):
        """
        Get the total portfolio balance from the latest snapshot.

        Returns:
            float: Total portfolio balance
        """
        if self.snapshot_log.empty:
            return 0

        latest_snapshot = self.snapshot_log.sort_values('Date').iloc[-1]
        return latest_snapshot['Total_Balance']

    def _get_strategy_type(self, ticker):
        """
        Get the strategy type for a given ticker.

        Args:
            ticker (str): Stock ticker

        Returns:
            str: Strategy type ('trend' or 'range')
        """
        stock_info = self.strategy_results[self.strategy_results['Stock'] == ticker]
        if stock_info.empty:
            return 'trend'  # Default to trend if not found

        return stock_info.iloc[0]['Chosen Strategy'].lower().strip()

    def _get_strategy_params(self, ticker):
        """
        Get the strategy parameters for a given ticker.

        Args:
            ticker (str): Stock ticker

        Returns:
            dict: Strategy parameters
        """
        stock_info = self.strategy_results[self.strategy_results['Stock'] == ticker]
        if stock_info.empty:
            return {}  # Default to empty dict if not found

        try:
            return ast.literal_eval(stock_info.iloc[0]['Best Params'])
        except:
            return {}  # Return empty dict if parsing fails

    def analyze_latest_data(self, lookback_days=5):
        """
        Analyze stock data for all hours after the last timestamp in the combined log.

        Args:
            lookback_days (int): Number of days to look back for data if no log entries exist
        """
        eligible_stocks = self.strategy_results[self.strategy_results['Eligible'] == True]['Stock'].tolist()

        # Get current time in both UTC and PST
        now_utc = datetime.now(pytz.UTC)
        now_pst = convert_to_pst(now_utc) + timedelta(days=1)

        # Get the latest timestamp in the combined log (in UTC)
        if not self.combined_log.empty:
            latest_log_time = self.combined_log['Date'].max()
            # Add a small buffer to avoid duplicates (1 minute)
            start_date_utc = latest_log_time + timedelta(minutes=1)
            start_date_pst = convert_to_pst(start_date_utc)
            print(f"Starting analysis from the last log entry: {start_date_pst.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        else:
            # If no log entries exist, use lookback_days
            start_date_pst = now_pst - timedelta(days=lookback_days)
            print(f"No previous log entries found. Starting analysis from {lookback_days} days ago.")

        # Format dates for API calls
        start_date_str = start_date_pst.strftime("%Y-%m-%d")
        end_date_str = now_pst.strftime("%Y-%m-%d")

        print(f"Analyzing data from {start_date_str} to {end_date_str} (PST)")

        for ticker in eligible_stocks:
            try:
                # Get data since the last log entry
                df = fetch_stock_data(ticker, start_date_str, end_date_str, "hour")
                if df.empty:
                    print(f"No data available for {ticker}")
                    continue

                # Apply technical indicators
                strategy_type = self._get_strategy_type(ticker)
                params = self._get_strategy_params(ticker)
                window = params.get("bollinger_window", 50)
                df = compute_indicators(df, bollinger_window=window)

                # Sort by datetime to ensure chronological order
                df = df.sort_values('datetime')

                # Get current position info
                if ticker in self.current_positions:
                    position_info = self.current_positions[
                        ticker].copy()  # Create a copy to avoid modifying the original
                else:
                    # If we don't have this stock in our records, initialize it
                    position_info = {
                        'position': 0,
                        'strategy_type': strategy_type,
                        'params': params,
                        'in_position': False,
                        'entry_price': 0,
                        'max_profit_price': 0,
                        'realized_profit': 0,
                        'balance': 0
                    }

                # Filter out rows that are earlier than the last log entry
                if not self.combined_log.empty:
                    # Get the latest timestamp for this ticker
                    ticker_log = self.combined_log[self.combined_log['Stock'] == ticker]
                    if not ticker_log.empty:
                        ticker_latest_time = ticker_log['Date'].max()
                        df = df[df['datetime'] > ticker_latest_time]

                if df.empty:
                    print(f"No new data for {ticker} after the last log entry")
                    continue

                print(f"Analyzing {len(df)} new hours of data for {ticker}")

                # Process each hour of data
                for idx, row in df.iterrows():
                    # Create a mock account with the current position state
                    mock_account = type('MockAccount', (), position_info)

                    # Check sell recommendation first (if we have a position)
                    sell_recommendation = None
                    if position_info['in_position'] and position_info['position'] > 0:
                        if strategy_type == 'trend':
                            sell_recommendation = run_trend_sell_logic(mock_account, row, params)
                        elif strategy_type == 'range':
                            sell_recommendation = run_range_sell_logic(mock_account, row, params)

                        # Update position info from mock account after sell logic
                        if sell_recommendation:
                            position_info['position'] = mock_account.position
                            position_info['in_position'] = mock_account.in_position
                            position_info['entry_price'] = mock_account.entry_price
                            position_info['max_profit_price'] = mock_account.max_profit_price
                            position_info['realized_profit'] = mock_account.realized_profit
                            position_info['balance'] = mock_account.balance

                    # Check buy recommendation (if we don't have a position or for range trading)
                    buy_recommendation = None
                    if strategy_type == 'trend' and not position_info['in_position']:
                        trend_strength = compute_trend_strength(row, params)
                        print(trend_strength)
                        if trend_strength > 0:
                            # Calculate position size
                            allocation_pct = float(
                                self.strategy_results[self.strategy_results['Stock'] == ticker]['Allocation (%)'].iloc[
                                    0])
                            funds_to_use = self.total_balance * (allocation_pct / 100) * 0.9  # Use 90% of allocation
                            shares_to_buy = int(funds_to_use / row['close'])

                            if shares_to_buy > 0:
                                buy_recommendation = {
                                    "Date": row['datetime'],
                                    "Action": "BUY",
                                    "Reason": f"Trend strength: {trend_strength:.2f}",
                                    "Price": row['close'],
                                    "Shares": shares_to_buy,
                                    "Stock": ticker
                                }

                                # Update position info to reflect the buy
                                position_info['position'] = shares_to_buy
                                position_info['in_position'] = True
                                position_info['entry_price'] = row['close']
                                position_info['max_profit_price'] = row['close']
                                position_info['balance'] -= shares_to_buy * row['close']

                    elif strategy_type == 'range' and row['close'] < row['Lower']:
                        range_strength = compute_range_strength(row, params)
                        if range_strength >= params.get("min_strength", 0.3):
                            # Calculate position size
                            allocation_pct = float(
                                self.strategy_results[self.strategy_results['Stock'] == ticker]['Allocation (%)'].iloc[
                                    0])
                            funds_to_use = self.total_balance * (allocation_pct / 100) * 0.9 * range_strength
                            shares_to_buy = int(funds_to_use / row['close'])

                            if shares_to_buy > 0:
                                buy_recommendation = {
                                    "Date": row['datetime'],
                                    "Action": "BUY",
                                    "Reason": f"Range strength: {range_strength:.2f}",
                                    "Price": row['close'],
                                    "Shares": shares_to_buy,
                                    "Stock": ticker
                                }

                                # Update position info for range strategy
                                position_info['position'] += shares_to_buy
                                if 'base_position' in position_info:
                                    position_info['base_position'] += shares_to_buy
                                else:
                                    position_info['base_position'] = shares_to_buy

                                if 'base_entry_prices' in position_info:
                                    position_info['base_entry_prices'].extend([row['close']] * shares_to_buy)
                                else:
                                    position_info['base_entry_prices'] = [row['close']] * shares_to_buy

                                position_info['last_buy_price'] = row['close']
                                position_info['balance'] -= shares_to_buy * row['close']

                    # Add PST datetime to recommendations
                    if sell_recommendation:
                        utc_date = sell_recommendation["Date"]
                        # Convert to datetime if it's not already
                        if not isinstance(utc_date, pd.Timestamp) and not isinstance(utc_date, datetime):
                            utc_date = pd.to_datetime(utc_date)

                        # Add PST date
                        sell_recommendation["Date_PST"] = convert_to_pst(utc_date)
                        sell_recommendation["Stock"] = ticker
                        self.recommendations.append(sell_recommendation)

                    if buy_recommendation:
                        utc_date = buy_recommendation["Date"]
                        # Convert to datetime if it's not already
                        if not isinstance(utc_date, pd.Timestamp) and not isinstance(utc_date, datetime):
                            utc_date = pd.to_datetime(utc_date)

                        # Add PST date
                        buy_recommendation["Date_PST"] = convert_to_pst(utc_date)
                        self.recommendations.append(buy_recommendation)

                # Update the current positions with the final state
                self.current_positions[ticker] = position_info

            except Exception as e:
                print(f"Error analyzing {ticker}: {e}")
                import traceback
                traceback.print_exc()

        # Sort recommendations by datetime
        if self.recommendations:
            self.recommendations.sort(key=lambda x: x['Date'])
            print(f"Generated {len(self.recommendations)} recommendations")

    def generate_recommendations_report(self):
        """
        Generate a human-readable report of trading recommendations.

        Returns:
            str: Recommendations report
        """
        if not self.recommendations:
            return "No trading recommendations for the current time period."

        # Get current time in PST for the report header
        now_utc = datetime.now(pytz.UTC)
        now_pst = convert_to_pst(now_utc)
        pst_time_str = now_pst.strftime("%Y-%m-%d %H:%M:%S %Z")

        report = f"Trading Recommendations (Generated: {pst_time_str}):\n"
        report += "=" * 70 + "\n"

        for rec in sorted(self.recommendations, key=lambda x: x['Stock']):
            # Convert the date to PST if not already
            if 'Date_PST' in rec:
                pst_date = rec['Date_PST']
            else:
                pst_date = convert_to_pst(rec['Date'])

            # Format the date nicely if it's not already a string
            if not isinstance(pst_date, str):
                pst_date_str = pst_date.strftime('%Y-%m-%d %H:%M:%S %Z')
            else:
                pst_date_str = pst_date

            report += f"Stock: {rec['Stock']}\n"
            report += f"Action: {rec['Action']}\n"
            report += f"Reason: {rec['Reason']}\n"
            report += f"Price: ${rec['Price']:.2f}\n"
            report += f"Shares: {rec['Shares']}\n"
            report += f"Date (PST): {pst_date_str}\n"
            report += "-" * 40 + "\n"

        return report

    def update_logs(self):
        """
        Update the trade log and snapshot with the new recommendations.

        Returns:
            tuple: (updated_log, updated_snapshot) DataFrames
        """
        if not self.recommendations:
            print("No recommendations to update logs with.")
            return self.combined_log, self.snapshot_log

        # Update combined log
        new_entries = []
        for rec in self.recommendations:
            # Calculate estimated impact on portfolio
            stock_position = self.current_positions.get(rec['Stock'],
                                                        {'position': 0, 'balance': 0, 'realized_profit': 0})

            # Estimate new position and balance after trade
            if rec['Action'] == 'BUY':
                new_position = stock_position['position'] + rec['Shares']
                new_balance = stock_position['balance'] - (rec['Shares'] * rec['Price'])
                realized_profit = stock_position.get('realized_profit', 0)
            else:  # SELL
                new_position = stock_position['position'] - rec['Shares']
                new_balance = stock_position['balance'] + (rec['Shares'] * rec['Price'])

                # Calculate profit for sells
                if 'entry_price' in stock_position and stock_position['entry_price'] > 0:
                    profit = (rec['Price'] - stock_position['entry_price']) * rec['Shares']
                    realized_profit = stock_position.get('realized_profit', 0) + profit
                else:
                    realized_profit = stock_position.get('realized_profit', 0)

            # Ensure we have both UTC and PST dates
            if 'Date_PST' not in rec and 'Date' in rec:
                date_pst = convert_to_pst(rec['Date'])
            else:
                date_pst = rec['Date_PST']

            # Create the entry with both UTC and PST dates
            entry = {
                "Date": rec['Date'],
                "Date_PST": date_pst,
                "Date_PST_Str": date_pst.strftime('%Y-%m-%d %H:%M:%S %Z') if not isinstance(date_pst,
                                                                                            str) else date_pst,
                "Action": rec['Action'],
                "Reason": rec['Reason'],
                "Price": rec['Price'],
                "Shares": rec['Shares'],
                "Balance": new_balance,
                "Position": new_position,
                "Realized_Profit": realized_profit,
                "Stock": rec['Stock'],
                "Pre_Total_Balance": self.total_balance,
                "Operation_Ratio": rec['Shares'] / max(stock_position['position'], 1) if rec['Action'] == 'SELL' else 0
            }

            new_entries.append(entry)

        # Create DataFrames for new entries
        new_log_entries = pd.DataFrame(new_entries)

        # Make sure Date is datetime
        new_log_entries['Date'] = pd.to_datetime(new_log_entries['Date'])

        # Ensure Date_PST is datetime
        if 'Date_PST' in new_log_entries.columns and not isinstance(new_log_entries['Date_PST'].iloc[0], str):
            new_log_entries['Date_PST'] = pd.to_datetime(new_log_entries['Date_PST'])

        # Append to combined log
        updated_log = pd.concat([self.combined_log, new_log_entries], ignore_index=True)
        updated_log = updated_log.sort_values('Date').reset_index(drop=True)

        # Update snapshot
        latest_date = new_log_entries['Date'].max()
        new_snapshot = self._create_new_snapshot(latest_date, updated_log)
        updated_snapshot = pd.concat([self.snapshot_log, new_snapshot], ignore_index=True)
        updated_snapshot = updated_snapshot.sort_values('Date').reset_index(drop=True)

        # Save updated logs
        updated_log.to_csv(self.combined_log_file, index=False)
        updated_snapshot.to_csv(self.snapshot_log_file, index=False)

        print(f"✅ Updated combined log with {len(new_entries)} new entries")
        print(f"✅ Updated snapshot log with new snapshot")

        return updated_log, updated_snapshot

    def _create_new_snapshot(self, date, updated_log):
        """
        Create a new snapshot entry based on the latest trade log.

        Args:
            date (datetime): Date for the snapshot
            updated_log (pd.DataFrame): Updated trade log

        Returns:
            pd.DataFrame: New snapshot entry
        """
        snapshot = {"Date": date}

        # Add PST date
        date_pst = convert_to_pst(date)
        snapshot["Date_PST"] = date_pst
        snapshot["Date_PST_Str"] = date_pst.strftime('%Y-%m-%d %H:%M:%S %Z')

        total_position_value = 0
        total_realized_profit = 0

        # Get latest position info for each stock
        for stock, group in updated_log.groupby('Stock'):
            latest = group.sort_values('Date').iloc[-1]
            position = latest['Position']
            price = latest['Price']
            snapshot[f"Position_{stock}"] = position
            snapshot[f"Price_{stock}"] = price
            total_position_value += position * price
            total_realized_profit += latest['Realized_Profit']

        # Get the latest total balance from the previous snapshot
        if not self.snapshot_log.empty:
            prev_snapshot = self.snapshot_log.sort_values('Date').iloc[-1]
            prev_total_balance = prev_snapshot['Total_Balance']
        else:
            # If no previous snapshot, estimate from trade log
            prev_total_balance = updated_log.groupby('Stock').last()['Balance'].sum()

        # Adjust for new trades
        new_entries = updated_log[~updated_log['Date'].isin(self.combined_log['Date'])]
        for _, entry in new_entries.iterrows():
            if entry['Action'] == 'BUY':
                # Buys decrease cash balance
                prev_total_balance -= entry['Shares'] * entry['Price']
            elif entry['Action'] == 'SELL':
                # Sells increase cash balance
                prev_total_balance += entry['Shares'] * entry['Price']

        snapshot["Total_Balance"] = prev_total_balance
        snapshot["Total_Equity"] = prev_total_balance + total_position_value
        snapshot["Total_Realized_Profit"] = total_realized_profit

        return pd.DataFrame([snapshot])

    def send_recommendations_email(self, email_config):
        """
        Send email with recommendations and updated log attachments.

        Args:
            email_config (dict): Email configuration
        """
        # Generate the recommendations report
        report = self.generate_recommendations_report()

        # Get the last 200 rows of logs
        combined_log = pd.read_csv(self.combined_log_file)
        snapshot_log = pd.read_csv(self.snapshot_log_file)

        # Convert dates to PST if needed
        if 'Date' in combined_log.columns and 'Date_PST' not in combined_log.columns:
            combined_log['Date'] = pd.to_datetime(combined_log['Date'])
            combined_log['Date_PST'] = combined_log['Date'].apply(convert_to_pst)
            combined_log['Date_PST_Str'] = combined_log['Date_PST'].dt.strftime('%Y-%m-%d %H:%M:%S %Z')
        elif 'Date' in combined_log.columns and 'Date_PST' in combined_log.columns:
            # Ensure Date columns are datetime
            combined_log['Date'] = pd.to_datetime(combined_log['Date'])
            combined_log['Date_PST'] = pd.to_datetime(combined_log['Date_PST'])
            # Add string representation if not present
            if 'Date_PST_Str' not in combined_log.columns:
                combined_log['Date_PST_Str'] = combined_log['Date_PST'].dt.strftime('%Y-%m-%d %H:%M:%S %Z')

        if 'Date' in snapshot_log.columns and 'Date_PST' not in snapshot_log.columns:
            snapshot_log['Date'] = pd.to_datetime(snapshot_log['Date'])
            snapshot_log['Date_PST'] = snapshot_log['Date'].apply(convert_to_pst)
            snapshot_log['Date_PST_Str'] = snapshot_log['Date_PST'].dt.strftime('%Y-%m-%d %H:%M:%S %Z')
        elif 'Date' in snapshot_log.columns and 'Date_PST' in snapshot_log.columns:
            # Ensure Date columns are datetime
            snapshot_log['Date'] = pd.to_datetime(snapshot_log['Date'])
            snapshot_log['Date_PST'] = pd.to_datetime(snapshot_log['Date_PST'])
            # Add string representation if not present
            if 'Date_PST_Str' not in snapshot_log.columns:
                snapshot_log['Date_PST_Str'] = snapshot_log['Date_PST'].dt.strftime('%Y-%m-%d %H:%M:%S %Z')

        last_200_combined = combined_log.tail(200)
        last_200_snapshot = snapshot_log.tail(200)

        # Save the last 200 rows to temporary files
        temp_combined = "last_200_combined.csv"
        temp_snapshot = "last_200_snapshot.csv"

        last_200_combined.to_csv(temp_combined, index=False)
        last_200_snapshot.to_csv(temp_snapshot, index=False)

        # Create email subject with PST time
        now_utc = datetime.now(pytz.UTC)
        now_pst = convert_to_pst(now_utc)
        pst_time_str = now_pst.strftime("%Y-%m-%d %H:%M %Z")
        subject = f"Trading Recommendations - {pst_time_str}"

        # Create email message
        message = f"Trading Recommendations for {pst_time_str}\n\n"
        message += report
        message += "\n\nPlease find attached the last 200 rows of the trade log and snapshot log."

        # Send email
        EmailNotifier.send_email(
            subject=subject,
            message=message,
            from_addr=email_config['from_addr'],
            to_addrs=email_config['to_addrs'],
            smtp_server=email_config['smtp_server'],
            smtp_port=email_config['smtp_port'],
            smtp_user=email_config['smtp_user'],
            smtp_password=email_config['smtp_password'],
            attachments=[temp_combined, temp_snapshot]
        )

        # Clean up temporary files
        os.remove(temp_combined)
        os.remove(temp_snapshot)

        print("✅ Email sent with recommendations and log attachments")


# Main execution
if __name__ == "__main__":
    # Configuration
    strategy_results_file = "strategy_results.csv"
    combined_log_file = "combined_trade_log.csv"
    snapshot_log_file = "snapshot_trade_log.csv"

    # Email configuration
    email_config = {
        'from_addr': "ylzhao3377@gmail.com",
        'to_addrs': ["ylzhao3377@gmail.com"],
        'smtp_server': "smtp.gmail.com",
        'smtp_port': 587,
        'smtp_user': "ylzhao3377@gmail.com",
        'smtp_password': "pntr minq hlcb uikz"  # Use app-specific password
    }

    # Make sure the pytz library is installed
    try:
        import pytz
    except ImportError:
        print("pytz library not found. Installing...")
        import pip

        pip.main(['install', 'pytz'])
        import pytz

    # Create advisor
    advisor = PortfolioAdvisor(strategy_results_file, combined_log_file, snapshot_log_file)

    # Analyze latest data (past 5 days)
    advisor.analyze_latest_data(lookback_days=200)

    # Update logs
    advisor.update_logs()

    # Send recommendations email
    # advisor.send_recommendations_email(email_config)