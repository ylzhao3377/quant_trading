import pandas as pd
import numpy as np
import ast
import os
import smtplib
from email.message import EmailMessage
from datetime import datetime, timedelta
import pytz

# Import shared components from strategy_logic.py (renamed from simple_strategy.py)
from strategy_logic import (
    fetch_stock_data,
    compute_indicators,
    compute_trend_strength,
    compute_range_strength,
    run_trend_sell_logic,
    run_range_sell_logic,
    CapitalPool,
    get_available_funds,
    HiddenPrints
)

# Environment configuration
os.environ["POLYGON_API_KEY"] = "0Fp6qkxgz6QugnvLPiR6d9cEMpK3hxFF"


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


class EmailNotifier:
    """
    Handles email notifications with optional attachments.
    """

    @staticmethod
    def send_email(subject, message, from_addr, to_addrs, smtp_server, smtp_port, smtp_user, smtp_password,
                   attachments=None):
        """
        Send an email with optional attachments.

        Args:
            subject (str): Email subject
            message (str): Email body content
            from_addr (str): Sender email address
            to_addrs (list): List of recipient email addresses
            smtp_server (str): SMTP server address
            smtp_port (int): SMTP server port
            smtp_user (str): SMTP username
            smtp_password (str): SMTP password
            attachments (list, optional): List of file paths to attach
        """
        # Create email object
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = from_addr
        msg["To"] = ", ".join(to_addrs)
        msg.set_content(message)

        # Add attachments
        if attachments:
            for file_path in attachments:
                try:
                    with open(file_path, "rb") as f:
                        file_data = f.read()
                        file_name = os.path.basename(file_path)
                    # Add attachment
                    msg.add_attachment(file_data, maintype="application", subtype="octet-stream", filename=file_name)
                except Exception as e:
                    print(f"Unable to attach file {file_path}: {e}")

        # Send email
        try:
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()  # Enable TLS encryption
                server.login(smtp_user, smtp_password)
                server.send_message(msg)
            print("Email sent successfully!")
        except Exception as e:
            print(f"Failed to send email: {e}")


class SubAccount:
    """
    Represents a trading account for a specific asset and strategy.
    """

    def __init__(self, name, strategy_type, allocation_pct, capital_pool, params):
        """
        Initialize a sub-account.

        Args:
            name (str): Account name (typically ticker symbol)
            strategy_type (str): Strategy type ('trend' or 'range')
            allocation_pct (float): Capital allocation percentage
            capital_pool (CapitalPool): Reference to the capital pool
            params (dict): Strategy parameters
        """
        self.name = name
        self.strategy_type = strategy_type.lower()
        self.allocation_pct = allocation_pct
        self.capital_pool = capital_pool
        self.params = params
        self.balance = capital_pool.total_capital * allocation_pct / 100
        self.trade_log = []
        self.last_price = None  # For handling missing data

        # Initialize strategy-specific attributes
        if self.strategy_type == 'trend':
            self.in_position = False
            self.position = 0
            self.entry_price = 0
            self.max_profit_price = 0
            self.realized_profit = 0
        elif self.strategy_type == 'range':
            self.base_position = 0
            self.float_position = 0
            self.base_entry_prices = []
            self.float_entry_prices = []
            self.last_buy_price = None
            self.last_sell_price = None
            self.realized_profit = 0

    def update_from_last_state(self, last_state):
        """
        Update account state from the last known state.

        Args:
            last_state (dict): The last state information from previous logs
        """
        # Update balance
        if 'Balance' in last_state:
            self.balance = last_state['Balance']

        # Update realized profit
        if 'Realized_Profit' in last_state:
            self.realized_profit = last_state['Realized_Profit']

        # Update strategy-specific attributes
        if self.strategy_type == 'trend':
            if 'Position' in last_state and last_state['Position'] > 0:
                self.in_position = True
                self.position = last_state['Position']
                self.entry_price = last_state.get('Price', 0)
                self.max_profit_price = last_state.get('Price', 0)
            else:
                self.in_position = False
                self.position = 0
                self.entry_price = 0
                self.max_profit_price = 0

        elif self.strategy_type == 'range':
            if 'Position' in last_state:
                self.base_position = last_state['Position']
                # Initialize entry prices (estimating with the last known price)
                price = last_state.get('Price', 0)
                if price > 0 and self.base_position > 0:
                    self.base_entry_prices = [price] * self.base_position
                self.float_position = 0  # Assume no float position initially
                self.last_buy_price = price if self.base_position > 0 else None
                self.last_sell_price = None


class SynchronizedPortfolio:
    """
    Manages a portfolio of multiple trading accounts with synchronized execution.
    """

    def __init__(self, total_capital):
        """
        Initialize a synchronized portfolio.

        Args:
            total_capital (float): Total capital for the portfolio
        """
        self.capital_pool = CapitalPool(total_capital)
        self.accounts = {}  # {ticker: {'account': SubAccount, 'data': DataFrame}}
        self.trade_log = []
        self.snapshot_log = []
        self.last_date = None  # Track the last processed date

    def add_account(self, name, strategy_type, params, df, allocation_pct):
        """
        Add a trading account to the portfolio.

        Args:
            name (str): Account name (typically ticker symbol)
            strategy_type (str): Strategy type ('trend' or 'range')
            params (dict): Strategy parameters
            df (pd.DataFrame): Price data
            allocation_pct (float): Capital allocation percentage
        """
        df = df.reset_index()
        df['datetime'] = pd.to_datetime(df['datetime'])

        # Compute technical indicators using the function from simple_strategy.py
        window = params.get("bollinger_window", 50)
        df = compute_indicators(df, bollinger_window=window)
        df.sort_values('datetime', inplace=True)

        # Allocate capital
        allocated = self.capital_pool.allocate(self.capital_pool.total_capital * allocation_pct / 100)
        account = SubAccount(name, strategy_type, allocation_pct, self.capital_pool, params)
        account.balance = allocated

        # Initialize strategy-specific attributes
        if strategy_type.lower() == 'range':
            account.base_position = 0
            account.float_position = 0
            account.base_entry_prices = []
            account.float_entry_prices = []
            account.last_buy_price = None
            account.last_sell_price = None
            account.realized_profit = 0
        elif strategy_type.lower() == 'trend':
            account.in_position = False
            account.position = 0
            account.entry_price = 0
            account.max_profit_price = 0
            account.realized_profit = 0

        self.accounts[name] = {'account': account, 'data': df}

    def load_from_existing_logs(self, combined_log_path, snapshot_log_path):
        """
        Load portfolio state from existing log files.

        Args:
            combined_log_path (str): Path to combined trade log file
            snapshot_log_path (str): Path to snapshot log file

        Returns:
            bool: True if state was successfully loaded, False otherwise
        """
        try:
            # Check if both files exist
            if not os.path.exists(combined_log_path) or not os.path.exists(snapshot_log_path):
                print("One or more log files do not exist. Starting fresh.")
                return False

            # Load the combined trade log
            combined_df = pd.read_csv(combined_log_path)
            if combined_df.empty:
                return False

            # Convert date columns to datetime
            combined_df['Date'] = pd.to_datetime(combined_df['Date'])

            # Load the snapshot log
            snapshot_df = pd.read_csv(snapshot_log_path)
            if snapshot_df.empty:
                return False

            # Convert date columns to datetime
            snapshot_df['Date'] = pd.to_datetime(snapshot_df['Date'])

            # Find the last date in the logs
            self.last_date = max(combined_df['Date'].max(), snapshot_df['Date'].max())
            print(f"Loading portfolio state as of {self.last_date}")

            # Load existing trade logs to preserve history
            self.trade_log = combined_df.to_dict('records')
            self.snapshot_log = snapshot_df.to_dict('records')

            # Get the last state for each stock from the combined log
            stocks = set(combined_df['Stock'])
            for stock in stocks:
                stock_df = combined_df[combined_df['Stock'] == stock]
                if not stock_df.empty:
                    last_state = stock_df.iloc[-1].to_dict()

                    # If this stock exists in the current accounts, update its state
                    if stock in self.accounts:
                        self.accounts[stock]['account'].update_from_last_state(last_state)
                        print(f"Updated {stock} position from existing logs")

            return True

        except Exception as e:
            print(f"Error loading existing logs: {e}")
            return False

    def run(self, start_from_last_date=False):
        """
        Run the portfolio simulation across all accounts.

        Args:
            start_from_last_date (bool): If True, only process data after the last date in the logs
        """
        # Build unified timeline by merging all datetime columns
        all_times = sorted(set().union(*[set(data['data']['datetime']) for data in self.accounts.values()]))

        # Filter for dates after the last processed date if requested
        if start_from_last_date and self.last_date is not None:
            all_times = [dt for dt in all_times if dt > self.last_date]
            print(f"Processing only new data from {self.last_date} onward")

        if not all_times:
            print("No new data to process.")
            return

        print(f"Processing {len(all_times)} time points from {all_times[0]} to {all_times[-1]}")
        n_times = len(all_times)

        for i, current_time in enumerate(all_times):
            # Calculate pre-operation total balance
            pre_total_balance = self.capital_pool.status() + sum(
                [data['account'].balance for data in self.accounts.values()])

            # === SELL PHASE ===
            for name, data in self.accounts.items():
                df = data['data']
                account = data['account']
                current_rows = df[df['datetime'] == current_time]

                if not current_rows.empty:
                    row = current_rows.iloc[0]

                    # Record pre-operation position
                    if account.strategy_type == 'trend':
                        pre_position = account.position
                        # Use the shared sell logic function from simple_strategy.py
                        log_entry = run_trend_sell_logic(account, row, account.params)
                    elif account.strategy_type == 'range':
                        pre_position = account.base_position + account.float_position
                        # Use the shared sell logic function from simple_strategy.py
                        log_entry = run_range_sell_logic(account, row, account.params)
                    else:
                        log_entry = None

                    if log_entry and log_entry["Action"] != "HOLD":
                        # Add pre-operation total balance
                        log_entry["Pre_Total_Balance"] = pre_total_balance

                        # Record sell ratio
                        if pre_position > 0:
                            log_entry["Operation_Ratio"] = log_entry["Shares"] / pre_position
                        else:
                            log_entry["Operation_Ratio"] = 0

                        log_entry["Stock"] = name
                        self.trade_log.append(log_entry)
                        account.trade_log.append(log_entry)

            # === BUY PHASE: First collect all valid buy orders ===
            valid_orders = []
            total_strength = 0.0

            for name, data in self.accounts.items():
                df = data['data']
                account = data['account']
                current_rows = df[df['datetime'] == current_time]

                if not current_rows.empty:
                    row = current_rows.iloc[0]

                    if account.strategy_type == 'trend' and not account.in_position:
                        # Use the shared trend strength function from simple_strategy.py
                        strength = compute_trend_strength(row, account.params)
                        if strength > 0:
                            valid_orders.append((account, row, strength))
                            total_strength += strength
                    elif account.strategy_type == 'range':
                        if row['close'] < row['Lower']:
                            # Use the shared range strength function from simple_strategy.py
                            strength = compute_range_strength(row, account.params)
                            if strength >= account.params.get("min_strength", 0.3):
                                valid_orders.append((account, row, strength))
                                total_strength += strength

            # Execute buy orders with weight-based allocation
            for (account, row, strength) in valid_orders:
                weight = strength / total_strength if total_strength > 0 else 0
                allocated_funds = pre_total_balance * weight

                # Limit to current account balance
                funds_to_use = min(allocated_funds, account.balance * 2)
                price = row['close']
                balance_before = account.balance
                shares_to_buy = int(funds_to_use / price)

                if shares_to_buy > 0:
                    money_spent = shares_to_buy * price
                    # Calculate buy ratio: amount spent / pre-operation total balance
                    buy_ratio = money_spent / pre_total_balance

                    if account.strategy_type == 'trend':
                        account.entry_price = price
                        account.max_profit_price = price
                        account.position = shares_to_buy
                        account.balance -= money_spent
                        account.in_position = True
                        log_entry = {
                            "Date": row['datetime'],
                            "Action": "BUY",
                            "Reason": "Dynamic Buy Allocation (Trend)",
                            "Price": price,
                            "Shares": shares_to_buy,
                            "Balance": account.balance,
                            "Position": account.position,
                            "Realized_Profit": account.realized_profit,
                            "Pre_Total_Balance": pre_total_balance,
                            "Operation_Ratio": buy_ratio
                        }
                    elif account.strategy_type == 'range':
                        account.base_position += shares_to_buy
                        account.base_entry_prices.extend([price] * shares_to_buy)
                        account.balance -= money_spent
                        log_entry = {
                            "Date": row['datetime'],
                            "Action": "BUY",
                            "Reason": "Dynamic Buy Allocation (Range)",
                            "Price": price,
                            "Shares": shares_to_buy,
                            "Balance": account.balance,
                            "Position": account.base_position + account.float_position,
                            "Realized_Profit": account.realized_profit,
                            "Pre_Total_Balance": pre_total_balance,
                            "Operation_Ratio": buy_ratio
                        }
                    log_entry["Stock"] = account.name
                    self.trade_log.append(log_entry)
                    account.trade_log.append(log_entry)

            # === SNAPSHOT RECORDING ===
            current_day = pd.to_datetime(current_time).date()
            is_last_of_day = (i == n_times - 1) or (pd.to_datetime(all_times[i + 1]).date() != current_day)

            if valid_orders or is_last_of_day:
                snapshot = self.take_snapshot(current_time)
                self.snapshot_log.append(snapshot)

        # Update the last date processed
        if all_times:
            self.last_date = all_times[-1]

        # --- Add final position state (Final HOLD) ---
        if all_times:
            final_time = all_times[-1]

            # Calculate final pre-operation total balance
            final_pre_total_balance = self.capital_pool.status() + sum(
                data['account'].balance for data in self.accounts.values()
            )

            for name, data in self.accounts.items():
                account = data['account']
                df = data['data']

                # Get latest data row
                latest_row = df.iloc[-1]
                price = latest_row['close']

                if account.strategy_type == 'trend':
                    final_position = account.position
                    final_balance = account.balance
                elif account.strategy_type == 'range':
                    final_position = account.base_position + account.float_position
                    final_balance = account.balance

                final_log = {
                    "Date": latest_row['datetime'],
                    "Action": "HOLD_FINAL",
                    "Reason": "Final holding state",
                    "Price": price,
                    "Shares": final_position,
                    "Balance": final_balance,
                    "Position": final_position,
                    "Realized_Profit": account.realized_profit,
                    "Pre_Total_Balance": final_pre_total_balance,
                    "Operation_Ratio": 0,  # No trade ratio for final state
                    "Stock": name
                }
                self.trade_log.append(final_log)

    def take_snapshot(self, current_time):
        """
        Take a snapshot of the portfolio state at the current time.

        Args:
            current_time (datetime): Current time for the snapshot

        Returns:
            dict: Portfolio snapshot
        """
        snapshot = {"Date": current_time}
        # Add the PST version of time
        current_time_pst = convert_to_pst(current_time)
        snapshot["Date_PST"] = current_time_pst

        total_position_value = 0
        total_realized_profit = 0

        for name, data in self.accounts.items():
            account = data['account']
            df = data['data']
            current_rows = df[df['datetime'] == current_time]

            if not current_rows.empty:
                current_price = current_rows.iloc[0]['close']
                # Update last price
                account.last_price = current_price
            else:
                # Use last recorded price if data is missing
                current_price = account.last_price if account.last_price is not None else np.nan

            pos = account.position if account.strategy_type == "trend" else account.base_position
            snapshot[f"Position_{name}"] = pos
            snapshot[f"Price_{name}"] = current_price
            total_position_value += pos * (current_price if pd.notna(current_price) else 0)
            total_realized_profit += account.realized_profit

        total_account_balance = sum(data['account'].balance for data in self.accounts.values())
        snapshot["Total_Balance"] = self.capital_pool.status() + total_account_balance
        snapshot["Total_Equity"] = snapshot["Total_Balance"] + total_position_value
        snapshot["Total_Realized_Profit"] = total_realized_profit

        return snapshot

    def combined_trade_log(self):
        """
        Get combined trade log for all accounts.

        Returns:
            pd.DataFrame: Combined trade log
        """
        if self.trade_log:
            combined_df = pd.DataFrame(self.trade_log)
            combined_df["Date"] = pd.to_datetime(combined_df["Date"], errors="coerce")

            # Add PST date column
            combined_df["Date_PST"] = combined_df["Date"].apply(convert_to_pst)
            combined_df["Date_PST_Str"] = combined_df["Date_PST"].dt.strftime('%Y-%m-%d %H:%M:%S %Z')

            combined_df = combined_df.sort_values("Date").reset_index(drop=True)
            return combined_df
        return pd.DataFrame()

    def snapshot_log_df(self):
        """
        Get snapshot log as DataFrame.

        Returns:
            pd.DataFrame: Snapshot log
        """
        df = pd.DataFrame(self.snapshot_log)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

        # Add PST date column
        df["Date_PST"] = df["Date"].apply(convert_to_pst)
        df["Date_PST_Str"] = df["Date_PST"].dt.strftime('%Y-%m-%d %H:%M:%S %Z')

        df = df.sort_values("Date").reset_index(drop=True)
        return df

    def append_to_existing_logs(self, combined_log_path, snapshot_log_path):
        """
        Append new trade and snapshot data to existing log files.

        Args:
            combined_log_path (str): Path to combined trade log file
            snapshot_log_path (str): Path to snapshot log file

        Returns:
            tuple: (new_combined_df, old_combined_df) - DataFrames containing new and old records
        """
        combined_df = self.combined_trade_log()
        snapshot_df = self.snapshot_log_df()

        # Check if files exist
        combined_exists = os.path.exists(combined_log_path)
        snapshot_exists = os.path.exists(snapshot_log_path)

        existing_combined = pd.DataFrame()  # Initialize with empty DataFrame
        new_combined_df = combined_df.copy()  # Make a copy of the new data

        if combined_exists:
            try:
                # Load existing combined log
                existing_combined = pd.read_csv(combined_log_path)
                existing_combined['Date'] = pd.to_datetime(existing_combined['Date'])

                # Remove any overlap (based on Date and Stock columns)
                if not combined_df.empty and not existing_combined.empty:
                    # Create a unique key for merging
                    combined_df['merge_key'] = combined_df['Date'].astype(str) + '_' + combined_df['Stock'].astype(str)
                    existing_combined['merge_key'] = existing_combined['Date'].astype(str) + '_' + existing_combined[
                        'Stock'].astype(str)

                    # Remove rows from existing_combined that have the same merge_key as in combined_df
                    merge_keys_to_remove = set(combined_df['merge_key'])
                    existing_combined = existing_combined[~existing_combined['merge_key'].isin(merge_keys_to_remove)]

                    # Remove the merge_key column
                    existing_combined = existing_combined.drop(columns=['merge_key'])
                    combined_df = combined_df.drop(columns=['merge_key'])

                    # Combine the dataframes with proper handling of empty frames
                    if not existing_combined.empty:
                        combined_df = pd.concat([existing_combined, combined_df], ignore_index=True)
                    # If existing_combined is empty, combined_df is already correct
            except Exception as e:
                print(f"Error when processing combined logs: {e}")
                # If there's an error, keep the original combined_df

        if snapshot_exists:
            try:
                # Load existing snapshot log
                existing_snapshot = pd.read_csv(snapshot_log_path)
                existing_snapshot['Date'] = pd.to_datetime(existing_snapshot['Date'])

                # Remove any overlap (based on Date column only)
                if not snapshot_df.empty and not existing_snapshot.empty:
                    # Find the latest date in the existing snapshot
                    latest_date = existing_snapshot['Date'].max()

                    # Keep only rows from the new snapshot_df that are after the latest date
                    new_snapshot_rows = snapshot_df[snapshot_df['Date'] > latest_date]

                    # Combine the dataframes with proper handling of empty frames
                    if not existing_snapshot.empty:
                        snapshot_df = pd.concat([existing_snapshot, new_snapshot_rows], ignore_index=True)
                    # If existing_snapshot is empty, snapshot_df is already correct
            except Exception as e:
                print(f"Error when processing snapshot logs: {e}")
                # If there's an error, keep the original snapshot_df

        # Save the updated logs
        combined_df.to_csv(combined_log_path, index=False)
        snapshot_df.to_csv(snapshot_log_path, index=False)
        print(f"✅ Updated log files saved to {combined_log_path} and {snapshot_log_path}")

        # Return the new and existing DataFrames for use in generating the summary
        return (new_combined_df, existing_combined)


# Main execution block
if __name__ == "__main__":
    # Define file paths
    combined_log_path = "combined_trade_log.csv"
    snapshot_log_path = "snapshot_trade_log.csv"

    # Main process: Read strategy results and build portfolio
    results_df = pd.read_csv("strategy_results.csv")
    ticker_list = results_df["Stock"]
    filtered_results = results_df[
        (results_df['Stock'].isin(ticker_list)) &
        (results_df['Eligible'] == True) &
        (results_df['Allocation (%)'] > 0)
        ]

    print("Filtered results:")
    print(filtered_results[['Stock', 'Chosen Strategy', 'Best Params', 'Allocation (%)']])

    total_capital = 100000  # Set according to actual capital
    portfolio = SynchronizedPortfolio(total_capital=total_capital)

    # Check for existing log files
    existing_logs_exist = os.path.exists(combined_log_path) and os.path.exists(snapshot_log_path)

    # First add all accounts to the portfolio
    for idx, row in filtered_results.iterrows():
        ticker = row['Stock']
        strategy_type = row['Chosen Strategy'].lower().strip()  # 'trend' or 'range'
        best_params = ast.literal_eval(row['Best Params'])
        allocation_pct = float(row['Allocation (%)'])

        # Get the date range for fetching data
        now = datetime.now()
        end_date = now.strftime("%Y-%m-%d")

        # If we have existing logs, fetch data from a bit before the last date
        # to ensure we have enough data for technical indicators
        if existing_logs_exist:
            # Load the last processed date to determine the start date for new data
            try:
                existing_combined = pd.read_csv(combined_log_path)
                existing_combined['Date'] = pd.to_datetime(existing_combined['Date'])
                last_date = existing_combined['Date'].max()
                # Get data from 60 days before the last date to ensure proper indicator calculation
                start_date = (last_date - timedelta(days=60)).strftime("%Y-%m-%d")
            except Exception as e:
                print(f"Error determining start date from logs: {e}")
                start_date = "2025-03-13"  # Default start date
        else:
            # If no existing logs, use a fixed start date
            start_date = "2025-03-13"

        print(f"Fetching data for {ticker} from {start_date} to {end_date}")

        # Use fetch_stock_data from simple_strategy.py
        df = fetch_stock_data(ticker, start_date, end_date, "hour")
        if df.empty:
            print(f"{ticker} has no data, skipping.")
            continue

        portfolio.add_account(ticker, strategy_type, best_params, df, allocation_pct)
        print(f"Added {ticker}: strategy={strategy_type}, Allocation={allocation_pct}%")

    # If existing logs exist, load the portfolio state
    start_from_last_date = False
    if existing_logs_exist:
        print("Loading portfolio state from existing logs...")
        if portfolio.load_from_existing_logs(combined_log_path, snapshot_log_path):
            start_from_last_date = True
            print("Successfully loaded portfolio state from existing logs.")
        else:
            print("Could not load portfolio state from existing logs. Starting fresh.")

    # Run the portfolio with or without starting from the last date
    portfolio.run(start_from_last_date=start_from_last_date)

    # Append to or create new log files
    try:
        if existing_logs_exist and start_from_last_date:
            new_combined_df, old_combined_df = portfolio.append_to_existing_logs(combined_log_path, snapshot_log_path)
            print("✅ Appended new trades to existing log files")
        else:
            # Save as new files
            combined_log = portfolio.combined_trade_log()
            combined_log.to_csv(combined_log_path, index=False)
            print(f"✅ Combined trade log saved to {combined_log_path}")

            snapshot_df = portfolio.snapshot_log_df()
            snapshot_df.to_csv(snapshot_log_path, index=False)
            print(f"✅ Snapshot log saved to {snapshot_log_path}")

            new_combined_df = combined_log
            old_combined_df = pd.DataFrame()  # Empty DataFrame as there was no previous data
    except Exception as e:
        print(f"Error during file operations: {e}")
        # In case of error, try to get the data directly from the portfolio
        new_combined_df = portfolio.combined_trade_log()
        old_combined_df = pd.DataFrame()  # Empty DataFrame on error

    # Email configuration (replace smtp_password with your app-specific password)
    from_addr = "ylzhao3377@gmail.com"
    to_addrs = ["zhaoyilin3377@gmail.com"]
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    smtp_user = "ylzhao3377@gmail.com"
    smtp_password = "pntr minq hlcb uikz"  # Recommend using app-specific password

    # Create trimmed versions of the log files (last 200 rows only)
    try:
        # For combined trade log
        full_combined_log = pd.read_csv(combined_log_path)
        trimmed_combined_log = full_combined_log.tail(200)  # Get last 200 rows
        trimmed_combined_path = "combined_trade_log_latest.csv"
        trimmed_combined_log.to_csv(trimmed_combined_path, index=False)

        # For snapshot log
        full_snapshot_log = pd.read_csv(snapshot_log_path)
        trimmed_snapshot_log = full_snapshot_log.tail(200)  # Get last 200 rows
        trimmed_snapshot_path = "snapshot_trade_log_latest.csv"
        trimmed_snapshot_log.to_csv(trimmed_snapshot_path, index=False)

        # Update the message to indicate these are trimmed files
        message += "\nNote: The attached files contain only the latest 200 records for space efficiency."

        # Use the trimmed files for email attachments
        attachments = [trimmed_combined_path, trimmed_snapshot_path]
        print(f"✅ Created trimmed log files with the latest 200 records for email attachments")
    except Exception as e:
        print(f"Error creating trimmed log files: {e}")
        # Fallback to full files if trimming fails
        attachments = [combined_log_path, snapshot_log_path]
        print("⚠️ Using full log files for attachments due to error when trimming")

    # Email subject and content
    now_pst = convert_to_pst(datetime.now()).strftime("%Y-%m-%d %H:%M:%S %Z")
    subject = f"Strategy Update Results - {now_pst}"

    # Generate summary of new actions
    message = ""

    if existing_logs_exist and start_from_last_date and isinstance(old_combined_df, pd.DataFrame):
        # Calculate the maximum timestamp from the old log file
        old_combined_df['Date'] = pd.to_datetime(old_combined_df['Date'])
        max_old_timestamp = old_combined_df['Date'].max().timestamp() if not old_combined_df.empty else 0

        # Convert the date to a user-friendly format
        if max_old_timestamp > 0:
            last_date_obj = datetime.fromtimestamp(max_old_timestamp, pytz.UTC)
            last_date_formatted = convert_to_pst(last_date_obj).strftime("%Y-%m-%d %H:%M:%S %Z")
        else:
            last_date_formatted = "previous run"

        message = f"Portfolio updated with new trades since {last_date_formatted}.\n\n"

        # Debug information
        message += "DEBUG INFO:\n"
        message += f"Last timestamp from previous logs: {max_old_timestamp}\n"

        # Identify new actions (excluding the ones already in old_combined_df)
        try:
            # Create new DataFrame with only the new actions
            new_combined_df['Date'] = pd.to_datetime(new_combined_df['Date'])

            # Get only the truly new records (those not in old_combined_df)
            if not old_combined_df.empty:
                # Convert dates to Unix timestamps for reliable comparison
                new_combined_df['unix_timestamp'] = new_combined_df['Date'].apply(lambda x: x.timestamp())

                # Find all entries in new_combined_df that have timestamps after max_old_timestamp
                new_actions = new_combined_df[new_combined_df['unix_timestamp'] > max_old_timestamp]

                # Add more debug info
                message += f"Number of entries in new log: {len(new_combined_df)}\n"
                message += f"Number of entries with timestamp > {max_old_timestamp}: {len(new_actions)}\n"

                # List all timestamps for debugging
                if len(new_combined_df) < 20:  # Only if the list is manageable
                    message += "All timestamps in new log (Unix format):\n"
                    for ts in sorted(new_combined_df['unix_timestamp'].unique()):
                        date_str = datetime.fromtimestamp(ts, pytz.UTC)
                        message += f"- {ts} ({date_str})\n"
            else:
                new_actions = new_combined_df
                message += "No previous log entries found - treating all actions as new.\n"

            # Filter for only actual trades
            all_actions = new_actions.copy()
            new_actions = new_actions[new_actions['Action'].isin(['BUY', 'SELL'])]

            message += f"Total new actions: {len(all_actions)}\n"
            message += f"New trade actions (BUY/SELL only): {len(new_actions)}\n\n"

            # List all actions
            if len(all_actions) < 20:  # Only if the list is manageable
                message += "All new actions:\n"
                for idx, row in all_actions.iterrows():
                    action_time = pd.to_datetime(row['Date']).timestamp()
                    message += f"- {row['Stock']}: {row['Action']} @ {action_time} ({row['Date']})\n"
                message += "\n"

            # Build summary message
            if len(new_actions) > 0:
                message += "TRADE SUMMARY:\n"
                message += "==============\n\n"

                # Summarize by stock and action type
                stocks = new_actions['Stock'].unique()
                for stock in stocks:
                    stock_actions = new_actions[new_actions['Stock'] == stock]
                    message += f"📊 {stock}:\n"

                    # Buy actions
                    buys = stock_actions[stock_actions['Action'] == 'BUY']
                    if len(buys) > 0:
                        total_bought = buys['Shares'].sum()
                        avg_price = buys['Price'].mean()
                        total_value = (buys['Shares'] * buys['Price']).sum()
                        message += f"   🟢 BUY: {len(buys)} orders, {total_bought} shares @ avg ${avg_price:.2f} (${total_value:.2f} total)\n"

                    # Sell actions
                    sells = stock_actions[stock_actions['Action'] == 'SELL']
                    if len(sells) > 0:
                        total_sold = sells['Shares'].sum()
                        avg_price = sells['Price'].mean()
                        total_value = (sells['Shares'] * sells['Price']).sum()
                        message += f"   🔴 SELL: {len(sells)} orders, {total_sold} shares @ avg ${avg_price:.2f} (${total_value:.2f} total)\n"

                        # Only add realized profit info if we have sell orders
                        try:
                            if 'Realized_Profit' in sells.columns and len(sells) > 0:
                                first_profit = sells['Realized_Profit'].iloc[0] if not pd.isna(
                                    sells['Realized_Profit'].iloc[0]) else 0
                                last_profit = sells['Realized_Profit'].iloc[-1] if not pd.isna(
                                    sells['Realized_Profit'].iloc[-1]) else 0
                                realized_profit = last_profit - first_profit
                                if realized_profit != 0:
                                    message += f"   💰 Realized profit in this period: ${realized_profit:.2f}\n"
                        except Exception as e:
                            message += f"   ⚠️ Could not calculate realized profit: {str(e)}\n"

                    # Current position
                    try:
                        current_entries = new_combined_df[new_combined_df['Stock'] == stock]
                        if not current_entries.empty:
                            final_position = current_entries.iloc[-1]['Position']
                            final_price = current_entries.iloc[-1]['Price']
                            current_value = final_position * final_price
                            message += f"   📈 Current position: {final_position} shares @ ${final_price:.2f} (${current_value:.2f})\n\n"
                    except Exception as e:
                        message += f"   ⚠️ Could not determine current position: {str(e)}\n\n"

                # Overall portfolio summary
                try:
                    snapshot_df = portfolio.snapshot_log_df()
                    if not snapshot_df.empty:
                        last_snapshot = snapshot_df.iloc[-1]
                        total_balance = last_snapshot['Total_Balance']
                        total_equity = last_snapshot['Total_Equity']
                        message += f"PORTFOLIO SUMMARY:\n"
                        message += f"Total Balance: ${total_balance:.2f}\n"
                        message += f"Total Equity: ${total_equity:.2f}\n\n"
                except Exception as e:
                    message += f"⚠️ Could not generate portfolio summary: {str(e)}\n\n"
            else:
                message += "No new trades were executed in this update period.\n\n"

                # Still add portfolio summary
                try:
                    snapshot_df = portfolio.snapshot_log_df()
                    if not snapshot_df.empty:
                        last_snapshot = snapshot_df.iloc[-1]
                        total_balance = last_snapshot['Total_Balance']
                        total_equity = last_snapshot['Total_Equity']
                        message += f"PORTFOLIO SUMMARY:\n"
                        message += f"Total Balance: ${total_balance:.2f}\n"
                        message += f"Total Equity: ${total_equity:.2f}\n\n"
                except Exception as e:
                    message += f"⚠️ Could not generate portfolio summary: {str(e)}\n\n"
        except Exception as e:
            message += f"⚠️ Error generating trade summary: {str(e)}\n\n"

            # Add basic portfolio summary even if there's an error
            try:
                snapshot_df = portfolio.snapshot_log_df()
                if not snapshot_df.empty:
                    last_snapshot = snapshot_df.iloc[-1]
                    total_balance = last_snapshot['Total_Balance']
                    total_equity = last_snapshot['Total_Equity']
                    message += f"PORTFOLIO SUMMARY:\n"
                    message += f"Total Balance: ${total_balance:.2f}\n"
                    message += f"Total Equity: ${total_equity:.2f}\n\n"
            except:
                pass
    else:
        message = "New portfolio simulation completed.\n\n"

        # Add portfolio summary for new simulations too
        try:
            snapshot_df = portfolio.snapshot_log_df()
            if not snapshot_df.empty:
                last_snapshot = snapshot_df.iloc[-1]
                total_balance = last_snapshot['Total_Balance']
                total_equity = last_snapshot['Total_Equity']
                message += f"PORTFOLIO SUMMARY:\n"
                message += f"Total Balance: ${total_balance:.2f}\n"
                message += f"Total Equity: ${total_equity:.2f}\n\n"
        except Exception as e:
            message += f"⚠️ Could not generate portfolio summary: {str(e)}\n\n"

    message += "Please find attached the combined_trade_log.csv and snapshot_trade_log.csv files."

    # Attachment paths
    EmailNotifier.send_email(subject, message, from_addr, to_addrs, smtp_server, smtp_port, smtp_user, smtp_password,
                             attachments)

    # Clean up the trimmed files after sending email
    try:
        if "trimmed_combined_path" in locals() and os.path.exists(trimmed_combined_path):
            os.remove(trimmed_combined_path)
        if "trimmed_snapshot_path" in locals() and os.path.exists(trimmed_snapshot_path):
            os.remove(trimmed_snapshot_path)
        print("✅ Cleaned up temporary trimmed files")
    except Exception as e:
        print(f"Note: Could not clean up temporary files: {e}")