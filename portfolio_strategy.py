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
            self.position = 0
            self.entry_prices = []
            self.last_buy_price = None
            self.last_sell_price = None
            self.realized_profit = 0


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
            account.position = 0
            account.entry_prices = []
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

    def save_state(self, state_file='portfolio_state.pkl'):
        """
        Save portfolio state to a pickle file for future runs.

        Args:
            state_file (str): Path to save the state file

        Returns:
            bool: True if state was successfully saved
        """
        try:
            # Create a state dictionary with all necessary information
            state = {
                'last_date': self.last_date,
                'capital_pool': {
                    'total_capital': self.capital_pool.total_capital,
                    'available_capital': self.capital_pool.available_capital
                },
                'trade_log': self.trade_log,
                'snapshot_log': self.snapshot_log,
                'accounts': {}
            }

            # Save state of each account
            for name, data in self.accounts.items():
                account = data['account']
                account_state = {
                    'name': name,
                    'strategy_type': account.strategy_type,
                    'allocation_pct': account.allocation_pct,
                    'balance': account.balance,
                    'realized_profit': account.realized_profit,
                    'last_price': account.last_price,
                    'params': account.params
                }

                # Add strategy-specific attributes
                if account.strategy_type == 'trend':
                    account_state.update({
                        'in_position': account.in_position,
                        'position': account.position,
                        'entry_price': account.entry_price,
                        'max_profit_price': account.max_profit_price
                    })
                elif account.strategy_type == 'range':
                    account_state.update({
                        'position': account.position,
                        'entry_prices': account.entry_prices,
                        'last_buy_price': account.last_buy_price,
                        'last_sell_price': account.last_sell_price
                    })

                state['accounts'][name] = account_state

            # Save to pickle file
            import pickle
            with open(state_file, 'wb') as f:
                pickle.dump(state, f)

            print(f"✅ Portfolio state saved to {state_file}")
            return True

        except Exception as e:
            print(f"❌ Error saving portfolio state: {e}")
            return False

    def load_state(self, state_file='portfolio_state.pkl'):
        """
        Load portfolio state from a pickle file.

        Args:
            state_file (str): Path to the state file

        Returns:
            bool: True if state was successfully loaded
        """
        if not os.path.exists(state_file):
            print(f"❌ State file {state_file} does not exist. Starting with fresh state.")
            return False

        try:
            # Load from pickle file
            import pickle
            with open(state_file, 'rb') as f:
                state = pickle.load(f)

            # Restore portfolio state
            self.last_date = state['last_date']
            self.trade_log = state['trade_log']
            self.snapshot_log = state['snapshot_log']

            # Restore capital pool
            self.capital_pool.total_capital = state['capital_pool']['total_capital']
            self.capital_pool.available_capital = state['capital_pool']['available_capital']

            # Restore accounts that exist in both the saved state and current portfolio
            for name, account_state in state['accounts'].items():
                if name in self.accounts:
                    account = self.accounts[name]['account']

                    # Update basic account properties
                    account.balance = account_state['balance']
                    account.realized_profit = account_state['realized_profit']
                    account.last_price = account_state['last_price']

                    # Update strategy parameters
                    account.params = account_state['params']

                    # Update strategy-specific attributes
                    if account.strategy_type == 'trend':
                        account.in_position = account_state['in_position']
                        account.position = account_state['position']
                        account.entry_price = account_state['entry_price']
                        account.max_profit_price = account_state['max_profit_price']
                    elif account.strategy_type == 'range':
                        account.position = account_state['position']
                        account.entry_prices = account_state['entry_prices']
                        account.last_buy_price = account_state['last_buy_price']
                        account.last_sell_price = account_state['last_sell_price']

                    print(f"✅ Restored state for {name} account")
                else:
                    print(f"⚠️ Account {name} from saved state not found in current portfolio")

            print(f"✅ Portfolio state loaded from {state_file}")
            return True

        except Exception as e:
            print(f"❌ Error loading portfolio state: {e}")
            return False

    def update_params(self, name, new_params):
        """
        Update strategy parameters for an account.

        Args:
            name (str): Account name
            new_params (dict): New strategy parameters

        Returns:
            bool: True if parameters were successfully updated
        """
        if name in self.accounts:
            self.accounts[name]['account'].params = new_params
            print(f"✅ Updated parameters for {name}")
            return True
        else:
            print(f"❌ Account {name} not found in portfolio")
            return False

    def run(self, start_from_last_date=True):
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
        # If we have a strategy start date defined, filter for dates after it
        elif hasattr(self, 'strategy_start_date'):
            all_times = [dt for dt in all_times if dt >= self.strategy_start_date]
            print(f"Processing only data from strategy start date {self.strategy_start_date} onward")

        if not all_times:
            print("No new data to process.")
            return

        print(f"Processing {len(all_times)} time points from {all_times[0]} to {all_times[-1]}")
        n_times = len(all_times)

        for i, current_time in enumerate(all_times):
            # Calculate pre-operation total balance
            pre_total_balance = self.capital_pool.status() + sum(
                [data['account'].balance for data in self.accounts.values()])

            # Track if any actions occurred in this timestep
            any_actions = False

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
                        pre_position = account.position
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
                        any_actions = True

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
                            # Calculate price drop from the middle band (as a percentage)
                            price_drop_pct = (row['Middle'] - row['close']) / row['Middle']
                            min_drop_pct = account.params.get("min_drop_pct", 0.01)

                            # Check if the drop meets minimum required percentage
                            if price_drop_pct >= min_drop_pct:
                                # Check if this is a first buy or if price dropped further from last buy
                                if account.last_buy_price is None or row['close'] < account.last_buy_price * (
                                        1 - min_drop_pct):
                                    # Use the shared range strength function from simple_strategy.py
                                    strength = compute_range_strength(row, account.params)
                                    if strength >= account.params.get("min_strength", 0.3):
                                        valid_orders.append((account, row, strength))
                                        total_strength += strength

            # Execute buy orders with weight-based allocation
            for (account, row, strength) in valid_orders:
                total_valid_orders = len(valid_orders)
                weight = strength / total_strength if total_strength > 0 else 0

                # Calculate total equity (sum of each account's cash balance + position value)
                total_equity = 0
                for name, data in self.accounts.items():
                    acct = data['account']
                    acct_df = data['data']
                    current_rows = acct_df[acct_df['datetime'] == current_time]
                    current_price = current_rows.iloc[0]['close'] if not current_rows.empty else 0
                    account_equity = acct.balance + (acct.position * current_price)
                    total_equity += account_equity

                # Add the capital pool's available funds
                total_equity += self.capital_pool.status()

                # Allocate funds based on total equity, stock's allocation percentage, and signal strength
                if account.strategy_type == 'trend':
                    funds_to_use = total_equity * (account.allocation_pct / 25) * weight
                else:
                    funds_to_use = total_equity * (account.allocation_pct / 80) * weight

                # Ensure we don't exceed current account balance
                funds_to_use = min(funds_to_use, pre_total_balance * weight)
                price = row['close']
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
                            "Reason": f"Dynamic Buy Allocation (Trend) - Strength: {strength:.2f}",
                            "Price": price,
                            "Shares": shares_to_buy,
                            "Balance": account.balance,
                            "Position": account.position,
                            "Realized_Profit": account.realized_profit,
                            "Pre_Total_Balance": pre_total_balance,
                            "Operation_Ratio": buy_ratio
                        }
                    elif account.strategy_type == 'range':
                        account.position += shares_to_buy
                        account.entry_prices.extend([price] * shares_to_buy)
                        account.balance -= money_spent
                        account.last_buy_price = price  # Update the last buy price
                        log_entry = {
                            "Date": row['datetime'],
                            "Action": "BUY",
                            "Reason": f"Dynamic Buy Allocation (Range) - Strength: {strength:.2f}",
                            "Price": price,
                            "Shares": shares_to_buy,
                            "Balance": account.balance,
                            "Position": account.position,
                            "Realized_Profit": account.realized_profit,
                            "Pre_Total_Balance": pre_total_balance,
                            "Operation_Ratio": buy_ratio
                        }
                    log_entry["Stock"] = account.name
                    self.trade_log.append(log_entry)
                    account.trade_log.append(log_entry)
                    any_actions = True

            # === SNAPSHOT RECORDING ===
            # Only take a snapshot when there's an action or it's the end of a day
            current_day = pd.to_datetime(current_time).date()
            is_last_of_day = (i == n_times - 1) or (pd.to_datetime(all_times[i + 1]).date() != current_day)

            if any_actions or is_last_of_day:
                snapshot = self.take_snapshot(current_time)
                self.snapshot_log.append(snapshot)

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

            pos = account.position
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


def generate_trade_report(portfolio, hours=96):
    """
    Generate a detailed report of all actions in the specified time period.

    Args:
        portfolio (SynchronizedPortfolio): Portfolio object
        hours (int): Number of hours to look back (default: 96 hours/4 days)

    Returns:
        str: Formatted report message
    """
    # Get current time in PST for reference
    now_utc = datetime.now(pytz.UTC)
    now_pst = convert_to_pst(now_utc)

    # Get specified hours ago in PST
    last_period = now_pst - timedelta(hours=hours)

    # Get trade log
    trade_log = portfolio.combined_trade_log()

    if trade_log.empty:
        return "No trades were executed in this simulation.\n"

    # Filter for actions in the specified time period (based on PST time)
    recent_actions = trade_log[trade_log['Date_PST'] >= last_period]

    # Start building the message
    message = f"📊 TRADING ACTIVITY REPORT ({now_pst.strftime('%Y-%m-%d %H:%M:%S %Z')})\n"
    message += "=" * 50 + "\n\n"

    if recent_actions.empty:
        message += f"No trading activity in the last {hours} hours.\n\n"
    else:
        message += f"🕒 LAST {hours} HOURS ACTIVITY SUMMARY\n"
        message += f"Total number of trades: {len(recent_actions)}\n"

        # Count buy and sell actions
        buys = recent_actions[recent_actions['Action'] == 'BUY']
        sells = recent_actions[recent_actions['Action'] == 'SELL']

        message += f"Buy orders: {len(buys)}\n"
        message += f"Sell orders: {len(sells)}\n\n"

        # Summarize by stock
        message += "ACTIVITY BY STOCK:\n"
        for stock in sorted(recent_actions['Stock'].unique()):
            stock_actions = recent_actions[recent_actions['Stock'] == stock]
            message += f"\n📈 {stock}:\n"

            # Buy summary
            stock_buys = stock_actions[stock_actions['Action'] == 'BUY']
            if not stock_buys.empty:
                total_shares_bought = stock_buys['Shares'].sum()
                avg_buy_price = stock_buys['Price'].mean()
                total_buy_value = (stock_buys['Shares'] * stock_buys['Price']).sum()

                message += f"  🟢 BUYS: {len(stock_buys)} orders\n"
                message += f"     Total shares: {total_shares_bought}\n"
                message += f"     Avg price: ${avg_buy_price:.2f}\n"
                message += f"     Total value: ${total_buy_value:.2f}\n"

                # List individual buy orders
                message += "     Transactions:\n"
                for _, row in stock_buys.iterrows():
                    pst_time = row['Date_PST_Str']
                    message += f"     - {pst_time}: {row['Shares']} shares @ ${row['Price']:.2f} (${row['Shares'] * row['Price']:.2f})\n"

            # Sell summary
            stock_sells = stock_actions[stock_actions['Action'] == 'SELL']
            if not stock_sells.empty:
                total_shares_sold = stock_sells['Shares'].sum()
                avg_sell_price = stock_sells['Price'].mean()
                total_sell_value = (stock_sells['Shares'] * stock_sells['Price']).sum()

                message += f"  🔴 SELLS: {len(stock_sells)} orders\n"
                message += f"     Total shares: {total_shares_sold}\n"
                message += f"     Avg price: ${avg_sell_price:.2f}\n"
                message += f"     Total value: ${total_sell_value:.2f}\n"

                # Realized profit
                if len(stock_sells) > 0 and len(stock_buys) > 0:
                    # Calculate realized profit directly from the transactions
                    total_buy_cost = (stock_buys['Shares'] * stock_buys['Price']).sum()
                    total_sell_value = (stock_sells['Shares'] * stock_sells['Price']).sum()
                    realized_profit = total_sell_value - total_buy_cost
                    message += f"     Realized profit: ${realized_profit:.2f}\n"
                elif len(stock_sells) > 0:
                    # Fallback to using the Realized_Profit column if available
                    realized_profit = stock_sells.iloc[-1]['Realized_Profit'] - stock_sells.iloc[0]['Realized_Profit']
                    message += f"     Realized profit: ${realized_profit:.2f}\n"

                # List individual sell orders
                message += "     Transactions:\n"
                for _, row in stock_sells.iterrows():
                    pst_time = row['Date_PST_Str']
                    message += f"     - {pst_time}: {row['Shares']} shares @ ${row['Price']:.2f} (${row['Shares'] * row['Price']:.2f})\n"
                    message += f"       Reason: {row['Reason']}\n"

    # Get the latest snapshot for current positions
    snapshot_df = portfolio.snapshot_log_df()
    if not snapshot_df.empty:
        latest_snapshot = snapshot_df.iloc[-1]

        message += "\n🏦 CURRENT PORTFOLIO STATUS\n"
        message += f"Timestamp: {latest_snapshot['Date_PST_Str']}\n"
        message += f"Total Balance: ${latest_snapshot['Total_Balance']:.2f}\n"
        message += f"Total Equity: ${latest_snapshot['Total_Equity']:.2f}\n"
        message += f"Total Realized Profit: ${latest_snapshot['Total_Realized_Profit']:.2f}\n\n"

        message += "CURRENT POSITIONS:\n"
        # Get all columns that start with "Position_"
        position_columns = [col for col in latest_snapshot.index if col.startswith("Position_")]
        price_columns = [col for col in latest_snapshot.index if col.startswith("Price_")]

        for pos_col in position_columns:
            ticker = pos_col.replace("Position_", "")
            price_col = f"Price_{ticker}"

            position = latest_snapshot[pos_col]
            if position > 0:
                price = latest_snapshot[price_col]
                value = position * price
                message += f"  {ticker}: {position} shares @ ${price:.2f} = ${value:.2f}\n"

    return message


# Main execution block
if __name__ == "__main__":
    # Define fixed start date when the strategy actually begins taking actions
    strategy_start_date = datetime(2024, 3, 15)  # March 13, 2025

    # Define file paths for saving logs
    combined_log_path = "combined_trade_log.csv"
    snapshot_log_path = "snapshot_trade_log.csv"
    state_file = "portfolio_state.pkl"

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

    # Try to load previous state
    previous_state_loaded = portfolio.load_state(state_file)

    # Add all accounts to the portfolio
    for idx, row in filtered_results.iterrows():
        ticker = row['Stock']
        strategy_type = row['Chosen Strategy'].lower().strip()  # 'trend' or 'range'
        best_params = ast.literal_eval(row['Best Params'])
        allocation_pct = float(row['Allocation (%)'])

        # Get the date range for fetching data
        # Always fetch data starting from 100 days before strategy start date
        # to properly calculate indicators
        start_date = (strategy_start_date - timedelta(days=100)).strftime("%Y-%m-%d")

        # Always fetch data up to the current date to ensure we have the latest information
        end_date = datetime.now().strftime("%Y-%m-%d")

        print(f"Fetching data for {ticker} from {start_date} to {end_date}")

        # Use fetch_stock_data from strategy_logic.py
        df = fetch_stock_data(ticker, start_date, end_date, "hour")
        if df.empty:
            print(f"{ticker} has no data, skipping.")
            continue

        # Filter data to only include rows from strategy_start_date onwards for decision making
        # But keep all data for indicator calculation
        df_for_indicators = df.copy()

        # Add account with properly calculated indicators
        portfolio.add_account(ticker, strategy_type, best_params, df_for_indicators, allocation_pct)
        print(f"Added {ticker}: strategy={strategy_type}, Allocation={allocation_pct}%")

    # Run the portfolio simulation - make sure it only processes dates after strategy_start_date
    if previous_state_loaded:
        # If continuing from previous state, start from last processed date
        portfolio.run(start_from_last_date=True)
    else:
        # If starting fresh, manually set a filter to only take action from strategy_start_date
        # Modify the portfolio.run method to respect this date
        portfolio.strategy_start_date = strategy_start_date
        portfolio.run(start_from_last_date=False)

    # Save current state for future runs
    portfolio.save_state(state_file)

    # Save logs to CSV files (append mode)
    combined_log = portfolio.combined_trade_log()
    if previous_state_loaded and os.path.exists(combined_log_path):
        # If we loaded previous state, try to append to existing log without duplicates
        try:
            existing_log = pd.read_csv(combined_log_path)
            existing_log['Date'] = pd.to_datetime(existing_log['Date'])

            # Get max date from existing log
            max_date = existing_log['Date'].max() if not existing_log.empty else pd.Timestamp.min

            # Filter new log entries to avoid duplicates
            combined_log['Date'] = pd.to_datetime(combined_log['Date'])
            new_entries = combined_log[combined_log['Date'] > max_date]

            # Append new entries to existing log
            updated_log = pd.concat([existing_log, new_entries]).reset_index(drop=True)
            updated_log.to_csv(combined_log_path, index=False)
            print(f"✅ Appended new entries to combined trade log at {combined_log_path}")
        except Exception as e:
            print(f"❌ Error appending to existing log: {e}")
            # Fallback to overwriting
            combined_log.to_csv(combined_log_path, index=False)
            print(f"✅ Combined trade log saved to {combined_log_path}")
    else:
        # Otherwise just save the log
        combined_log.to_csv(combined_log_path, index=False)
        print(f"✅ Combined trade log saved to {combined_log_path}")

    # Do the same for snapshot log
    snapshot_df = portfolio.snapshot_log_df()
    if previous_state_loaded and os.path.exists(snapshot_log_path):
        try:
            existing_snapshot = pd.read_csv(snapshot_log_path)
            existing_snapshot['Date'] = pd.to_datetime(existing_snapshot['Date'])

            # Get max date from existing log
            max_date = existing_snapshot['Date'].max() if not existing_snapshot.empty else pd.Timestamp.min

            # Filter new snapshot entries
            snapshot_df['Date'] = pd.to_datetime(snapshot_df['Date'])
            new_entries = snapshot_df[snapshot_df['Date'] > max_date]

            # Append new entries
            updated_snapshot = pd.concat([existing_snapshot, new_entries]).reset_index(drop=True)
            updated_snapshot.to_csv(snapshot_log_path, index=False)
            print(f"✅ Appended new entries to snapshot log at {snapshot_log_path}")
        except Exception as e:
            print(f"❌ Error appending to existing snapshot log: {e}")
            # Fallback to overwriting
            snapshot_df.to_csv(snapshot_log_path, index=False)
            print(f"✅ Snapshot log saved to {snapshot_log_path}")
    else:
        snapshot_df.to_csv(snapshot_log_path, index=False)
        print(f"✅ Snapshot log saved to {snapshot_log_path}")

    # Generate 96-hour report (4 days)
    report_message = generate_trade_report(portfolio, hours=96)

    # Email configuration
    from_addr = "ylzhao3377@gmail.com"
    to_addrs = ["zhaoyilin3377@gmail.com"]
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    smtp_user = "ylzhao3377@gmail.com"
    smtp_password = "pntr minq hlcb uikz"  # Recommend using app-specific password

    # Create trimmed versions of the log files (last 100 rows only)
    try:
        # For combined trade log
        trimmed_combined_log = combined_log.tail(100)  # Get last 100 rows
        trimmed_combined_path = "combined_trade_log_latest.csv"
        trimmed_combined_log.to_csv(trimmed_combined_path, index=False)

        # For snapshot log
        trimmed_snapshot_log = snapshot_df.tail(20)  # Get last 20 rows
        trimmed_snapshot_path = "snapshot_trade_log_latest.csv"
        trimmed_snapshot_log.to_csv(trimmed_snapshot_path, index=False)

        # Use the trimmed files for email attachments
        attachments = [trimmed_combined_path, trimmed_snapshot_path]
        print(f"✅ Created trimmed log files with the latest records for email attachments")
    except Exception as e:
        print(f"Error creating trimmed log files: {e}")
        # Fallback to full files if trimming fails
        attachments = [combined_log_path, snapshot_log_path]
        print("⚠️ Using full log files for attachments due to error when trimming")

    # Email subject
    pst_time_str = convert_to_pst(strategy_start_date).strftime("%Y-%m-%d %H:%M:%S %Z")
    subject = f"Trading Strategy Report - {pst_time_str}"

    # Send email with report and attachments
    EmailNotifier.send_email(subject, report_message, from_addr, to_addrs, smtp_server, smtp_port, smtp_user,
                             smtp_password, attachments)

    # Clean up the trimmed files after sending email
    try:
        if "trimmed_combined_path" in locals() and os.path.exists(trimmed_combined_path):
            os.remove(trimmed_combined_path)
        if "trimmed_snapshot_path" in locals() and os.path.exists(trimmed_snapshot_path):
            os.remove(trimmed_snapshot_path)
        print("✅ Cleaned up temporary trimmed files")
    except Exception as e:
        print(f"Note: Could not clean up temporary files: {e}")