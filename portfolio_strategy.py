import pandas as pd
import numpy as np
import ast
import os
from polygon import RESTClient
from datetime import datetime, timedelta


os.environ["POLYGON_API_KEY"] = "0Fp6qkxgz6QugnvLPiR6d9cEMpK3hxFF"


# === 屏蔽 Optuna 内部 trial 输出 ===
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def fetch_stock_data(ticker: str, start_date: str, end_date: str, timespan: str):
    """
    从 Polygon.io 获取指定股票的历史数据。
    """
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        raise ValueError("请设置 POLYGON_API_KEY 环境变量。")
    client = RESTClient(api_key)
    aggs = []
    for agg in client.list_aggs(
            ticker=ticker,
            multiplier=1,
            timespan=timespan,
            from_=start_date,
            to=end_date,
            limit=50000
    ):
        aggs.append(agg)
    df = pd.DataFrame(aggs)
    if df.empty:
        print(f"{ticker} 在指定日期范围内无数据。")
        return df
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df


def compute_indicators(df):
    # 均线与ATR计算
    df['EMA50'] = df['close'].ewm(span=50).mean()
    df['EMA200'] = df['close'].ewm(span=200).mean()
    df['H-L'] = df['high'] - df['low']
    df['H-PC'] = abs(df['high'] - df['close'].shift(1))
    df['L-PC'] = abs(df['low'] - df['close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(14).mean()

    # 布林带
    df['Middle'] = df['close'].rolling(50).mean()
    df['STD'] = df['close'].rolling(50).std()
    df['Upper'] = df['Middle'] + 2 * df['STD']
    df['Lower'] = df['Middle'] - 2 * df['STD']

    # ADX指标
    df['+DM'] = np.where(df['high'] > df['high'].shift(1), df['high'] - df['high'].shift(1), 0)
    df['-DM'] = np.where(df['low'] < df['low'].shift(1), df['low'].shift(1) - df['low'], 0)
    df['+DI'] = 100 * (df['+DM'].rolling(14).sum() / df['ATR'])
    df['-DI'] = 100 * (df['-DM'].rolling(14).sum() / df['ATR'])
    df['DX'] = (abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI']).replace(0, np.nan)) * 100
    df['ADX'] = df['DX'].rolling(14).mean()

    # 成交量均值（检测放量）
    df['Volume_MA'] = df['volume'].rolling(20).mean()
    return


# 修改后的 compute_indicators，支持传入布林带窗口参数
def compute_indicators(df, bollinger_window=50):
    # EMA 和 ATR 计算保持不变
    df['EMA50'] = df['close'].ewm(span=50).mean()
    df['EMA200'] = df['close'].ewm(span=200).mean()
    df['H-L'] = df['high'] - df['low']
    df['H-PC'] = abs(df['high'] - df['close'].shift(1))
    df['L-PC'] = abs(df['low'] - df['close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(14).mean()

    # 使用传入的 bollinger_window 计算布林带
    df['Middle'] = df['close'].rolling(window=bollinger_window).mean()
    df['STD'] = df['close'].rolling(window=bollinger_window).std()
    df['Upper'] = df['Middle'] + 2 * df['STD']
    df['Lower'] = df['Middle'] - 2 * df['STD']

    # ADX 指标
    df['+DM'] = np.where(df['high'] > df['high'].shift(1), df['high'] - df['high'].shift(1), 0)
    df['-DM'] = np.where(df['low'] < df['low'].shift(1), df['low'].shift(1) - df['low'], 0)
    df['+DI'] = 100 * (df['+DM'].rolling(14).sum() / df['ATR'])
    df['-DI'] = 100 * (df['-DM'].rolling(14).sum() / df['ATR'])
    df['DX'] = (abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI']).replace(0, np.nan)) * 100
    df['ADX'] = df['DX'].rolling(14).mean()

    # 成交量均值
    df['Volume_MA'] = df['volume'].rolling(20).mean()
    return df


def compute_trend_strength(row, params):
    price = row['close']
    adx = row['ADX']
    atr = row['ATR']
    atr_pct = atr / price if price > 0 else 0
    volume_spike = row['volume'] > row['Volume_MA'] * params.get("vol_threshold", 1.5)
    breakout_confirmed = (adx > params.get("adx_threshold", 30)) or volume_spike
    trend_up = (row['EMA50'] > row['EMA200']) and (adx > 20)
    if trend_up and (price > row['Upper']) and breakout_confirmed and (atr_pct > params.get("atr_pct_threshold", 0.005)):
        return 1.0
    return 0.0


def compute_range_strength(row, params):
    price = row['close']
    middle = row['Middle']
    std = row['STD']
    if std <= 0:
        return 0.0
    z = (price - middle) / std
    k, c = 3, 2
    strength = 1 / (1 + np.exp(-k * (abs(z) - c)))
    return strength


def get_available_funds(account, idle_multiplier=4):
    if account.strategy_type == "trend":
        return account.balance * idle_multiplier if account.position == 0 else account.balance
    elif account.strategy_type == "range":
        total_pos = account.base_position + account.float_position
        return account.balance * idle_multiplier if total_pos == 0 else account.balance


def run_trend_sell_logic(account, row, params):
    if account.in_position and account.position > 0:
        price = row['close']
        trailing_stop_pct = params.get("trailing_stop_pct", 0.95)
        account.max_profit_price = max(account.max_profit_price, price)
        trailing_stop_price = account.max_profit_price * trailing_stop_pct

        sell_shares = 0
        reason = ""
        # 止盈：跌破布林中轨
        if price < row['Middle']:
            sell_shares = account.position
            reason = "Price below Bollinger Middle (Take Profit)"
        # 止损：跌破EMA200或触发追踪止损
        elif price < row['EMA200'] or price <= trailing_stop_price:
            sell_shares = account.position
            reason = "Below EMA200 or Trailing Stop triggered"

        if sell_shares > 0:
            profit = (price - account.entry_price) * account.position
            account.realized_profit += profit
            account.balance += account.position * price
            log_entry = {
                "Date": row["datetime"],
                "Action": "SELL",
                "Reason": reason,
                "Price": price,
                "Shares": sell_shares,
                "Balance": account.balance,
                "Position": 0,
                "Realized_Profit": account.realized_profit
            }
            account.position = 0
            account.in_position = False
            account.entry_price = 0
            account.max_profit_price = 0
            return log_entry
    return None


def run_range_sell_logic(account, row, params):
    price = row['close']
    date = row['datetime']
    lower_band = row['Lower']
    upper_band = row['Upper']
    middle_band = row['Middle']
    std = row['STD']
    z_score = (price - middle_band) / std if std > 0 else 0
    k, c = 3, 2
    breakout_strength = 1 / (1 + np.exp(-k * (abs(z_score) - c)))
    min_rise_pct = params.get("min_rise_pct", 0.02)
    min_strength = params.get("min_strength", 0.3)
    init_bal = params.get("initial_balance", 10000)
    max_lot_value = init_bal / 5

    # 基础仓卖出
    if price > upper_band and breakout_strength >= min_strength and account.base_position > 0:
        if account.last_sell_price is None or price > account.last_sell_price * (1 + min_rise_pct):
            sell_value = min(max_lot_value * breakout_strength, account.base_position * price)
            shares = int(sell_value / price)
            if shares > 0:
                total_cost = sum(account.base_entry_prices[:shares])
                profit = shares * price - total_cost
                account.realized_profit += profit
                account.balance += shares * price
                account.base_position -= shares
                account.base_entry_prices = account.base_entry_prices[shares:]
                account.last_sell_price = price
                action = "SELL"
                reason = f"Base sell on break upper (Z={z_score:.2f}, Strength={breakout_strength:.2f})"
                if account.base_position == 0:
                    account.last_buy_price = None
                    account.last_sell_price = None
                return {"Date": date, "Action": action, "Reason": reason, "Price": price, "Shares": shares,
                        "Balance": account.balance, "Position": account.base_position, "Realized_Profit": account.realized_profit}
    # 浮动仓卖出
    if price > upper_band and breakout_strength >= min_strength and account.float_position > 0:
        float_sell_value = min(init_bal * params.get("float_position_pct", 0.1) * breakout_strength,
                               account.float_position * price)
        shares = int(float_sell_value / price)
        if shares > 0:
            float_total_cost = sum(account.float_entry_prices[:shares])
            profit = shares * price - float_total_cost
            account.realized_profit += profit
            account.balance += shares * price
            account.float_position -= shares
            account.float_entry_prices = account.float_entry_prices[shares:]
            action = "SELL"
            reason = f"Float sell on break upper (Z={z_score:.2f}, Strength={breakout_strength:.2f})"
            return {"Date": date, "Action": action, "Reason": reason, "Price": price, "Shares": shares,
                    "Balance": account.balance, "Position": account.float_position, "Realized_Profit": account.realized_profit}
    return None


class CapitalPool:
    def __init__(self, total_capital):
        self.total_capital = total_capital
        self.available_capital = total_capital

    def allocate(self, amount):
        if self.available_capital >= amount:
            self.available_capital -= amount
            return amount
        return 0

    def release(self, amount):
        self.available_capital += amount

    def status(self):
        return self.available_capital


class SubAccount:
    def __init__(self, name, strategy_type, allocation_pct, capital_pool, params):
        self.name = name
        self.strategy_type = strategy_type.lower()  # 'trend' 或 'range'
        self.allocation_pct = allocation_pct
        self.capital_pool = capital_pool
        self.params = params
        self.balance = capital_pool.total_capital * allocation_pct / 100
        self.trade_log = []
        # 用于记录上一次有效价格（用于 snapshot 缺失数据时）
        self.last_price = None
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


class SynchronizedPortfolio:
    def __init__(self, total_capital):
        self.capital_pool = CapitalPool(total_capital)
        self.accounts = {}  # {股票代码: {'account': SubAccount, 'data': DataFrame}}
        self.trade_log = []
        self.snapshot_log = []

    def add_account(self, name, strategy_type, params, df, allocation_pct):
        df = df.reset_index()
        df['datetime'] = pd.to_datetime(df['datetime'])
        # 使用传入参数中的 bollinger_window，若没有则默认 50
        window = params.get("bollinger_window", 50)
        df = compute_indicators(df, bollinger_window=window)
        df.sort_values('datetime', inplace=True)
        allocated = self.capital_pool.allocate(self.capital_pool.total_capital * allocation_pct / 100)
        account = SubAccount(name, strategy_type, allocation_pct, self.capital_pool, params)
        account.balance = allocated
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

    def run(self):
        # 构建统一时间线：合并所有账户 df 的 datetime 列
        all_times = sorted(set().union(*[set(data['data']['datetime']) for data in self.accounts.values()]))
        n_times = len(all_times)
        for i, current_time in enumerate(all_times):
            # 每个时间步前，计算全组合总余额（操作前的总balance）
            pre_total_balance = self.capital_pool.status() + sum(
                [data['account'].balance for data in self.accounts.values()])

            # === 卖出阶段 ===
            for name, data in self.accounts.items():
                df = data['data']
                account = data['account']
                current_rows = df[df['datetime'] == current_time]
                if not current_rows.empty:
                    row = current_rows.iloc[0]
                    # 记录操作前持仓（用于卖出占比计算）
                    if account.strategy_type == 'trend':
                        pre_position = account.position
                        log_entry = run_trend_sell_logic(account, row, account.params)
                    elif account.strategy_type == 'range':
                        pre_position = account.base_position + account.float_position
                        log_entry = run_range_sell_logic(account, row, account.params)
                    else:
                        log_entry = None
                    if log_entry and log_entry["Action"] != "HOLD":
                        # 添加操作前全组合总余额
                        log_entry["Pre_Total_Balance"] = pre_total_balance
                        # 如果是卖出，则记录卖出比例
                        # 注意：pre_position 可能为 0，此时设为 0
                        if pre_position > 0:
                            log_entry["Operation_Ratio"] = log_entry["Shares"] / pre_position
                        else:
                            log_entry["Operation_Ratio"] = 0
                        log_entry["Stock"] = name
                        self.trade_log.append(log_entry)
                        account.trade_log.append(log_entry)

            # === 买入阶段：先收集所有有效买入订单 ===
            valid_orders = []
            total_strength = 0.0
            for name, data in self.accounts.items():
                df = data['data']
                account = data['account']
                current_rows = df[df['datetime'] == current_time]
                if not current_rows.empty:
                    row = current_rows.iloc[0]
                    if account.strategy_type == 'trend' and not account.in_position:
                        strength = compute_trend_strength(row, account.params)
                        if strength > 0:
                            valid_orders.append((account, row, strength))
                            total_strength += strength
                    elif account.strategy_type == 'range':
                        if row['close'] < row['Lower']:
                            strength = compute_range_strength(row, account.params)
                            if strength >= account.params.get("min_strength", 0.3):
                                valid_orders.append((account, row, strength))
                                total_strength += strength
            # 使用同一时刻的 pre_total_balance（上面已计算）作为买入操作前的全组合余额
            for (account, row, strength) in valid_orders:
                weight = strength / total_strength if total_strength > 0 else 0
                allocated_funds = pre_total_balance * weight
                # 严格以当前账户余额为上限
                funds_to_use = min(allocated_funds, account.balance * 2)
                price = row['close']
                balance_before = account.balance  # 买入前账户余额
                shares_to_buy = int(funds_to_use / price)
                if shares_to_buy > 0:
                    money_spent = shares_to_buy * price
                    # 计算买入占比：买入金额占操作前全组合余额的百分比
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
                    # 更新 global_available（此处不再严格使用）
            # === 快照记录 ===
            current_day = pd.to_datetime(current_time).date()
            is_last_of_day = (i == n_times - 1) or (pd.to_datetime(all_times[i + 1]).date() != current_day)
            if valid_orders or is_last_of_day:
                snapshot = self.take_snapshot(current_time)
                self.snapshot_log.append(snapshot)

    def take_snapshot(self, current_time):
        snapshot = {"Date": current_time}
        total_position_value = 0
        total_realized_profit = 0
        for name, data in self.accounts.items():
            account = data['account']
            df = data['data']
            current_rows = df[df['datetime'] == current_time]
            if not current_rows.empty:
                current_price = current_rows.iloc[0]['close']
                # 更新账户的 last_price
                account.last_price = current_price
            else:
                # 若当前时刻缺失数据，使用上一次记录的价格
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
        if self.trade_log:
            combined_df = pd.DataFrame(self.trade_log)
            combined_df["Date"] = pd.to_datetime(combined_df["Date"], errors="coerce")
            combined_df = combined_df.sort_values("Date").reset_index(drop=True)
            return combined_df
        return pd.DataFrame()

    def snapshot_log_df(self):
        df = pd.DataFrame(self.snapshot_log)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.sort_values("Date").reset_index(drop=True)
        return df


if __name__ == "__main__":
    # 主流程：读取策略结果并构建组合
    results_df = pd.read_csv("strategy_results.csv")
    ticker_list = results_df["Stock"]
    filtered_results = results_df[
        (results_df['Stock'].isin(ticker_list)) &
        (results_df['Eligible'] == True) &
        (results_df['Allocation (%)'] > 0)
    ]

    print("过滤后的结果：")
    print(filtered_results[['Stock', 'Chosen Strategy', 'Best Params', 'Allocation (%)']])

    total_capital = 100000  # 根据实际资金设置
    portfolio = SynchronizedPortfolio(total_capital=total_capital)

    for idx, row in filtered_results.iterrows():
        ticker = row['Stock']
        strategy_type = row['Chosen Strategy'].lower().strip()  # 'trend' 或 'range'
        best_params = ast.literal_eval(row['Best Params'])
        allocation_pct = float(row['Allocation (%)'])
        now = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        df = fetch_stock_data(ticker, "2025-03-12", now, "hour")
        if df.empty:
            print(f"{ticker} 无数据，跳过。")
            continue
        portfolio.add_account(ticker, strategy_type, best_params, df, allocation_pct)
        print(f"已添加 {ticker}: 策略={strategy_type}, Allocation={allocation_pct}%")

    portfolio.run()

    combined_log = portfolio.combined_trade_log()
    combined_log.to_csv("combined_trade_log.csv", index=False)
    print("✅ 合并交易日志已保存至 combined_trade_log.csv")

    snapshot_df = portfolio.snapshot_log_df()
    snapshot_df.to_csv("snapshot_trade_log.csv", index=False)
    print("✅ 快照日志已保存至 snapshot_trade_log.csv")

    # 配置邮箱信息（请将 smtp_password 替换为你的应用专用密码）
    from_addr = "ylzhao3377@gmail.com"
    to_addrs = ["ylzhao3377@gmail.com"]
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    smtp_user = "ylzhao3377@gmail.com"
    smtp_password = "pntr minq hlcb uikz"  # 推荐使用应用专用密码

    # 设置邮件主题和正文
    subject = "策略回测结果"
    sms_message = "请查收附件中的 combined_trade_log.csv 和 snapshot_trade_log.csv 文件。"

    # 附件路径（请确保这两个文件已生成在当前目录或指定路径中）
    attachments = ["combined_trade_log.csv", "snapshot_trade_log.csv"]

    send_email(subject, sms_message, from_addr, to_addrs, smtp_server, smtp_port, smtp_user, smtp_password, attachments)
