import pandas as pd
import numpy as np
import json
import os
from itertools import product
from polygon import RESTClient
from datetime import datetime, timedelta
import argparse


os.environ["POLYGON_API_KEY"] = "0Fp6qkxgz6QugnvLPiR6d9cEMpK3hxFF"


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


def strategy(df, initial_balance=10000,
             base_position_pct=0.3,
             trailing_stop_pct=0.90,
             partial_profit_threshold=0.05):
    balance = initial_balance
    position = 0
    entry_price = 0
    trade_log = []
    last_partial_sell_price = None  # 记录上一次部分止盈的价格

    # 均线 & 波动指标
    df['EMA50'] = df['close'].ewm(span=50).mean()
    df['EMA100'] = df['close'].ewm(span=100).mean()
    df['MA200'] = df['close'].rolling(200).mean()
    df['High20'] = df['high'].rolling(20).max()

    # ATR 计算
    df['H-L'] = df['high'] - df['low']
    df['H-PC'] = abs(df['high'] - df['close'].shift(1))
    df['L-PC'] = abs(df['low'] - df['close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(14).mean()

    # 计算RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # 计算MACD
    df['EMA12'] = df['close'].ewm(span=12).mean()
    df['EMA26'] = df['close'].ewm(span=26).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal'] = df['MACD'].ewm(span=9).mean()

    # ADX 计算
    df['+DM'] = np.where(df['high'] > df['high'].shift(1), df['high'] - df['high'].shift(1), 0)
    df['-DM'] = np.where(df['low'] < df['low'].shift(1), df['low'].shift(1) - df['low'], 0)
    df['+DI'] = 100 * (df['+DM'].rolling(14).sum() / df['ATR'])
    df['-DI'] = 100 * (df['-DM'].rolling(14).sum() / df['ATR'])
    df['DX'] = (abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI']).replace(0, np.nan)) * 100
    df['ADX'] = df['DX'].rolling(14).mean()

    in_position = False
    max_profit_price = 0
    # 初始化状态变量
    entered_before = False  # 是否完成过第一次入场
    has_experienced_down = False  # 是否在卖出后曾经历过下降趋势
    partial_sell_count = 0  # 部分止盈操作次数，最大3次（第3次全仓卖出）
    last_partial_sell_price = None  # 上一次部分止盈的价格

    for i in range(200, len(df)):
        row = df.iloc[i]

        # 假设各项指标已经计算好，如 EMA50、EMA100、MA200、ATR 等
        # 计算趋势条件
        trend_up = (
                (row['EMA50'] > row['EMA100']) and
                (row['MA200'] > df['MA200'].shift(1).iloc[i]) and
                (row['MACD'] > row['Signal']) and
                (row['RSI'] > 50)  # RSI 在中性偏上
        )
        trend_down = (
                (row['EMA50'] < row['EMA100']) and
                (row['MA200'] < df['MA200'].shift(1).iloc[i]) and
                (row['MACD'] < row['Signal']) and
                (row['RSI'] < 50)  # RSI 在中性偏下
        )

        total_equity = balance + position * row['close']

        # 持仓时：检查卖出条件
        if in_position:
            # 更新持仓期间的最高价
            max_profit_price = max(max_profit_price, row['close'])

            # 1. 部分止盈逻辑
            # 如果当前价格比上一次部分止盈价格高出一定比例（partial_profit_threshold）
            if last_partial_sell_price is not None and row['close'] >= last_partial_sell_price * (
                    1 + partial_profit_threshold):
                if partial_sell_count < 4:
                    # 前四次部分止盈：卖出25%仓位
                    partial_shares = int(position * 0.25)
                    if partial_shares > 0:
                        position -= partial_shares
                        balance += partial_shares * row['close']
                        partial_sell_count += 1
                        trade_log.append({
                            'Date': row['datetime'],
                            'Total_Equity': balance + position * row['close'],
                            'Action': 'PARTIAL_SELL',
                            'Price': row['close'],
                            'Shares': partial_shares,
                            'Balance': balance,
                            'Profit': (row['close'] - entry_price) * partial_shares,
                            'Reason': f'Partial profit taking ({partial_sell_count}/3)'
                        })
                        # 更新部分止盈参考价
                        last_partial_sell_price = row['close']
                else:
                    # 第5次操作，全部卖出
                    profit = (row['close'] - entry_price) * position
                    balance += row['close'] * position
                    trade_log.append({
                        'Date': row['datetime'],
                        'Action': 'SELL_ALL',
                        'Price': row['close'],
                        'Shares': position,
                        'Balance': balance,
                        'Profit': profit,
                        'Reason': 'Fifth partial sell: full exit',
                        'Total_Equity': balance

                    })
                    position = 0
                    in_position = False
                    partial_sell_count = 0
                    last_partial_sell_price = None
                    # 退出后等待重新入场（has_experienced_down 将由下方逻辑更新）
                    has_experienced_down = False
                    continue

            # 2. Trailing Stop 逻辑
            trailing_stop_price = max_profit_price * trailing_stop_pct
            if row['close'] <= trailing_stop_price and row["close"] >= entry_price:
                profit = (row['close'] - entry_price) * position
                balance += row['close'] * position
                trade_log.append({
                    'Date': row['datetime'],
                    'Action': 'SELL_ALL',
                    'Price': row['close'],
                    'Shares': position,
                    'Balance': balance,
                    'Profit': profit,
                    'Reason': 'Trailing Stop triggered',
                    'Total_Equity': balance
                })
                position = 0
                in_position = False
                partial_sell_count = 0
                last_partial_sell_price = None
                continue

            # 3. Stop Loss 逻辑
            # stop_loss_price = entry_price - atr_multiplier * row['ATR']
            # if row['close'] <= stop_loss_price:
            #     profit = (row['close'] - entry_price) * position
            #     balance += row['close'] * position
            #     trade_log.append({
            #         'Date': row['datetime'],
            #         'Action': 'SELL_ALL',
            #         'Price': row['close'],
            #         'Shares': position,
            #         'Balance': balance,
            #         'Profit': profit,
            #         'Reason': 'Stop Loss triggered'
            #     })
            #     position = 0
            #     in_position = False
            #     partial_sell_count = 0
            #     last_partial_sell_price = None
            #     continue

        # 非持仓时，执行入场逻辑
        else:
            # 第一次入场：只需 trend up
            if not entered_before:
                if trend_up:
                    shares = int((total_equity) / row['close'])
                    if shares > 0:
                        position = shares
                        entry_price = row['close']
                        max_profit_price = entry_price
                        balance -= shares * row['close']
                        trade_log.append({
                            'Date': row['datetime'],
                            'Total_Equity': balance + position * row['close'],
                            'Action': 'BUY',
                            'Price': row['close'],
                            'Shares': shares,
                            'Balance': balance,
                            'Reason': 'First entry: Trend Up'
                        })
                        in_position = True
                        entered_before = True
                        partial_sell_count = 0
                        last_partial_sell_price = entry_price
            # 后续入场：要求卖出后经历过一次下降（trend down）后，遇到第一个 trend up 即可入场
            else:
                if has_experienced_down and trend_up:
                    shares = int((total_equity) / row['close'])
                    if shares > 0:
                        position = shares
                        entry_price = row['close']
                        max_profit_price = entry_price
                        balance -= shares * row['close']
                        trade_log.append({
                            'Date': row['datetime'],
                            'Total_Equity': balance + position * row['close'],
                            'Action': 'BUY',
                            'Price': row['close'],
                            'Shares': shares,
                            'Balance': balance,
                            'Reason': 'Re-entry: After downtrend, first Trend Up'
                        })
                        in_position = True
                        partial_sell_count = 0
                        last_partial_sell_price = entry_price
                        # 重入后重置下降标志
                        has_experienced_down = False

        # 更新下降标志：非持仓时如果出现下降趋势，则记录曾经历过下降
        if not in_position and trend_down:
            has_experienced_down = True


        # 加仓逻辑
        # current_position_pct = (position * row['close']) / total_equity if total_equity > 0 else 0
        # breakout_price = df['High20'].shift(1).iloc[i] * (1 + breakout_buffer)
        # if in_position and current_position_pct < max_position_pct and row['close'] > breakout_price and adx > 15:
        #     add_amount = total_equity * min(add_position_pct, max_position_pct - current_position_pct)
        #     add_shares = int(add_amount / row['close'])
        #     if add_shares > 0:
        #         position += add_shares
        #         balance -= add_shares * row['close']
        #         trade_log.append({
        #             'Date': row['datetime'],
        #             'Total_Equity': balance + position * row['close'],
        #             'Action': 'ADD',
        #             'Price': row['close'],
        #             'Shares': add_shares,
        #             'Balance': balance,
        #             'Reason': 'Breakout + ADX'
        #         })

    # 最后仅计算总资产（未平仓部分）
    final_price = df.iloc[-1]['close']
    final_equity = balance + position * final_price
    final_profit = final_equity - initial_balance

    # 增加总结信息
    trade_log.append({
        'Date': df.iloc[-1]['datetime'],
        'Action': 'HOLD_FINAL',
        'Price': final_price,
        'Shares': position,
        'Balance': balance,
        'Profit': final_profit,
        'Reason': 'Final Holding Value',
        'Total_Equity': final_equity
    })

    total_profit = final_equity - initial_balance
    roi = (total_profit / initial_balance * 100)
    hold_without_action = (df['close'].iloc[-1] - df['close'].iloc[0])/df['close'].iloc[0] * 100
    return pd.DataFrame(trade_log), {"Final Equity ($)": round(final_equity, 2),
                                     "Total Profit ($)": round(total_profit, 2),
                                     "ROI (%)": round(roi, 2),
                                     "Total Trades": len(trade_log),
                                     "No Action ROI (%)": round(hold_without_action, 2),}


def save_best_params(stock_name, params):
    with open(f'data/{stock_name}_best_params.json', 'w') as f:
        json.dump(params, f)


def load_best_params(stock_name):
    with open(f'data/{stock_name}_best_params.json', 'r') as f:
        return json.load(f)


# ===== 参数优化器 =====
def optimize_params(df, ticker, param_grid):
    best_roi = -float('inf')
    best_params = None
    best_summary = None
    keys, values = zip(*param_grid.items())
    for combo in product(*values):
        params = dict(zip(keys, combo))
        _, summary = strategy(df, **params)
        if summary['ROI (%)'] > best_roi:
            best_roi = summary['ROI (%)']
            best_params = params
            best_summary = summary
        print(f"Tested params: {params}, ROI: {summary['ROI (%)']}%")
        save_best_params(ticker, best_params)
    return best_params, best_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stock Strategy Optimizer')
    parser.add_argument('--ticker', type=str, required=True, help='Ticker symbol, e.g., TSLA')
    parser.add_argument('--start_date', type=str, required=True)
    parser.add_argument('--end_date', type=str, required=True)
    parser.add_argument('--option', type=str, required=True)
    args = parser.parse_args()
    ticker = args.ticker.upper()
    option = args.option

    # 读取数据
    df = fetch_stock_data(ticker, args.start_date, args.end_date, "hour")
    if option == "train":
        # 参数网格
        param_grid = {
            'trailing_stop_pct': [0.9, 0.92, 0.95],  # 浮动止盈比例 0.9, 0.92, 0.95
            'partial_profit_threshold': [0.01, 0.03, 0.05, 0.08]
        }

        print(f"Running optimization for {ticker}...")

        # 优化
        best_param, best_summary = optimize_params(df, ticker, param_grid)
        print(f"Best Parameters for {ticker}: {best_param}: {best_summary}")

        # 保存
        save_best_params(ticker, best_param)
        print(f"Best parameters saved to data/{ticker}_best_params.json")
    elif option == "test":
        best_param = load_best_params(ticker)
        _, summary = strategy(df, **best_param)
        print(f"Test performance {summary}")
        _.to_csv(f"output/{ticker}-{days}.csv")