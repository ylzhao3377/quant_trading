import pandas as pd
import numpy as np
import json
import os
from itertools import product
from polygon import RESTClient
from datetime import datetime, timedelta
import argparse


os.environ["POLYGON_API_KEY"] = "0Fp6qkxgz6QugnvLPiR6d9cEMpK3hxFF"


def fetch_stock_data(ticker: str, last_n_days: int, timespan: str):
    """
    从 Polygon.io 获取指定股票的历史数据。
    """
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        raise ValueError("请设置 POLYGON_API_KEY 环境变量。")
    client = RESTClient(api_key)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=last_n_days)
    aggs = []
    for agg in client.list_aggs(
            ticker=ticker,
            multiplier=1,
            timespan=timespan,
            from_=start_date.strftime('%Y-%m-%d'),
            to=end_date.strftime('%Y-%m-%d'),
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
    """
    计算常用指标：EMA50, EMA100, MA200, ATR, RSI, MACD, Signal。
    """
    df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['EMA100'] = df['close'].ewm(span=100, adjust=False).mean()
    df['MA200'] = df['close'].rolling(window=200).mean()
    # ATR计算
    df['H-L'] = df['high'] - df['low']
    df['H-PC'] = abs(df['high'] - df['close'].shift(1))
    df['L-PC'] = abs(df['low'] - df['close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(14).mean()
    # RSI计算
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    window_length = 14
    avg_gain = gain.rolling(window=window_length, min_periods=window_length).mean()
    avg_loss = loss.rolling(window=window_length, min_periods=window_length).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    # MACD计算
    df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    return df

def get_signals(row, prev_ma200):
    """
    根据当前行数据和上一时刻MA200判断信号：
      - trend_up: EMA50 > EMA100 且 MA200 > prev_ma200 且 MACD > Signal 且 RSI > 50
      - trend_down: 相反条件
    """
    trend_up = (row['EMA50'] > row['EMA100']) and (row['MA200'] > prev_ma200) and (row['MACD'] > row['Signal']) and (row['RSI'] > 50)
    trend_down = (row['EMA50'] < row['EMA100']) and (row['MA200'] < prev_ma200) and (row['MACD'] < row['Signal']) and (row['RSI'] < 50)
    return trend_up, trend_down

def backtest_portfolio(stock_data, initial_balance=10000):
    """
    对多只股票执行回测，返回快照日志（trade_df）、详细交易日志（detailed_trade_df）和 summary。
    参数：
      - stock_data: dict，键为股票代码，值为对应DataFrame（含 'datetime','open','high','low','close'）
      - initial_balance: 初始资金
    返回：
      - trade_df: 快照日志 DataFrame
      - detailed_trade_df: 详细交易日志 DataFrame（记录每笔实际交易，不含最终 HOLD_FINAL）
      - summary: dict，总结信息
    """
    n = len(stock_data)
    balance = initial_balance
    overall_trade_log = []
    snapshot_log = []
    min_length = min(len(df) for df in stock_data.values())

    # 初始化各股票状态，并加载各自最佳参数
    portfolio = {}
    for stock, df in stock_data.items():
        df = df.copy().reset_index(drop=True)
        df = compute_indicators(df)
        params_path = os.path.join("data", f"{stock}_best_params.json")
        if os.path.exists(params_path):
            with open(params_path, "r") as f:
                best_params = json.load(f)
        else:
            best_params = {"partial_profit_threshold": 0.05, "trailing_stop_pct": 0.90}
        portfolio[stock] = {
            'df': df,
            'in_position': False,
            'position': 0,
            'entry_price': 0,
            'max_profit_price': 0,
            'partial_sell_count': 0,
            'last_partial_sell_price': None,
            'entered_before': False,
            'has_experienced_down': False,
            'realized_profit': 0,
            'params': best_params
        }

    def calc_total_equity(time_idx):
        total_val = balance
        for stock, state in portfolio.items():
            df = state['df']
            if time_idx < len(df):
                total_val += state['position'] * df.iloc[time_idx]['close']
        return total_val

    def calc_stock_profit(state, current_price):
        if state['in_position']:
            return state['realized_profit'] + (current_price - state['entry_price']) * state['position']
        else:
            return state['realized_profit']

    # 回测循环：从第200根K线开始
    for i in range(200, min_length):
        current_dt = pd.to_datetime(portfolio[list(portfolio.keys())[0]]['df'].iloc[i]['datetime'])
        current_day = current_dt.date()
        actions_this_step = []
        reasons_this_step = []
        # 卖出操作
        for stock, state in portfolio.items():
            df = state['df']
            row = df.iloc[i]
            prev_ma200 = df.iloc[i-1]['MA200'] if i > 0 else row['MA200']
            trend_up, trend_down = get_signals(row, prev_ma200)
            if state['in_position']:
                state['max_profit_price'] = max(state['max_profit_price'], row['close'])
                # Stop Loss
                if row['close'] <= state['entry_price'] * 0.5:
                    trade_profit = (row['close'] - state['entry_price']) * state['position']
                    state['realized_profit'] += trade_profit
                    balance += state['position'] * row['close']
                    total_equity = calc_total_equity(i)
                    overall_profit = total_equity - initial_balance
                    stock_profit = calc_stock_profit(state, row['close'])
                    overall_trade_log.append({
                        'Date': row['datetime'],
                        'Stock': stock,
                        'Action': 'STOP_LOSS',
                        'Price': row['close'],
                        'Shares': state['position'],
                        'Profit': trade_profit,
                        'Total_Equity': total_equity,
                        'Stock_Profit': stock_profit,
                        'All_Profit': overall_profit,
                        'Reason': 'Stop Loss triggered (price fell 50% from entry)'
                    })
                    actions_this_step.append(f"{stock}: STOP_LOSS")
                    reasons_this_step.append("Price fell below 50% of entry")
                    state['position'] = 0
                    state['in_position'] = False
                    state['partial_sell_count'] = 0
                    state['last_partial_sell_price'] = None
                    state['has_experienced_down'] = False
                    continue

                # 部分止盈
                if (state['last_partial_sell_price'] is not None and
                    row['close'] >= state['last_partial_sell_price'] * (1 + state['params']['partial_profit_threshold'])):
                    if state['partial_sell_count'] < 4:
                        partial_shares = int(state['position'] * 0.25)
                        if partial_shares > 0:
                            state['position'] -= partial_shares
                            trade_profit = (row['close'] - state['entry_price']) * partial_shares
                            state['realized_profit'] += trade_profit
                            balance += partial_shares * row['close']
                            state['partial_sell_count'] += 1
                            total_equity = calc_total_equity(i)
                            overall_profit = total_equity - initial_balance
                            stock_profit = calc_stock_profit(state, row['close'])
                            overall_trade_log.append({
                                'Date': row['datetime'],
                                'Stock': stock,
                                'Action': 'PARTIAL_SELL',
                                'Price': row['close'],
                                'Shares': partial_shares,
                                'Profit': trade_profit,
                                'Total_Equity': total_equity,
                                'Stock_Profit': stock_profit,
                                'All_Profit': overall_profit,
                                'Reason': f"Partial profit taking ({state['partial_sell_count']}/5)"
                            })
                            actions_this_step.append(f"{stock}: PARTIAL_SELL")
                            reasons_this_step.append(f"Partial profit taking ({state['partial_sell_count']}/5)")
                            state['last_partial_sell_price'] = row['close']
                    else:
                        trade_profit = (row['close'] - state['entry_price']) * state['position']
                        state['realized_profit'] += trade_profit
                        balance += state['position'] * row['close']
                        total_equity = calc_total_equity(i)
                        overall_profit = total_equity - initial_balance
                        stock_profit = calc_stock_profit(state, row['close'])
                        overall_trade_log.append({
                            'Date': row['datetime'],
                            'Stock': stock,
                            'Action': 'SELL_ALL',
                            'Price': row['close'],
                            'Shares': state['position'],
                            'Profit': trade_profit,
                            'Total_Equity': total_equity,
                            'Stock_Profit': stock_profit,
                            'All_Profit': overall_profit,
                            'Reason': "Fifth partial sell: full exit"
                        })
                        actions_this_step.append(f"{stock}: SELL_ALL")
                        reasons_this_step.append("Fifth partial sell: full exit")
                        state['position'] = 0
                        state['in_position'] = False
                        state['partial_sell_count'] = 0
                        state['last_partial_sell_price'] = None
                        state['has_experienced_down'] = False

                # Trailing Stop
                trailing_stop_price = state['max_profit_price'] * state['params']['trailing_stop_pct']
                if row['close'] <= trailing_stop_price and row['close'] >= state['entry_price']:
                    trade_profit = (row['close'] - state['entry_price']) * state['position']
                    state['realized_profit'] += trade_profit
                    balance += state['position'] * row['close']
                    total_equity = calc_total_equity(i)
                    overall_profit = total_equity - initial_balance
                    stock_profit = calc_stock_profit(state, row['close'])
                    overall_trade_log.append({
                        'Date': row['datetime'],
                        'Stock': stock,
                        'Action': 'SELL_ALL',
                        'Price': row['close'],
                        'Shares': state['position'],
                        'Profit': trade_profit,
                        'Total_Equity': total_equity,
                        'Stock_Profit': stock_profit,
                        'All_Profit': overall_profit,
                        'Reason': "Trailing Stop triggered"
                    })
                    actions_this_step.append(f"{stock}: SELL_ALL")
                    reasons_this_step.append("Trailing Stop triggered")
                    state['position'] = 0
                    state['in_position'] = False
                    state['partial_sell_count'] = 0
                    state['last_partial_sell_price'] = None
                    state['has_experienced_down'] = False

                if not state['in_position'] and trend_down:
                    state['has_experienced_down'] = True

        # 买入操作：对于未持仓股票
        eligible_stocks = []
        for stock, state in portfolio.items():
            df = state['df']
            row = df.iloc[i]
            prev_ma200 = df.iloc[i-1]['MA200'] if i > 0 else row['MA200']
            trend_up, _ = get_signals(row, prev_ma200)
            if not state['in_position']:
                if not state['entered_before']:
                    if trend_up:
                        eligible_stocks.append(stock)
                else:
                    if state['has_experienced_down'] and trend_up:
                        eligible_stocks.append(stock)
                    elif ((row['EMA50'] < row['EMA100']) and
                          (row['MA200'] < prev_ma200) and
                          (row['MACD'] < row['Signal']) and
                          (row['RSI'] < 50)):
                        state['has_experienced_down'] = True

        if eligible_stocks:
            allocated_amount = min(balance / len(eligible_stocks), balance / 2)
            for stock in eligible_stocks:
                state = portfolio[stock]
                df = state['df']
                row = df.iloc[i]
                shares = int(allocated_amount / row['close'])
                if shares > 0:
                    state['position'] = shares
                    state['entry_price'] = row['close']
                    state['max_profit_price'] = row['close']
                    balance -= shares * row['close']
                    state['in_position'] = True
                    state['entered_before'] = True
                    state['partial_sell_count'] = 0
                    state['last_partial_sell_price'] = row['close']
                    state['has_experienced_down'] = False
                    total_equity = calc_total_equity(i)
                    overall_profit = total_equity - initial_balance
                    stock_profit = calc_stock_profit(state, row['close'])
                    overall_trade_log.append({
                        'Date': row['datetime'],
                        'Stock': stock,
                        'Action': 'BUY',
                        'Price': row['close'],
                        'Shares': shares,
                        'Profit': 0,
                        'Total_Equity': total_equity,
                        'Stock_Profit': stock_profit,
                        'All_Profit': overall_profit,
                        'Reason': "Buy signal triggered"
                    })
                    actions_this_step.append(f"{stock}: BUY")
                    reasons_this_step.append("Buy signal triggered")

        # 判断是否为日末快照（最后一根或下一时点日期不同）
        record_snapshot = False
        if i == min_length - 1:
            record_snapshot = True
        else:
            next_dt = pd.to_datetime(portfolio[list(portfolio.keys())[0]]['df'].iloc[i+1]['datetime'])
            record_snapshot = (next_dt.date() != current_day)

        if actions_this_step:
            total_position_value = sum(state['position'] * state['df'].iloc[i]['close'] for state in portfolio.values())
            total_equity = balance + total_position_value
            snapshot = {
                'Time': current_dt,
                'Action': ", ".join(actions_this_step),
                'Reason': ", ".join(reasons_this_step),
                'Cash Balance': balance
            }
            for stock, state in portfolio.items():
                snapshot[f'{stock} Shares'] = state['position']
            snapshot['Total Equity'] = total_equity
            snapshot_log.append(snapshot)
        elif record_snapshot:
            total_position_value = sum(state['position'] * state['df'].iloc[i]['close'] for state in portfolio.values())
            total_equity = balance + total_position_value
            snapshot = {
                'Time': current_dt,
                'Action': "NO ACTION",
                'Reason': "",
                'Cash Balance': balance
            }
            for stock, state in portfolio.items():
                snapshot[f'{stock} Shares'] = state['position']
            snapshot['Total Equity'] = total_equity
            snapshot_log.append(snapshot)

    # 回测结束后，计算最终结果
    overall_position_value = sum(state['position'] * state['df'].iloc[-1]['close'] for state in portfolio.values())
    final_equity = balance + overall_position_value
    total_profit = final_equity - initial_balance
    roi = (total_profit / initial_balance) * 100

    for stock, state in portfolio.items():
        final_price = state['df'].iloc[-1]['close']
        stock_profit = calc_stock_profit(state, final_price)
        overall_trade_log.append({
            'Date': state['df'].iloc[-1]['datetime'],
            'Stock': stock,
            'Action': 'HOLD_FINAL',
            'Price': final_price,
            'Shares': state['position'],
            'Profit': stock_profit,
            'Total_Equity': final_equity,
            'Stock_Profit': stock_profit,
            'All_Profit': total_profit,
            'Reason': 'Final Holding Value'
        })
        snapshot = {
            'Time': state['df'].iloc[-1]['datetime'],
            **{f'{s} Shares': portfolio[s]['position'] for s in portfolio},
            'Cash Balance': balance,
            'Total Equity': final_equity,
            'Action': "HOLD_FINAL",
            'Reason': "Final snapshot"
        }
        snapshot_log.append(snapshot)

    trade_df = pd.DataFrame(snapshot_log)
    trade_df['Time'] = pd.to_datetime(trade_df['Time'])
    trade_df = trade_df.sort_values(by='Time').reset_index(drop=True)

    trade_count = len([entry for entry in overall_trade_log if entry['Action'] != 'HOLD_FINAL'])
    summary = {
        "Final Equity ($)": round(final_equity, 2),
        "Total Profit ($)": round(total_profit, 2),
        "ROI (%)": round(roi, 2),
        "Total Trades": trade_count
    }

    detailed_trade_df = pd.DataFrame(overall_trade_log)
    detailed_trade_df['Date'] = pd.to_datetime(detailed_trade_df['Date'])
    detailed_trade_df = detailed_trade_df.sort_values(by='Date').reset_index(drop=True)

    return trade_df, detailed_trade_df, summary


def get_current_position(transactions):
    """
    根据交易记录列表计算当前净持仓。
    transactions: list of dict，每个 dict 包含 "date", "price", "shares", "action"（"buy" 或 "sell"）
    采用平均成本法：
      - 累加buy的成本和股数，sell则按平均成本减少。
    返回 dict，如：
      { "shares": net_shares, "entry_price": effective_entry_price,
        "partial_sell_count": sell次数, "last_partial_sell_price": 最后卖出价,
        "entry_date": 最早的buy日期 }
    """
    total_shares = 0
    total_cost = 0.0
    partial_sell_count = 0
    last_partial_sell_price = None
    entry_dates = []
    for tx in transactions:
        action = tx.get("action", "").lower()
        shares = tx.get("shares", 0)
        price = tx.get("price", 0)
        date = tx.get("date")
        if action == "buy":
            total_cost += price * shares
            total_shares += shares
            entry_dates.append(date)
        elif action == "sell" or action == "sell_all":
            if total_shares > 0:
                avg_cost = total_cost / total_shares
            else:
                avg_cost = 0
            total_cost -= avg_cost * shares
            total_shares -= shares
            partial_sell_count += 1
            last_partial_sell_price = price
    if total_shares <= 0:
        return {"shares": 0, "entry_price": 0, "partial_sell_count": partial_sell_count,
                "last_partial_sell_price": None, "entry_date": None}
    else:
        effective_entry_price = total_cost / total_shares
        entry_date = min(entry_dates) if entry_dates else None
        return {"shares": total_shares, "entry_price": effective_entry_price, "partial_sell_count": partial_sell_count,
                "last_partial_sell_price": last_partial_sell_price, "entry_date": entry_date}


def has_sold_all(transactions):
    """
    检查交易记录列表中是否存在 sell_all 记录
    """
    for tx in transactions:
        if tx.get("action", "").lower() == "sell_all":
            return True
    return False


def check_trend_down_since_sell_all(df, sell_all_date):
    """
    从 df 中筛选出 sell_all_date 之后的数据，判断是否有至少一次下降趋势信号
    """
    df_after = df[df['datetime'] >= pd.to_datetime(sell_all_date)]
    if df_after.empty:
        return False
    # 检查df_after中是否有行满足 trend_down
    for i in range(1, len(df_after)):
        row = df_after.iloc[i]
        prev_ma200 = df_after.iloc[i - 1]['MA200'] if i > 0 else row['MA200']
        _, trend_down = get_signals(row, prev_ma200)
        if trend_down:
            return True
    return False


def simulate_next_hour_suggestion(stock_data, current_positions):
    """
    根据过去365天的小时数据和当前持仓交易记录，模拟下一个小时的操作建议，
    逻辑与 backtest_portfolio() 保持一致：

      - 如果持仓（shares > 0）：
            * 如果当前价格 <= 入场价 * 0.5，则建议 "SELL_ALL (STOP LOSS)"
            * 如果 partial_sell_count == 4，则建议 "SELL_ALL"
            * 否则，利用自入场以来的最高价格 (max_profit_price)：
                  如果当前价格 <= max_profit_price * trailing_stop_pct 且当前价 >= 入场价，则建议 "SELL_ALL"（Trailing Stop）
            * 否则，如果存在 last_partial_sell_price 且当前价格 >= last_partial_sell_price*(1+partial_profit_threshold) 且 partial_sell_count < 4，
                  则建议 "PARTIAL_SELL"
            * 否则建议 "HOLD"
      - 如果无持仓：
            * 如果当前没有卖出记录（即从未完全卖出过），则只要检测到上升趋势（trend_up）就建议 "BUY"
            * 如果存在 sell_all 记录，则必须检查从最后一次卖出以来，在历史数据中是否经历过下降趋势；
                - 如果经历过且当前检测到上升趋势，则建议 "BUY"
                - 否则建议 "HOLD"
    参数：
      - stock_data: dict，键为股票代码，值为对应的 DataFrame（过去365天数据，包含 'datetime','open','high','low','close'）
      - current_positions: dict，键为股票代码，值为交易记录列表（格式见示例）
    返回：
      - suggestions: dict，每只股票给出建议及详细理由
    """
    suggestions = {}
    for ticker, df in stock_data.items():
        # 确保时间排序
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values(by='datetime').reset_index(drop=True)
        df = compute_indicators(df)
        df['MA200_prev'] = df['MA200'].shift(1)
        latest = df.iloc[-1]
        prev_ma200 = latest['MA200_prev'] if pd.notna(latest['MA200_prev']) else latest['MA200']
        trend_up, trend_down = get_signals(latest, prev_ma200)
        current_price = latest['close']

        # 从 current_positions 获取交易记录列表，并计算当前净持仓信息
        tx_list = current_positions.get(ticker, [])
        pos = get_current_position(tx_list)
        shares = pos.get("shares", 0)
        entry_price = pos.get("entry_price", current_price)
        partial_sell_count = pos.get("partial_sell_count", 0)
        last_partial_sell_price = pos.get("last_partial_sell_price", entry_price)
        entry_date = pos.get("entry_date", None)

        # 如果持仓且有入场日期，则从过去数据中计算持仓以来的最高价
        if shares > 0 and entry_date is not None:
            entry_date = pd.to_datetime(entry_date)
            df_since_entry = df[df['datetime'] >= entry_date]
            max_profit_price = df_since_entry['close'].max() if not df_since_entry.empty else current_price
        else:
            max_profit_price = current_price

        # 尝试加载该股票最佳参数
        params_path = os.path.join("data", f"{ticker}_best_params.json")
        if os.path.exists(params_path):
            with open(params_path, "r") as f:
                best_params = json.load(f)
        else:
            best_params = {"partial_profit_threshold": 0.05, "trailing_stop_pct": 0.90}
        trailing_stop_pct = best_params.get("trailing_stop_pct", 0.90)
        partial_profit_threshold = best_params.get("partial_profit_threshold", 0.05)

        # 决策建议
        if shares > 0:
            # 持仓时判断卖出条件
            if current_price <= entry_price * 0.5:
                suggestion = "SELL_ALL (STOP LOSS)"
                reason = f"当前价 {current_price} <= 50% 的入场价 {entry_price}"
            elif partial_sell_count == 4:
                suggestion = "SELL_ALL"
                reason = "已执行4次部分止盈，建议全仓卖出"
            elif current_price <= max_profit_price * trailing_stop_pct and current_price >= entry_price:
                suggestion = "SELL_ALL"
                reason = f"Trailing Stop触发，建议全仓卖出：当前价 {current_price} <= {max_profit_price} * {trailing_stop_pct}"
            elif (last_partial_sell_price is not None and
                  current_price >= last_partial_sell_price * (1 + partial_profit_threshold) and
                  partial_sell_count < 4):
                suggestion = "PARTIAL_SELL"
                reason = f"部分止盈条件满足，卖出1/4：当前价 {current_price} >= {last_partial_sell_price}*(1+{partial_profit_threshold}), 部分卖出次数 {partial_sell_count} < 4"
            else:
                suggestion = "HOLD"
                reason = "无卖出信号"
        else:
            # 无持仓时判断买入条件
            # 首先判断是否存在 sell_all 记录
            if not tx_list or not any(tx.get("action", "").lower() == "sell_all" for tx in tx_list):
                # 从未卖出过，则只要上升趋势即 BUY
                if trend_up:
                    suggestion = "BUY"
                    reason = "检测到上升趋势信号，且未曾卖出过，建议用余额1/2购买"
                else:
                    suggestion = "HOLD"
                    reason = "未触发买入条件"
            else:
                # 存在sell_all记录，需确认从最后一次sell_all以来经历过下降趋势
                sell_all_dates = [tx["date"] for tx in tx_list if tx.get("action", "").lower() == "sell_all"]
                last_sell_date = max(sell_all_dates) if sell_all_dates else None
                if last_sell_date is not None and check_trend_down_since_sell_all(df, last_sell_date):
                    if trend_up:
                        suggestion = "BUY"
                        reason = "检测到上升趋势信号，并且卖出后已经历过下降趋势，建议用余额1/2购买"
                    else:
                        suggestion = "HOLD"
                        reason = "未触发买入条件（上升趋势未出现）"
                else:
                    suggestion = "HOLD"
                    reason = "卖出后尚未出现下降趋势，等待调整"

        suggestions[ticker] = {
            "suggestion": suggestion,
            "reason": reason,
            "current_price": current_price,
            "trend_up": trend_up,
            "trend_down": trend_down,
            "entry_price": entry_price,
            "shares": shares,
            "partial_sell_count": partial_sell_count,
            "last_partial_sell_price": last_partial_sell_price,
            "max_profit_price": max_profit_price
        }
    return suggestions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stock Strategy Optimizer')
    parser.add_argument('--days', type=int, required=True)
    args = parser.parse_args()
    days = args.days

    stock_data = {}
    for stock in ["AMZN", "META", "NVDA", "TSLA", "TQQQ", "PLTR", "SNOW", "GOOGL", "CRWD", "AAPL", "AVGO", "KO", "SPY", "VRT", "AMD"]:
        df = fetch_stock_data(stock, days, "hour")
        stock_data[stock] = df
    # trade_df, detailed_trade_df, summary = backtest_portfolio(stock_data, initial_balance=20000)
    # print(f"Test performance {summary}")
    # trade_df.to_csv(f"output/portfolio-{days}.csv")
    pos_path = os.path.join("position", "position.json")
    if os.path.exists(pos_path):
        with open(pos_path, "r") as f:
            current_positions = json.load(f)
    else:
        current_positions = {}
    suggestions = simulate_next_hour_suggestion(stock_data, current_positions)
    print("下一个小时操作建议：")
    for ticker, info in suggestions.items():
        print(f"{ticker}: {info['suggestion']} (当前价 {info['current_price']}) - {info['reason']}")