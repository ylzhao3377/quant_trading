import pandas as pd
import numpy as np
import json
import os
import boto3
from itertools import product
from polygon import RESTClient
from datetime import datetime, timedelta
import argparse
import smtplib
from email.mime.text import MIMEText


os.environ["POLYGON_API_KEY"] = "0Fp6qkxgz6QugnvLPiR6d9cEMpK3hxFF"


def send_email(subject, message, from_addr, to_addrs, smtp_server, smtp_port, smtp_user, smtp_password):
    """
    发送电子邮件通知
    参数:
      - subject: 邮件主题
      - message: 邮件正文
      - from_addr: 发件人邮箱
      - to_addrs: 收件人邮箱列表
      - smtp_server: SMTP服务器地址
      - smtp_port: SMTP服务器端口（通常587）
      - smtp_user: SMTP用户名
      - smtp_password: SMTP密码
    """
    msg = MIMEText(message, 'plain', 'utf-8')
    msg['Subject'] = subject
    msg['From'] = from_addr
    msg['To'] = ", ".join(to_addrs)

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()  # 启用TLS安全传输
            server.login(smtp_user, smtp_password)
            server.sendmail(from_addr, to_addrs, msg.as_string())
        print("邮件发送成功！")
    except Exception as e:
        print("邮件发送失败：", e)



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
        if stock == "SQQQ":
            # 对冲品种单独处理
            portfolio[stock] = {
                'df': df,
                'in_position': False,
                'position': 0,
                'entry_price': 0,
                'max_profit_price': 0,
                'realized_profit': 0,
                'params': best_params,
                'hedge': True
            }
        else:
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
                'params': best_params,
                'hedge': False
            }

    def calc_total_equity(time_idx):
        total_val = balance
        for stock, state in portfolio.items():
            # 计算时包含所有品种（含对冲品种）
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

        # 卖出操作（跳过对冲品种 SQQQ）
        for stock, state in portfolio.items():
            if state.get('hedge', False):
                continue  # 对冲品种不参与常规逻辑
            df = state['df']
            row = df.iloc[i]
            prev_ma200 = df.iloc[i - 1]['MA200'] if i > 0 else row['MA200']
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
                        row['close'] >= state['last_partial_sell_price'] * (
                                1 + state['params']['partial_profit_threshold'])):
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

        # 买入操作（跳过对冲品种 SQQQ）
        eligible_stocks = []
        for stock, state in portfolio.items():
            if state.get('hedge', False):
                continue
            df = state['df']
            row = df.iloc[i]
            prev_ma200 = df.iloc[i - 1]['MA200'] if i > 0 else row['MA200']
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

        # ===== 新增 SQQQ 对冲逻辑 =====
        # 遍历所有非对冲品种，判断是否全部出现下跌趋势
        non_hedge_all_down = True
        non_hedge_any_up = False
        for stock, state in portfolio.items():
            if state.get("hedge", False):
                continue
            df = state['df']
            row = df.iloc[i]
            prev_ma200 = df.iloc[i - 1]['MA200'] if i > 0 else row['MA200']
            trend_up, trend_down = get_signals(row, prev_ma200)
            if trend_up:
                non_hedge_any_up = True
            if not trend_down:
                non_hedge_all_down = False

        hedge_state = portfolio["SQQQ"]
        sqqq_price = hedge_state['df'].iloc[i]['close']
        if non_hedge_all_down and not non_hedge_any_up:
            # 若所有非对冲品种均呈下跌趋势，且当前未持有 SQQQ，则用剩余资金50%买入对冲
            if not hedge_state['in_position']:
                allocated_amount = balance * 0.5
                shares = int(allocated_amount / sqqq_price)
                if shares > 0:
                    hedge_state['in_position'] = True
                    hedge_state['position'] = shares
                    hedge_state['entry_price'] = sqqq_price
                    hedge_state['max_profit_price'] = sqqq_price
                    balance -= shares * sqqq_price
                    total_equity = calc_total_equity(i)
                    overall_profit = total_equity - initial_balance
                    overall_trade_log.append({
                        'Date': hedge_state['df'].iloc[i]['datetime'],
                        'Stock': "SQQQ",
                        'Action': 'HEDGE_BUY',
                        'Price': sqqq_price,
                        'Shares': shares,
                        'Profit': 0,
                        'Total_Equity': total_equity,
                        'Stock_Profit': 0,
                        'All_Profit': overall_profit,
                        'Reason': 'All non-hedge stocks trending down; hedge activated with 50% cash'
                    })
                    actions_this_step.append("SQQQ: BUY (HEDGE)")
                    reasons_this_step.append("All non-hedge stocks trending down; hedge activated")
        elif non_hedge_any_up:
            # 若任一非对冲品种出现上升趋势，且持有 SQQQ，则全部卖出 SQQQ
            if hedge_state['in_position']:
                trade_profit = (sqqq_price - hedge_state['entry_price']) * hedge_state['position']
                hedge_state['realized_profit'] += trade_profit
                balance += hedge_state['position'] * sqqq_price
                total_equity = calc_total_equity(i)
                overall_profit = total_equity - initial_balance
                overall_trade_log.append({
                    'Date': hedge_state['df'].iloc[i]['datetime'],
                    'Stock': "SQQQ",
                    'Action': 'HEDGE_SELL',
                    'Price': sqqq_price,
                    'Shares': hedge_state['position'],
                    'Profit': trade_profit,
                    'Total_Equity': total_equity,
                    'Stock_Profit': trade_profit,
                    'All_Profit': overall_profit,
                    'Reason': 'At least one non-hedge stock trending up; hedge deactivated'
                })
                actions_this_step.append("SQQQ: SELL_ALL (HEDGE)")
                reasons_this_step.append("At least one non-hedge stock trending up; hedge deactivated")
                hedge_state['position'] = 0
                hedge_state['in_position'] = False
                hedge_state['entry_price'] = 0
                hedge_state['max_profit_price'] = 0

        # 判断是否为日末快照（最后一根或下一时点日期不同）
        record_snapshot = False
        if i == min_length - 1:
            record_snapshot = True
        else:
            next_dt = pd.to_datetime(portfolio[list(portfolio.keys())[0]]['df'].iloc[i + 1]['datetime'])
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
    逻辑与 backtest_portfolio() 保持一致（同时加入 SQQQ 对冲逻辑）。
    """
    suggestions = {}
    for ticker, df in stock_data.items():
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values(by='datetime').reset_index(drop=True)
        df = compute_indicators(df)
        df['MA200_prev'] = df['MA200'].shift(1)
        latest = df.iloc[-1]
        prev_ma200 = latest['MA200_prev'] if pd.notna(latest['MA200_prev']) else latest['MA200']
        current_price = latest['close']

        # 对 SQQQ 做特殊处理：根据所有非对冲品种的最新趋势决定买入或卖出
        if ticker == "SQQQ":
            non_hedge_all_down = True
            non_hedge_any_up = False
            for t, d in stock_data.items():
                if t == "SQQQ":
                    continue
                d['datetime'] = pd.to_datetime(d['datetime'])
                d = d.sort_values(by='datetime').reset_index(drop=True)
                d = compute_indicators(d)
                row = d.iloc[-1]
                prev = d.iloc[-2]['MA200'] if len(d) > 1 else row['MA200']
                trend_up, trend_down = get_signals(row, prev)
                if trend_up:
                    non_hedge_any_up = True
                if not trend_down:
                    non_hedge_all_down = False
            if non_hedge_all_down and not non_hedge_any_up:
                suggestion = "BUY"
                reason = "所有非对冲品种均呈下跌趋势；建议用余额50%买入对冲品 SQQQ"
            elif non_hedge_any_up:
                suggestion = "SELL_ALL"
                reason = "至少有一只非对冲品种呈上升趋势；建议卖出所有 SQQQ 对冲仓位"
            else:
                suggestion = "HOLD"
                reason = "无明显对冲信号"
            suggestions[ticker] = {
                "suggestion": suggestion,
                "reason": reason,
                "current_price": current_price,
                "trend_up": None,
                "trend_down": None,
                "entry_price": None,
                "shares": None,
                "partial_sell_count": None,
                "last_partial_sell_price": None,
                "max_profit_price": None
            }
            continue

        # 非对冲品种采用原有逻辑
        trend_up, trend_down = get_signals(latest, prev_ma200)
        # 从 current_positions 获取交易记录列表，并计算当前净持仓信息
        tx_list = current_positions.get(ticker, [])
        pos = get_current_position(tx_list)
        shares = pos.get("shares", 0)
        entry_price = pos.get("entry_price", current_price)
        partial_sell_count = pos.get("partial_sell_count", 0)
        last_partial_sell_price = pos.get("last_partial_sell_price", entry_price)
        entry_date = pos.get("entry_date", None)

        if shares > 0 and entry_date is not None:
            entry_date = pd.to_datetime(entry_date)
            df_since_entry = df[df['datetime'] >= entry_date]
            max_profit_price = df_since_entry['close'].max() if not df_since_entry.empty else current_price
        else:
            max_profit_price = current_price

        params_path = os.path.join("data", f"{ticker}_best_params.json")
        if os.path.exists(params_path):
            with open(params_path, "r") as f:
                best_params = json.load(f)
        else:
            best_params = {"partial_profit_threshold": 0.05, "trailing_stop_pct": 0.90}
        trailing_stop_pct = best_params.get("trailing_stop_pct", 0.90)
        partial_profit_threshold = best_params.get("partial_profit_threshold", 0.05)

        if shares > 0:
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
                reason = f"部分止盈条件满足，卖出1/4：当前价 {current_price} >= {last_partial_sell_price}*(1+{partial_profit_threshold})，部分卖出次数 {partial_sell_count} < 4"
            else:
                suggestion = "HOLD"
                reason = "无卖出信号"
        else:
            if not tx_list or not any(tx.get("action", "").lower() == "sell_all" for tx in tx_list):
                if trend_up:
                    suggestion = "BUY"
                    reason = "检测到上升趋势信号，且未曾卖出过，建议用余额1/2购买"
                else:
                    suggestion = "HOLD"
                    reason = "未触发买入条件"
            else:
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
    for stock in ["AMZN", "META", "NVDA", "TSLA", "TQQQ", "PLTR", "SNOW",
                  "GOOGL", "AAPL", "AVGO", "KO", "SPY", "VRT", "AMD", "COST",
                  "JNJ", "JPM", "ARKK", 'SQQQ']:
        df = fetch_stock_data(stock, days, "hour")
        stock_data[stock] = df
    trade_df, detailed_trade_df, summary = backtest_portfolio(stock_data, initial_balance=20000)
    print(f"Test performance {summary}")
    trade_df.to_csv(f"output/portfolio-{days}.csv")
    pos_path = os.path.join("position", "position.json")
    if os.path.exists(pos_path):
        with open(pos_path, "r") as f:
            current_positions = json.load(f)
    else:
        current_positions = {}
    suggestions = simulate_next_hour_suggestion(stock_data, current_positions)
    print("下一个小时操作建议：")
    subject = "股票策略通知"
    sms_message = "下一个小时建议：\n"
    for ticker, info in suggestions.items():
        line = f"{ticker}: {info['suggestion']} (当前价 {info['current_price']}) - {info['reason']}\n"
        print(line)
        sms_message += line
    from_addr = "ylzhao3377@gmail.com"
    to_addrs = ["ylzhao3377@gmail.com"]
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    smtp_user = "ylzhao3377@gmail.com"
    smtp_password = "pntr minq hlcb uikz"  # 推荐使用应用专用密码
    send_email(subject, sms_message, from_addr, to_addrs, smtp_server, smtp_port, smtp_user, smtp_password)
