import json
import time
import os
import asyncio
import aiofiles
import numpy as np
from datetime import datetime
from collections import defaultdict, deque
from autoTrader import *
from richardML import RichardML
import asyncio

class TokenAnalyzer:
    def __init__(self):
        # Configuration
        self.autoFarm = True
        self.debugPrint = False
        self.TAKE_PROFIT_PERCENT = 15.0
        self.STOP_LOSS_PERCENT = -15.0
        self.TRAILING_STOP_PERCENT = 3.0
        self.TRAILING_ACTIVATION_PERCENT = 5.0
        self.INITIAL_PRICE = 0.0000043
        self.TRADE_COOLDOWN = 30
        self.MIN_PRICE_CHANGE = 0.05
        self.initialPercent = 0.0
        self.buyDuringGrace = True
        # Add an instance of RichardML
        self.richard_ml = RichardML(snapshot_interval=5, monitoring_period_seconds=60)
        self.trade_outcomes = [] # To store results for the ML model
        
        # Trading fees
        self.JITO_TIP = 0.00007
        self.TIP_FEE_PERCENT = 0.01
        self.BCURVE_FEE_PERCENT = 0.01
        
        # Auto conditions
        self.aBuyBase1 = True
        self.aBuyAverage = True
        self.aBuyBuyers = True
        self.aBuyVolBar = True
        self.autoBuyRequireAll = False
        self.aSellTakeProfit = True
        self.aSellStopLoss = True
        self.aSellTrailing = True
        self.aSellPrice = True
        self.aSellVolume = True
        self.aSellEMA = False
        self.aSellFlagged = True
        self.aSellVolBar = True
        
        # Cache and state
        self.cache = self.ThreadSafeCache()
        self.trade_logger = self.BatchWriter('trade_history.log')
        self.status_updates = {}
        self.running = True
        self.last_display = 0
        self.display_interval = 1
        self.token_workers = {}
        self.results = {}
        self.results_lock = asyncio.Lock()
        self.clear_screen = True
    
    class ThreadSafeCache:
        def __init__(self):
            self.ema_data = {}
            self.trade_history = {}
            self.trailing_highs = {}
            self.positions_cache = {}
            self.volume_snapshots = {}
            self.last_positions_update = 0
            self.lock = asyncio.Lock()
        
        async def get_positions(self):
            current_time = time.time()
            async with self.lock:
                if current_time - self.last_positions_update > 0.5:
                    try:
                        async with aiofiles.open('positions.json', 'r') as f:
                            data = json.loads(await f.read())
                        self.positions_cache = {k: v for k, v in data.items() if k != 'paper_balance'}
                        self.last_positions_update = current_time
                    except:
                        self.positions_cache = {}
            return self.positions_cache.copy()
    
    class BatchWriter:
        def __init__(self, file_path, flush_interval=5.0):
            self.file_path = file_path
            self.buffer = []
            self.flush_interval = flush_interval
            self.last_flush = time.time()
            self.lock = asyncio.Lock()
        
        async def write(self, data):
            async with self.lock:
                self.buffer.append(data)
                if time.time() - self.last_flush > self.flush_interval:
                    await self._flush()
        
        async def _flush(self):
            if self.buffer:
                try:
                    async with aiofiles.open(self.file_path, 'a') as f:
                        await f.writelines(self.buffer)
                    self.buffer.clear()
                    self.last_flush = time.time()
                except Exception as e:
                    if self.debugPrint:
                        print(f"Error flushing to {self.file_path}: {e}")
    
    def calculate_trading_fees(self, sol_amount, is_opening=True):
        tip_fee = sol_amount * self.TIP_FEE_PERCENT
        total_fees = self.JITO_TIP + tip_fee
        if is_opening:
            total_fees += sol_amount * self.BCURVE_FEE_PERCENT
        return total_fees
    
    def calculate_pnl(self, position, current_price):
        if not position or not position.get('entry_price') or not position.get('sol_amount'):
            return 0, 0
        
        entry_price = position['entry_price']
        sol_amount = position['sol_amount']
        token_amount = position.get('token_amount', 0)
        
        current_value = token_amount * current_price
        opening_fees = self.calculate_trading_fees(sol_amount, True)
        closing_fees = self.calculate_trading_fees(current_value, False)
        
        pnl_sol = current_value - sol_amount - opening_fees - closing_fees
        pnl_percent = (pnl_sol / sol_amount * 100) if sol_amount > 0 else 0
        
        return pnl_sol, pnl_percent + self.initialPercent
    
    async def log_trade_reason(self, action, mint, price, reason, position=None):
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            if action == "BUY":
                log_entry = f"[{timestamp}] BUY: {mint[:20]}... @ ${price:.10f} | Reason: {reason}\n"
            else:
                if position:
                    pnl_sol, pnl_percent = self.calculate_pnl(position, price)
                    entry_price = position.get('entry_price', price)
                    log_entry = f"[{timestamp}] SELL: {mint[:20]}... Entry: ${entry_price:.10f} Exit: ${price:.10f} | PnL: {pnl_sol:.6f} SOL ({pnl_percent:.2f}%) | Reason: {reason}\n"
                else:
                    log_entry = f"[{timestamp}] SELL: {mint[:20]}... @ ${price:.10f} | Reason: {reason}\n"
            
            if action == "SELL" and position:
                pnl_sol, pnl_percent = self.calculate_pnl(position, price)

                # Get the features that led to the initial buy
                trade_entry_features = self.richard_ml.memory['tokens'].get(mint, {}).get('last_features')

                if trade_entry_features:
                    self.trade_outcomes.append({
                        "features": trade_entry_features,
                        "pnl_percent": pnl_percent,
                        # You can add more data here, like how long the trade was open
                    })



            await self.trade_logger.write(log_entry)
        except Exception as e:
            if self.debugPrint:
                print(f"Error logging trade: {e}")
    
    async def update_contract_status(self, mint, status, flag_reason=None):
        self.status_updates[mint] = {'status': status, 'flag_reason': flag_reason}
    
    async def flush_status_updates(self):
        if not self.status_updates:
            return
        
        try:
            async with aiofiles.open('contract_addresses.json', 'r') as f:
                contract_data = json.loads(await f.read())
            
            for token in contract_data:
                mint = token.get('mint_address')
                if mint in self.status_updates:
                    token['status'] = self.status_updates[mint]['status']
                    if self.status_updates[mint]['flag_reason']:
                        token['flag_reason'] = self.status_updates[mint]['flag_reason']
            
            async with aiofiles.open('contract_addresses.json', 'w') as f:
                await f.write(json.dumps(contract_data, indent=2))
            
            self.status_updates.clear()
        except Exception as e:
            if self.debugPrint:
                print(f"Error updating contract status: {e}")
    
    def update_volume_snapshot(self, mint, volume):
        current_time = time.time()
        
        if mint not in self.cache.volume_snapshots:
            self.cache.volume_snapshots[mint] = {
                'snapshots': deque(maxlen=5),
                'last_update': current_time,
                'last_volume': volume,
                'period_volume': 0
            }
        
        data = self.cache.volume_snapshots[mint]
        volume_diff = max(0, volume - data['last_volume'])
        data['period_volume'] += volume_diff
        data['last_volume'] = volume
        
        if current_time - data['last_update'] >= 1.0:
            data['snapshots'].append(data['period_volume'])
            data['last_update'] = current_time
            data['period_volume'] = 0
        
        return sum(data['snapshots']) / len(data['snapshots']) if data['snapshots'] else 0
    
    def get_volume_bars_count(self, avg_vol_1s):
        if avg_vol_1s <= 0: return 0
        elif avg_vol_1s <= 50: return 1
        elif avg_vol_1s <= 150: return 2
        elif avg_vol_1s <= 250: return 3
        elif avg_vol_1s <= 500: return 4
        else: return 5
    
    def get_volume_indicator(self, avg_vol_1s):
        bars = self.get_volume_bars_count(avg_vol_1s)
        return "◻" * bars + "◼" * (5 - bars)
    
    def fast_ema(self, mint, period, price):
        if mint not in self.cache.ema_data:
            self.cache.ema_data[mint] = {3: price, 9: price, 21: price, 'prev': {3: price, 9: price, 21: price}}
        
        data = self.cache.ema_data[mint]
        k = 2 / (period + 1)
        prev_ema = data[period]
        new_ema = (price - prev_ema) * k + prev_ema
        change = ((new_ema - data['prev'][period]) / data['prev'][period] * 100) if data['prev'][period] != 0 else 0
        data['prev'][period] = data[period]
        data[period] = new_ema
        
        return new_ema, change
    
    def fast_volume_stats(self, candles):
        if len(candles) < 1: return 0, "none"
        volumes = [float(c['volume_usd']) for c in candles[-3:]]
        avg = sum(volumes) / len(volumes)
        trend = "rising" if len(volumes) > 1 and volumes[-1] > volumes[0] else "falling" if len(volumes) > 1 else "none"
        return avg, trend
    
    def check_tweezer_top(self, candles):
        if len(candles) < 2: return False
        c1, c2 = candles[-2], candles[-1]
        high1, close1 = float(c1['high_usd']), float(c1['close_usd'])
        high2, close2 = float(c2['high_usd']), float(c2['close_usd'])
        
        if high1 == 0:
            return False
        
        return high1 > close1 and abs(high2 - high1) / high1 <= 0.07 and close2 < close1
    
    async def execute_instant_trade(self, mint, price, avg_vol, ema_changes, positions, alerts, current_candle, flags, avg_vol_1s):
        if not self.autoFarm:
            return False
        
        current_time = time.time()
        
        if mint in self.cache.trade_history and (current_time - self.cache.trade_history[mint]['time']) < self.TRADE_COOLDOWN:
            return False
        
        volume_bars = self.get_volume_bars_count(avg_vol_1s)
        
        # Auto-buy logic
        if mint not in positions:
            if mint in self.cache.trade_history and 'price' in self.cache.trade_history[mint]:
                price_change = abs(price - self.cache.trade_history[mint]['price']) / self.cache.trade_history[mint]['price']
                if price_change < self.MIN_PRICE_CHANGE:
                    return False
            
            buy_conditions = 0
            required_conditions = 0
            
            if self.aBuyBase1:
                required_conditions += 1
                if price > self.INITIAL_PRICE * 1.1 and avg_vol > 500:
                    buy_conditions += 1
            
            if self.aBuyAverage:
                required_conditions += 1
                if ema_changes[3] >= 0 and ema_changes[9] > 0 and ema_changes[21] > 0:
                    buy_conditions += 1
            
            if self.aBuyBuyers:
                required_conditions += 1
                total_txns = current_candle.get('buy_count', 0) + current_candle.get('sell_count', 0)
                if total_txns > 3:
                    buy_ratio = current_candle.get('buy_count', 0) / total_txns
                    if buy_ratio >= 0.6:
                        buy_conditions += 1
            
            if self.aBuyVolBar:
                required_conditions += 1
                if volume_bars >= 3:
                    buy_conditions += 1
            
            should_buy = (self.autoBuyRequireAll and required_conditions > 0 and buy_conditions == required_conditions) or \
                       (not self.autoBuyRequireAll and required_conditions > 0 and buy_conditions > 0)
            
            if should_buy:
                try:
                    result = await buy_token(mint, price)
                    if result.get('success'):
                        self.cache.trade_history[mint] = {'time': current_time, 'action': 'BUY', 'price': price}
                        await self.log_trade_reason("BUY", mint, price, "Auto conditions met")
                        print(f"✅ INSTANT BUY: {mint[:20]}... @ ${price:.10f}")
                        return True
                    else:
                        print(f"❌ BUY FAILED: {mint[:20]}... @ ${price:.10f}")
                        return False
                except Exception as e:
                    print(f"❌ BUY ERROR: {mint[:20]}... @ ${price:.10f} | {str(e)}")
                    return False
        
        # Auto-sell logic
        else:
            position = positions[mint]
            pnl_sol, pnl_percent = self.calculate_pnl(position, price)
            
            if pnl_percent == 0:
                return False
            
            if mint not in self.cache.trailing_highs:
                self.cache.trailing_highs[mint] = pnl_percent
            elif pnl_percent >= self.TRAILING_ACTIVATION_PERCENT:
                self.cache.trailing_highs[mint] = max(self.cache.trailing_highs[mint], pnl_percent)
            
            sell_reason = None
            if self.aSellTakeProfit and pnl_percent >= self.TAKE_PROFIT_PERCENT:
                sell_reason = f"Take Profit ({pnl_percent:.1f}%)"
            elif self.aSellStopLoss and pnl_percent <= self.STOP_LOSS_PERCENT:
                sell_reason = f"Stop Loss ({pnl_percent:.1f}%)"
            elif self.aSellTrailing and pnl_percent >= self.TRAILING_ACTIVATION_PERCENT and (self.cache.trailing_highs[mint] - pnl_percent) >= self.TRAILING_STOP_PERCENT:
                sell_reason = f"Trailing Stop ({pnl_percent:.1f}%)"
            elif self.aSellFlagged and flags:
                sell_reason = f"Flagged ({flags[0]})"
            elif self.aSellPrice and price < self.INITIAL_PRICE * 0.8:
                sell_reason = "Low Price"
            elif self.aSellVolume and avg_vol < 500:
                sell_reason = "Low Volume"
            elif self.aSellEMA and ema_changes[3] < 0 and ema_changes[9] < 0 and ema_changes[21] < 0:
                sell_reason = "EMA Bearish"
            elif self.aSellVolBar and volume_bars <= 1:
                sell_reason = f"Low Volume Bars ({volume_bars})"
            
            if sell_reason:
                try:
                    result = await sell_token(mint, price)
                    if result.get('success'):
                        self.cache.trade_history[mint] = {'time': current_time, 'action': 'SELL', 'price': price}
                        await self.log_trade_reason("SELL", mint, price, sell_reason, position)
                        if mint in self.cache.trailing_highs:
                            del self.cache.trailing_highs[mint]
                        print(f"✅ INSTANT SELL: {mint[:20]}... @ ${price:.10f} | {sell_reason}")
                        return True
                    else:
                        print(f"❌ SELL FAILED: {mint[:20]}... @ ${price:.10f}")
                        return False
                except Exception as e:
                    print(f"❌ SELL ERROR: {mint[:20]}... @ ${price:.10f} | {str(e)}")
                    return False
        
        return False
    
    async def analyze_token_worker(self, token_info):
        mint = token_info.get('mint_address')
        if not mint: return
            
        while self.running:
            try:
                ohlcv_file = f"tokenOHLCV/{mint}_ohlcv.json"
                if not os.path.exists(ohlcv_file):
                    await asyncio.sleep(0.05)
                    continue
                
                try:
                    async with aiofiles.open(ohlcv_file, 'r') as f:
                        data = json.loads(await f.read())
                except:
                    await asyncio.sleep(0.05)
                    continue
                
                if not data.get('ohlcv'):
                    await asyncio.sleep(0.05)
                    continue
                
                current_candle = data['ohlcv'][-1]
                price = float(current_candle['close_usd'])
                current_vol = float(current_candle['volume_usd'])

                #NEW LOGIC
                # Get a pump prediction from RichardML
                pump_prediction = await self.richard_ml.predict(mint, data)

                # Modify your auto-buy logic to use the prediction
                if pump_prediction is not None and pump_prediction > 0.75: # 0.75 is a confidence threshold
                    try:
                        # Instead of fixed settings, the ML model could eventually suggest these
                        # For now, we use the prediction as a trigger
                        result = await buy_token(mint, price)
                        if result.get('success'):
                            self.cache.trade_history[mint] = {'time': time.time(), 'action': 'BUY', 'price': price, 'prediction_confidence': pump_prediction}
                            await self.log_trade_reason("BUY", mint, price, f"RichardML Signal ({pump_prediction:.2f})")
                            print(f"✅ ML-DRIVEN BUY: {mint[:20]}... @ ${price:.10f}")
                            # ...
                    except Exception as e:
                        print(f"❌ ML BUY ERROR: {mint[:20]}... | {str(e)}")
                #END OF NEW LOGIC
                
                avg_vol, vol_trend = self.fast_volume_stats(data['ohlcv'])
                avg_vol_1s = self.update_volume_snapshot(mint, current_vol)
                
                # Calculate EMAs
                ema_changes = {}
                for period in [3, 9, 21]:
                    _, change = self.fast_ema(mint, period, price)
                    ema_changes[period] = change
                
                positions = await self.cache.get_positions()
                total_change = (price - self.INITIAL_PRICE) / self.INITIAL_PRICE * 100
                
                # Generate alerts/warnings/flags
                alerts, warnings, flags = [], [], []
                candles = data['ohlcv']
                
                buy_count = current_candle.get('buy_count', 0)
                sell_count = current_candle.get('sell_count', 0)
                total_txns = buy_count + sell_count
                volume_bars = self.get_volume_bars_count(avg_vol_1s)
                
                # Alerts
                if avg_vol > 2500: alerts.append(f"Volume > 2500 ({avg_vol:.0f})")
                if current_vol > avg_vol * 2: alerts.append("Volume 2x above average")
                if current_vol > 10000: alerts.append("Volume > 10,000")
                if vol_trend == "rising": alerts.append("Volume increasing")
                if total_txns > 0 and buy_count / total_txns >= 0.7: alerts.append("BUYERS DOMINATING")
                if volume_bars == 4: alerts.append("Super Volume")
                elif volume_bars == 5: alerts.append("Exceeding Volume")
                
                # Price spike alerts
                if len(candles) >= 2:
                    prev_price = float(candles[-2]['close_usd'])
                    spike = (price - prev_price) / prev_price * 100
                    if spike > 25: alerts.append(f"Price spike: +{spike:.1f}%")
                    elif spike < -25: warnings.append(f"Price drop: {spike:.1f}%")
                
                # EMA alerts
                if ema_changes[3] > 0 and ema_changes[9] > 0 and ema_changes[21] > 0:
                    alerts.append("EMA increasing")
                    if ema_changes[21] > 10: alerts.append("EMA VERY BULLISH")
                
                # Flags and warnings
                added_time = token_info.get('added_at', time.time())
                grace_period = (time.time() - added_time) < 30
                immune = price > self.INITIAL_PRICE * 2.0
                
                if not grace_period and not immune:
                    if avg_vol < 1000: flags.append("Low volume")
                    if price < self.INITIAL_PRICE * 0.8: flags.append("Low price")
                    if self.check_tweezer_top(candles): flags.append("Tweezer Top")
                
                # Warnings
                if price < self.INITIAL_PRICE * 0.8: warnings.append("WARNING: LOW PRICE")
                if avg_vol < 1000: warnings.append("Average volume < 1000")
                if vol_trend == "falling": warnings.append("Volume decreasing")
                if total_txns > 0 and sell_count / total_txns >= 0.7: warnings.append("SELLERS DOMINATING")
                if volume_bars == 1: warnings.append("Volume Dying")
                elif volume_bars == 0: warnings.append("Dead Volume")
                
                if ema_changes[3] < 0 and ema_changes[9] < 0 and ema_changes[21] < 0:
                    warnings.append("EMA decreasing")
                    if ema_changes[21] < -10: warnings.append("EMA VERY BEARISH")
                
                # Update status when flagged
                if flags:
                    current_status = token_info.get('status', '')
                    if current_status != 'FLAGGED':
                        token_info['status'] = 'FLAGGED'
                        token_info['flag_reason'] = flags[0]
                        await self.update_contract_status(mint, 'FLAGGED', flags[0])
                
                status = "FLAGGED" if flags else "EXCEEDING" if (avg_vol > 2000 and price > 0.00001) else "GOOD"
                
                # Execute trade
                traded = await self.execute_instant_trade(mint, price, avg_vol, ema_changes, positions, alerts, current_candle, flags, avg_vol_1s)
                
                result = {
                    'mint': mint, 'token_info': token_info, 'price': price, 'total_change': total_change,
                    'ema_changes': ema_changes, 'current_vol': current_vol, 'avg_vol': avg_vol,
                    'avg_vol_1s': avg_vol_1s, 'vol_trend': vol_trend, 'current_candle': current_candle, 
                    'total_candles': data['total_candles'], 'status': status, 'alerts': alerts, 
                    'warnings': warnings, 'flags': flags, 'grace_period': grace_period, 
                    'immune': immune, 'has_position': mint in positions, 'traded': traded
                }
                
                async with self.results_lock:
                    self.results[mint] = result
                
                await asyncio.sleep(0.05)
                
            except Exception as e:
                if self.debugPrint:
                    print(f"Error in worker for {mint}: {e}")
                await asyncio.sleep(0.5)
    
    async def display_summary(self):
        current_time = time.time()
        if current_time - self.last_display < self.display_interval:
            return
        
        if self.clear_screen:
            print("\033[H\033[J", end="")
        
        print(f"PAPER TRADING: {'ON' if PAPER_TRADE else 'OFF'} | {datetime.now().strftime('%H:%M:%S')}")
        if not self.autoFarm: print("autoFarm is turned OFF")
        
        analyses = list(self.results.values())
        print()
        
        for i, analysis in enumerate(analyses):
            if i > 0: print("-" * 80)
            
            mint = analysis['mint']
            price = analysis['price']
            total_change = analysis['total_change']
            ema_changes = analysis['ema_changes']
            current_vol = analysis['current_vol']
            avg_vol = analysis['avg_vol']
            avg_vol_1s = analysis['avg_vol_1s']
            vol_trend = analysis['vol_trend']
            current_candle = analysis['current_candle']
            total_candles = analysis['total_candles']
            status = analysis['status']
            alerts = analysis['alerts']
            warnings = analysis['warnings']
            flags = analysis['flags']
            grace_period = analysis['grace_period']
            has_position = analysis['has_position']
            
            # Status indicators
            if mint in self.cache.trade_history:
                time_since_last = current_time - self.cache.trade_history[mint]['time']
                if time_since_last < self.TRADE_COOLDOWN:
                    print(f"COOLDOWN: {int(self.TRADE_COOLDOWN - time_since_last)}s")
            
            if grace_period:
                remaining = max(0, 30 - (current_time - analysis['token_info'].get('added_at', current_time)))
                print(f"GRACE: {int(remaining)}s")
            elif analysis['immune']:
                print("IMMUNITY")
            
            pos_indicator = " | POS" if has_position else ""
            print(f"TOKEN MINT: {mint}{pos_indicator}")
            print(f"CURRENT PRICE: {price:.10f} (TOTAL % CHANGE): {total_change:.2f}%")
            
            # EMA info
            print("PRICE EMA:")
            for period in [3, 9, 21]:
                change = ema_changes[period]
                arrow = "↑" if change > 0 else "↓" if change < 0 else "-"
                color = "\033[92m" if change > 0 else "\033[91m" if change < 0 else "\033[0m"
                print(f"  {period}s Average: {change:.2f}% ({arrow}){color}\033[0m")
            
            # Volume info
            vol_color = "\033[92m" if vol_trend == "rising" else "\033[91m" if vol_trend == "falling" else "\033[0m"
            print(f"CURRENT VOL: {current_vol:.0f} AVERAGE VOL: {avg_vol:.0f} ({vol_trend}){vol_color}\033[0m")
            print(f"1S AVG VOL: {avg_vol_1s:.0f} (5s rolling)")
            print(f"VOL BARS: {self.get_volume_indicator(avg_vol_1s)}")
            print(f"BUYS: {current_candle.get('buy_count', 0)} SELLS: {current_candle.get('sell_count', 0)}")
            print(f"TOTAL CANDLES: {total_candles}")
            
            # Status
            status_color = "\033[91m" if status == "FLAGGED" else "\033[92m" if status == "GOOD" else "\033[93m"
            print(f"STATUS: {status_color}{status}\033[0m")
            
            # Alerts, warnings, flags
            print("\nALERTS, WARNINGS, or REASON FOR FLAG:")
            for alert in alerts: print(f"⚠️  {alert}")
            for warning in warnings: print(f"\033[91m🚨 {warning}\033[0m")
            for flag in flags: print(f"\033[91m🚩 {flag}\033[0m")
            
            if not alerts and not warnings and not flags:
                print("  None")
        
        self.last_display = current_time
    
    async def manage_workers(self):
        while self.running:
            try:
                if not os.path.exists('contract_addresses.json'):
                    await asyncio.sleep(1)
                    continue
                
                try:
                    async with aiofiles.open('contract_addresses.json', 'r') as f:
                        contract_data = json.loads(await f.read())
                except:
                    await asyncio.sleep(1)
                    continue
                
                current_mints = {token.get('mint_address') for token in contract_data if token.get('mint_address')}
                existing_mints = set(self.token_workers.keys())
                
                # Start new workers
                for token in contract_data:
                    mint = token.get('mint_address')
                    if mint and mint not in existing_mints:
                        task = asyncio.create_task(self.analyze_token_worker(token))
                        self.token_workers[mint] = task
                
                # Remove old workers
                for mint in existing_mints - current_mints:
                    if mint in self.token_workers:
                        self.token_workers[mint].cancel()
                        del self.token_workers[mint]
                    async with self.results_lock:
                        if mint in self.results:
                            del self.results[mint]
                
                await asyncio.sleep(1)
                
            except Exception as e:
                if self.debugPrint:
                    print(f"Manager error: {e}")
                await asyncio.sleep(1)
    
    async def periodic_flush(self):
        while self.running:
            try:
                await self.flush_status_updates()
                await self.trade_logger._flush()
                await asyncio.sleep(5)
            except Exception as e:
                if self.debugPrint:
                    print(f"Flush error: {e}")
                await asyncio.sleep(5)
    
    async def display_loop(self):
        while self.running:
            try:
                await self.display_summary()
                await asyncio.sleep(1)
            except Exception as e:
                if self.debugPrint:
                    print(f"Display error: {e}")
                await asyncio.sleep(1)
    
    async def run(self):
        """Main entry point to start the token analyzer"""
        try:
            # Start all concurrent tasks
            tasks = [
                asyncio.create_task(self.manage_workers()),
                asyncio.create_task(self.periodic_flush()),
                asyncio.create_task(self.display_loop())
            ]
            
            # Wait for all tasks to complete
            await asyncio.gather(*tasks)
            
        except KeyboardInterrupt:
            print("\nShutting down...")
            self.running = False
            
            # Cancel all worker tasks
            for task in self.token_workers.values():
                task.cancel()
            
            # Cancel main tasks
            for task in tasks:
                task.cancel()
            
            # Final flush
            await self.trade_logger._flush()
            await self.flush_status_updates()
            
        except Exception as e:
            if self.debugPrint:
                print(f"Main loop error: {e}")
            self.running = False

# Main execution
# In analyze.py, at the bottom
if __name__ == "__main__":
    analyzer = TokenAnalyzer()

    async def main():
        # This task will run the token analysis and trading
        analysis_task = asyncio.create_task(analyzer.run())

        # This loop will periodically retrain the ML model
        while analyzer.running:
            await asyncio.sleep(600) # Retrain every 10 minutes
            
            if len(analyzer.trade_outcomes) > 20: # Make sure you have enough new data
                print("🧠 Retraining RichardML with new trade data...")
                
                # You may need to run this in an executor to avoid blocking the event loop
                loop = asyncio.get_running_loop()
                trade_outcomes_copy = analyzer.trade_outcomes.copy()
                analyzer.trade_outcomes.clear() # Clear for the next batch

                await loop.run_in_executor(
                    None, # Use the default executor
                    analyzer.richard_ml.train_model_sync,
                    trade_outcomes_copy
                )
        
        await analysis_task

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Shutting down bot.")