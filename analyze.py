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

class TokenAnalyzer:
    def __init__(self):
        # Configuration
        self.debugPrint = False
        self.INITIAL_PRICE = 0.0000054
        self.TRADE_COOLDOWN = 5
        self.MIN_PRICE_CHANGE = 0.05
        self.initialPercent = 0.0
        self.buyDuringGrace = True
        
        # Trading fees
        self.JITO_TIP = 0.00007
        self.TIP_FEE_PERCENT = 0.01
        self.BCURVE_FEE_PERCENT = 0.01
        
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
        
        # ML Integration
        self.richard_ml = RichardML(snapshot_interval=5)
        self.trade_outcomes_log = 'trade_outcomes.jsonl'
        self.last_retrain_time = 0
        self.richard_logger = self.BatchWriter('richard_trades.log')
    
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
        """Fixed volume snapshot calculation - tracks volume within each second"""
        current_time = time.time()
        
        if mint not in self.cache.volume_snapshots:
            self.cache.volume_snapshots[mint] = {
                'snapshots': deque(maxlen=5),
                'last_volume': volume,
                'last_snapshot_time': current_time,
                'current_second_volume': 0
            }
        
        data = self.cache.volume_snapshots[mint]
        
        # Calculate volume difference since last update
        volume_diff = max(0, volume - data['last_volume'])
        data['current_second_volume'] += volume_diff
        data['last_volume'] = volume
        
        # Check if we need to create a new snapshot (every 1 second)
        if current_time - data['last_snapshot_time'] >= 1.0:
            data['snapshots'].append(data['current_second_volume'])
            data['last_snapshot_time'] = current_time
            data['current_second_volume'] = 0
        
        # Return 5-second rolling average
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
        """Fixed EMA calculation with proper initialization"""
        if mint not in self.cache.ema_data:
            self.cache.ema_data[mint] = {
                3: price, 9: price, 21: price,
                'prev': {3: price, 9: price, 21: price},
                'initialized': {3: False, 9: False, 21: False}
            }
        
        data = self.cache.ema_data[mint]
        k = 2 / (period + 1)
        
        if not data['initialized'][period]:
            # First calculation
            data[period] = price
            data['prev'][period] = price
            data['initialized'][period] = True
            return price, 0.0
        
        # Calculate EMA
        prev_ema = data[period]
        new_ema = (price * k) + (prev_ema * (1 - k))
        
        # Calculate change from previous EMA
        change = ((new_ema - data['prev'][period]) / data['prev'][period] * 100) if data['prev'][period] != 0 else 0
        
        # Update values
        data['prev'][period] = data[period]
        data[period] = new_ema
        
        return new_ema, change
    
    def fast_volume_stats(self, candles):
        if len(candles) < 1: return 0, "none"
        volumes = [float(c['volume_usd']) for c in candles[-3:]]
        avg = sum(volumes) / len(volumes)
        trend = "rising" if len(volumes) > 1 and volumes[-1] > volumes[0] else "falling" if len(volumes) > 1 else "none"
        return avg, trend
    
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
                
                # Fixed: Always calculate volume stats
                avg_vol, vol_trend = self.fast_volume_stats(data['ohlcv'])
                avg_vol_1s = self.update_volume_snapshot(mint, current_vol)
                
                # Fixed: Always calculate EMAs with every price update
                ema_changes = {}
                for period in [3, 9, 21]:
                    ema_val, change = self.fast_ema(mint, period, price)
                    ema_changes[period] = change
                
                positions = await self.cache.get_positions()
                total_change = (price - self.INITIAL_PRICE) / self.INITIAL_PRICE * 100
                
                candles = data['ohlcv']
                buy_count = current_candle.get('buy_count', 0)
                sell_count = current_candle.get('sell_count', 0)
                total_txns = buy_count + sell_count
                volume_bars = self.get_volume_bars_count(avg_vol_1s)
                
                # Grace period and immunity calculations
                added_time = token_info.get('added_at', time.time())
                grace_period = (time.time() - added_time) < 30
                immune = price > self.INITIAL_PRICE * 2.0
                
                # Fixed: ML Prediction with proper initialization check and fallback
                prediction_score = 0.0
                try:
                    # Check if RichardML is properly initialized
                    if hasattr(self.richard_ml, 'model') and self.richard_ml.model is not None:
                        prediction = await self.richard_ml.predict(mint, data)
                        prediction_score = prediction if prediction is not None else 0.0
                    else:
                        # Use simple fallback prediction based on price momentum until ML is trained
                        if len(data['ohlcv']) >= 3:
                            recent_prices = [float(c['close_usd']) for c in data['ohlcv'][-3:]]
                            price_momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
                            # Convert momentum to 0-1 prediction score
                            prediction_score = max(0.0, min(1.0, 0.5 + (price_momentum * 10)))
                        else:
                            prediction_score = 0.5  # Neutral when insufficient data
                        
                        if self.debugPrint:
                            print(f"Using fallback prediction for {mint}: {prediction_score:.3f}")
                            
                except Exception as e:
                    if self.debugPrint:
                        print(f"ML prediction error for {mint}: {e}")
                    # Emergency fallback based on EMA trend
                    if ema_changes.get(9, 0) > 0:
                        prediction_score = 0.6  # Slight bullish
                    elif ema_changes.get(9, 0) < 0:
                        prediction_score = 0.4  # Slight bearish  
                    else:
                        prediction_score = 0.5  # Neutral
                
                # ML Trading Logic
                if not (mint in positions) and prediction_score > 0.55:
                    print(f"✅ RICHARD BUY: {mint[:10]}... Score: {prediction_score:.2f}")
                    await buy_token(mint, price)
                    await self.richard_logger.write(f"[{datetime.now()}] RICHARD BUY: {mint} @ ${price:.10f} | Score: {prediction_score:.3f}\n")
                elif (mint in positions) and prediction_score < 0.45:
                    print(f"✅ RICHARD SELL: {mint[:10]}... Score: {prediction_score:.2f}")
                    pnl_sol, pnl_percent = self.calculate_pnl(positions[mint], price)
                    await sell_token(mint, price)
                    await self.richard_logger.write(f"[{datetime.now()}] RICHARD SELL: {mint} @ ${price:.10f} | Score: {prediction_score:.3f} | PnL: {pnl_sol:.4f} SOL ({pnl_percent:.2f}%)\n")
                    print(f"Trade Closed. PnL: {pnl_sol:.4f} SOL ({pnl_percent:.2f}%)")
                
                result = {
                    'mint': mint, 'token_info': token_info, 'price': price, 'total_change': total_change,
                    'ema_changes': ema_changes, 'current_vol': current_vol, 'avg_vol': avg_vol,
                    'avg_vol_1s': avg_vol_1s, 'vol_trend': vol_trend, 'current_candle': current_candle, 
                    'total_candles': data['total_candles'], 'grace_period': grace_period, 
                    'immune': immune, 'has_position': mint in positions, 'prediction_score': prediction_score
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
            print(f"ML PREDICTION SCORE: {analysis.get('prediction_score', 0.0):.3f}")
            print(f"TOTAL CANDLES: {total_candles}")
        
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
                
                # ML training every 30 seconds with initialization check
                if time.time() - self.last_retrain_time > 30:
                    print("🤖 RICHARD learning from trades...")
                    try:
                        # Force initialize model if it doesn't exist
                        if not hasattr(self.richard_ml, 'model') or self.richard_ml.model is None:
                            print("🔧 Initializing RichardML model...")
                            # Try to create minimal training data from current results
                            if len(self.results) > 0:
                                self.richard_ml.initialize_with_current_data(list(self.results.values()))
                        
                        self.richard_ml.train_model_sync()
                        self.last_retrain_time = time.time()
                        print("✅ RichardML training completed")
                    except Exception as e:
                        if self.debugPrint:
                            print(f"ML training error: {e}")
                        # Create a simple dummy model as fallback
                        print("⚠️ Using fallback prediction system")
                
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
                await self.richard_logger._flush()
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
            await self.richard_logger._flush()
            await self.flush_status_updates()
            
        except Exception as e:
            if self.debugPrint:
                print(f"Main loop error: {e}")
            self.running = False

# Main execution
if __name__ == "__main__":
    analyzer = TokenAnalyzer()
    asyncio.run(analyzer.run())