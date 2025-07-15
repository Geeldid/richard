#!/usr/bin/env python3
import json
import os
import time
import logging
import asyncio
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
import glob
from scipy import stats
from sklearn.preprocessing import StandardScaler
import concurrent.futures

class PumpNet(nn.Module):
    def __init__(self, input_size=25, hidden_size=48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_size, 2)
        )
    
    def forward(self, x):
        return self.net(x)

class RichardML:
    def __init__(self, snapshot_interval=5, monitoring_period_seconds=30, max_duds=150):
        self.setup_logging()
        self.snapshot_interval = snapshot_interval
        self.monitoring_period_seconds = monitoring_period_seconds
        self.max_duds = max_duds
        self.memory_file = f"richard_memory_{snapshot_interval}s.json"
        self.pump_memory_file = f"richard_pump_memory_{snapshot_interval}s.json"
        self.model_file = f"richard_model_{snapshot_interval}s.pth"
        self.scaler_file = f"scaler_{snapshot_interval}s.pkl"
        self.contracts_file = "contract_addresses.json"
        
        self.memory = self.load_memory()
        self.pump_memory = self.load_pump_memory()
        self.last_snapshot_time = self.memory.get("last_snapshot_time", {})
        self.migrate_pump_data()
        self.active_contracts = set()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = PumpNet().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        self.criterion = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.tokens_processed = 0
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        
        self.load_model()
        self.load_scaler()
        self.update_active_contracts()
        self.logger.info(f"RichardML initialized with {snapshot_interval}s intervals, {monitoring_period_seconds}s monitoring")
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('richard.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('RichardML')
        
    def load_memory(self):
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {"tokens": {}, "token_states": {}, "last_snapshot_time": {}}
    
    def load_pump_memory(self):
        if os.path.exists(self.pump_memory_file):
            try:
                with open(self.pump_memory_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {"tokens": {}, "token_states": {}, "last_snapshot_time": {}}
    
    def migrate_pump_data(self):
        pumps_to_migrate = {mint: data for mint, data in self.memory["tokens"].items() 
                           if data.get("pump_status") == "confirmed_pump"}
        
        for mint, data in pumps_to_migrate.items():
            self.pump_memory["tokens"][mint] = data
            if mint in self.memory["token_states"]:
                self.pump_memory["token_states"][mint] = self.memory["token_states"][mint]
            del self.memory["tokens"][mint]
            if mint in self.memory["token_states"]:
                del self.memory["token_states"][mint]
        
        if pumps_to_migrate:
            self.logger.info(f"Migrated {len(pumps_to_migrate)} confirmed pumps to pump memory")
            self.save_memory()
            self.save_pump_memory()
    
    def save_memory(self):
        self.memory["last_snapshot_time"] = self.last_snapshot_time
        with open(self.memory_file, 'w') as f:
            json.dump(self.memory, f, indent=2)
    
    def save_pump_memory(self):
        with open(self.pump_memory_file, 'w') as f:
            json.dump(self.pump_memory, f, indent=2)
    
    def load_model(self):
        if os.path.exists(self.model_file):
            try:
                self.model.load_state_dict(torch.load(self.model_file, map_location=self.device))
                self.is_trained = True
                self.logger.info("Model loaded")
            except Exception as e:
                self.logger.error(f"Failed to load model: {e}")
    
    def load_scaler(self):
        if os.path.exists(self.scaler_file):
            try:
                self.scaler = joblib.load(self.scaler_file)
            except Exception as e:
                self.logger.error(f"Failed to load scaler: {e}")
    
    def save_model(self):
        torch.save(self.model.state_dict(), self.model_file)
        joblib.dump(self.scaler, self.scaler_file)
    
    def update_active_contracts(self):
        try:
            if os.path.exists(self.contracts_file):
                with open(self.contracts_file, 'r') as f:
                    contracts_data = json.load(f)
                    if isinstance(contracts_data, list):
                        self.active_contracts = set(item.get("mint_address") for item in contracts_data if item.get("mint_address"))
                    elif isinstance(contracts_data, dict):
                        self.active_contracts = set(contracts_data.keys())
                    else:
                        self.active_contracts = set()
            else:
                self.active_contracts = set()
        except Exception as e:
            self.logger.error(f"Error loading contracts: {e}")
            self.active_contracts = set()
    
    def cleanup_inactive_tokens(self):
        if not self.active_contracts:
            self.logger.info("No active contracts defined - monitoring all tokens")
            return
        
        tokens_to_mark_dead = []
        for mint in list(self.memory["tokens"].keys()):
            if mint not in self.active_contracts and self.memory["token_states"].get(mint) != "dead":
                tokens_to_mark_dead.append(mint)
        
        for mint in tokens_to_mark_dead:
            self.memory["token_states"][mint] = "dead"
        
        if tokens_to_mark_dead:
            self.logger.info(f"Marked {len(tokens_to_mark_dead)} inactive tokens as dead")
    
    def prune_memory(self):
        dud_mints = {mint for mint, data in self.memory["tokens"].items() if data.get("pump_status") in ["dud", "pump_fake", "pump_and_dump"]}

        if len(dud_mints) > self.max_duds:
            duds_to_remove = sorted(list(dud_mints), key=lambda m: self.memory["tokens"][m].get("first_seen", ""))
            num_to_remove = len(duds_to_remove) - self.max_duds
            mints_to_remove = duds_to_remove[:num_to_remove]

            for mint in mints_to_remove:
                del self.memory["tokens"][mint]
                if mint in self.memory["token_states"]:
                    del self.memory["token_states"][mint]
                if mint in self.last_snapshot_time:
                    del self.last_snapshot_time[mint]
            
            self.logger.info(f"Pruned {len(mints_to_remove)} old dud tokens from memory.")
    
    def should_take_snapshot(self, mint, current_time):
        if mint not in self.last_snapshot_time:
            self.last_snapshot_time[mint] = 0
        
        if current_time - self.last_snapshot_time[mint] >= self.snapshot_interval:
            self.last_snapshot_time[mint] = current_time
            return True
        return False
    
    def classify_token_state(self, snapshots):
        if len(snapshots) < 3:
            return "new"
        
        first_timestamp = snapshots[0]["timestamp"]
        latest_timestamp = snapshots[-1]["timestamp"]
        time_elapsed = latest_timestamp - first_timestamp
        
        if time_elapsed < self.monitoring_period_seconds:
            return "monitoring"
        
        recent_volumes = [s["volume"] for s in snapshots[-5:]]
        avg_volume = np.mean(recent_volumes)
        
        if avg_volume < 100:
            return "dead"
        elif len(snapshots) > 50:
            return "mature"
        else:
            return "active"
    
    def checkForPump(self, snapshots):
        if len(snapshots) < 5:
            return False, "insufficient_data"

        prices = [s["price"] for s in snapshots]
        volumes = [s["volume"] for s in snapshots]

        baseline_price = np.mean([s["price"] for s in snapshots[:3]])
        if baseline_price <= 1e-9:
            return False, "invalid_baseline"

        max_price = max(prices)
        pump_multiplier = max_price / baseline_price

        baseline_volume = np.mean([s["volume"] for s in snapshots[:3]])
        max_volume = max(volumes)
        volume_spike = max_volume / (baseline_volume + 1e-9)

        cond1 = pump_multiplier >= 1.75 and volume_spike >= 1.3
        cond2 = pump_multiplier >= 1.4 and volume_spike >= 1.8

        if cond1 or cond2:
            return True, "confirmed_pump"
        elif pump_multiplier >= 1.3 and volume_spike >= 1.5:
            return True, "pump_fake"
        else:
            return False, "dud"
    
    def check_pump_and_dump(self, mint, current_price):
        """Check if a confirmed pump token has crashed and should be marked as pump_and_dump"""
        if mint not in self.pump_memory["tokens"]:
            return False
        
        token_data = self.pump_memory["tokens"][mint]
        if token_data.get("pump_status") != "confirmed_pump":
            return False
        
        peak_price = token_data.get("peak_price")
        if not peak_price:
            return False
        
        # Check if price dropped by 40% from peak
        if current_price <= peak_price * 0.6:
            token_data["pump_status"] = "pump_and_dump"
            self.logger.info(f"Token {mint[:8]}... marked as pump_and_dump (dropped {((peak_price - current_price) / peak_price * 100):.1f}%)")
            return True
        
        return False
    
    def calculate_ema(self, prices, span):
        if len(prices) < span:
            return np.mean(prices) if prices else 0
        
        alpha = 2.0 / (span + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
        return ema
    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        if len(prices) < slow:
            return 0, 0, 0
        
        ema_fast = self.calculate_ema(prices, fast)
        ema_slow = self.calculate_ema(prices, slow)
        macd_line = ema_fast - ema_slow
        
        signal_line = macd_line * 0.8
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def calculate_rsi(self, prices, period=14):
        if len(prices) < period + 1:
            return 50
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def safe_divide(self, a, b, default=0):
        if abs(b) < 1e-10:
            return default
        return a / b
    
    def extract_features(self, snapshots, candles=None):
        if len(snapshots) < 3:
            return None
        
        recent_snapshots = snapshots[-6:]
        
        prices = [s["price"] for s in recent_snapshots]
        volumes = [s["volume"] for s in recent_snapshots]
        
        if not prices or not volumes:
            return None
        
        price_change = self.safe_divide(prices[-1] - prices[0], prices[0])
        mean_price = np.mean(prices)
        volatility = self.safe_divide(np.std(prices), mean_price) if mean_price > 0 else 0
        avg_volume = np.mean(volumes)
        volume_trend = self.safe_divide(volumes[-1] - volumes[0], volumes[0])
        
        momentum = self.safe_divide(prices[-1] - prices[-2], prices[-2]) if len(prices) > 1 else 0
        acceleration = self.safe_divide(prices[-1] - 2*prices[-2] + prices[-3], prices[-3]) if len(prices) > 2 else 0
        volume_acceleration = self.safe_divide(volumes[-1] - 2*volumes[-2] + volumes[-3], volumes[-3]) if len(volumes) > 2 else 0
        
        rsi = self.calculate_rsi(prices)
        macd_line, signal_line, histogram = self.calculate_macd(prices)
        
        price_skew = stats.skew(prices) if len(prices) > 2 else 0
        volume_skew = stats.skew(volumes) if len(volumes) > 2 else 0
        price_range = self.safe_divide(np.max(prices), np.min(prices), 1)
        
        try:
            vol_price_corr = np.corrcoef(volumes, prices)[0, 1]
            vol_price_corr = 0 if np.isnan(vol_price_corr) else vol_price_corr
        except:
            vol_price_corr = 0
        
        features = [
            price_change, volatility, avg_volume, volume_trend,
            momentum, acceleration, volume_acceleration, rsi,
            macd_line, histogram, price_skew, volume_skew,
            price_range, vol_price_corr, len(recent_snapshots),
            np.sum(volumes), np.std(volumes), mean_price,
            self.safe_divide(prices[-1], prices[0], 1),
            self.safe_divide(volumes[-1], volumes[0], 1),
            self.safe_divide(np.max(volumes), np.mean(volumes), 1),
            self.safe_divide(np.max(prices), mean_price, 1),
            self.safe_divide(np.max(prices) - np.min(prices), np.min(prices)),
            self.safe_divide(1, 1 + volatility, 1) if volatility > 0 else 1,
            np.gradient(prices)[-1] if len(prices) > 1 else 0
        ]
        
        features = [f if np.isfinite(f) else 0 for f in features]
        return features
    
    async def update_token_data(self, mint, ohlcv_data):
        current_time = time.time()
        
        if not self.should_take_snapshot(mint, current_time):
            return
        
        if mint not in self.memory["tokens"]:
            self.memory["tokens"][mint] = {
                "first_seen": datetime.now().isoformat(),
                "snapshots": [],
                "pump_status": None
            }
        
        latest_candle = ohlcv_data.get('ohlcv', [])[-1] if ohlcv_data.get('ohlcv') else None
        if latest_candle:
            snapshot = {
                "timestamp": latest_candle["timestamp"],
                "price": float(latest_candle["close_usd"]),
                "volume": float(latest_candle["volume_usd"])
            }
            
            current_price = snapshot["price"]
            
            # Check for pump and dump before updating snapshots
            self.check_pump_and_dump(mint, current_price)
            
            self.memory["tokens"][mint]["snapshots"].append(snapshot)
            
            max_snapshots = max(100, self.monitoring_period_seconds // self.snapshot_interval + 20)
            if len(self.memory["tokens"][mint]["snapshots"]) > max_snapshots:
                self.memory["tokens"][mint]["snapshots"] = self.memory["tokens"][mint]["snapshots"][-max_snapshots:]
            
            old_state = self.memory["token_states"].get(mint, "new")
            new_state = self.classify_token_state(self.memory["tokens"][mint]["snapshots"])
            self.memory["token_states"][mint] = new_state
            
            if old_state == "monitoring" and new_state in ["active", "dead"]:
                is_pump, pump_type = self.checkForPump(self.memory["tokens"][mint]["snapshots"])
                self.memory["tokens"][mint]["pump_status"] = pump_type
                
                if pump_type == "confirmed_pump":
                    # Calculate pump multiplier and peak price
                    snapshots = self.memory["tokens"][mint]["snapshots"]
                    baseline_price = np.mean([s["price"] for s in snapshots[:3]])
                    peak_price = max(s["price"] for s in snapshots)
                    pump_multiplier = peak_price / baseline_price
                    
                    # Save pump data
                    self.memory["tokens"][mint]["pump_multiplier"] = pump_multiplier
                    self.memory["tokens"][mint]["peak_price"] = peak_price
                    
                    # Move to pump memory
                    self.pump_memory["tokens"][mint] = self.memory["tokens"][mint]
                    self.pump_memory["token_states"][mint] = self.memory["token_states"][mint]
                    del self.memory["tokens"][mint]
                    del self.memory["token_states"][mint]
                    self.logger.info(f"Confirmed pump detected for {mint[:8]}... [{pump_type}] (multiplier: {pump_multiplier:.2f}x) - moved to pump memory")
                elif is_pump:
                    self.logger.info(f"Confirmed pump detected for {mint[:8]}... [{pump_type}]")
    
    def prepare_training_data(self):
        X, y, sample_weights = [], [], []
        
        all_tokens = {**self.memory["tokens"], **self.pump_memory["tokens"]}
        
        for mint, token_data in all_tokens.items():
            snapshots = token_data.get("snapshots", [])
            pump_status = token_data.get("pump_status")
            
            if len(snapshots) < 6 or pump_status is None:
                continue
            
            features = self.extract_features(snapshots)
            if features is None:
                continue
            
            if pump_status == "confirmed_pump":
                label = 1
                # Weighted learning by pump magnitude
                pump_multiplier = token_data.get("pump_multiplier", 1.0)
                weight = 1.0 + np.log(pump_multiplier)
            elif pump_status in ["pump_fake", "dud", "pump_and_dump"]:
                label = 0
                weight = 1.0
            else:
                continue
            
            X.append(features)
            y.append(label)
            sample_weights.append(weight)
        
        X, y, sample_weights = np.array(X), np.array(y), np.array(sample_weights)
        
        pump_count = np.sum(y == 1)
        dud_count = np.sum(y == 0)
        self.logger.info(f"Training data: {len(X)} samples, {pump_count} confirmed pumps, {dud_count} duds/fakes/pump_and_dumps")
        return X, y, sample_weights
    
    def train_model_sync(self):
        X, y, sample_weights = self.prepare_training_data()
        
        if len(X) < 20:
            self.logger.warning(f"Not enough training data: {len(X)} samples")
            return False
        
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            self.logger.warning("Invalid features detected, cleaning...")
            mask = np.all(np.isfinite(X), axis=1)
            X, y, sample_weights = X[mask], y[mask], sample_weights[mask]
        
        if len(y) < 20 or len(np.unique(y)) < 2:
            self.logger.warning(f"Not enough valid data or classes for training: {len(y)} samples, {len(np.unique(y))} classes")
            return False
        
        dud_count = np.sum(y == 0)
        pump_count = np.sum(y == 1)
        
        if dud_count == 0 or pump_count == 0:
            self.logger.warning("Training data only contains one class. Skipping training.")
            return False
        
        dud_weight = (dud_count + pump_count) / (2.0 * dud_count)
        pump_weight = (dud_count + pump_count) / (2.0 * pump_count)
        class_weights = torch.tensor([dud_weight, pump_weight], dtype=torch.float32).to(self.device)
        
        self.logger.info(f"Class weights -> Dud: {dud_weight:.2f}, Pump: {pump_weight:.2f}")
        self.criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='none')
        
        X_scaled = self.scaler.fit_transform(X)
        
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        weights_tensor = torch.FloatTensor(sample_weights).to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor, weights_tensor)
        dataloader = DataLoader(dataset, batch_size=min(32, len(X)), shuffle=True)
        
        epochs = 50
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y, batch_weights in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                losses = self.criterion(outputs, batch_y)
                # Apply sample weights
                weighted_loss = torch.mean(losses * batch_weights)
                weighted_loss.backward()
                self.optimizer.step()
                total_loss += weighted_loss.item()
            
            if epoch % 10 == 0:
                self.logger.info(f"Epoch {epoch}, Weighted Loss: {total_loss/len(dataloader):.4f}")
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            y_pred = torch.argmax(outputs, dim=1).cpu().numpy()
            
            for class_idx in range(2):
                class_mask = (y == class_idx)
                if np.sum(class_mask) > 0:
                    accuracy = np.mean(y_pred[class_mask] == class_idx)
                    class_name = ["Dud/Fake/Pump&Dump", "Confirmed Pump"][class_idx]
                    self.logger.info(f"{class_name} accuracy: {accuracy:.3f}")
        
        self.is_trained = True
        self.save_model()
        self.logger.info("Training completed")
        return True
    
    async def predict(self, mint, ohlcv_data):
        if not self.is_trained:
            return None
        
        token_data = self.memory["tokens"].get(mint, {})
        token_snapshots = token_data.get("snapshots", [])
        token_state = self.memory["token_states"].get(mint, "new")
        
        if token_state not in ["new", "monitoring"] or len(token_snapshots) < 4:
            return None
        
        features = self.extract_features(token_snapshots)
        if features is None:
            return None
        
        try:
            features_scaled = self.scaler.transform([features])
            
            self.model.eval()
            with torch.no_grad():
                x_tensor = torch.FloatTensor(features_scaled).to(self.device)
                outputs = self.model(x_tensor)
                probabilities = torch.softmax(outputs, dim=1)[0]
                
                pump_prob = probabilities[1].item()
            
            if pump_prob > 0.5:
                with open(f"predictions_{self.snapshot_interval}s.log", 'a') as f:
                    log_entry = {
                        "timestamp": datetime.now().isoformat(),
                        "mint": mint,
                        "pump_prob": pump_prob,
                        "state": token_state,
                        "snapshots_count": len(token_snapshots)
                    }
                    f.write(f"{json.dumps(log_entry)}\n")
            
            return pump_prob
        except Exception as e:
            self.logger.error(f"Prediction error for {mint}: {e}")
            return None
    
    async def scan_tokens(self):
        self.update_active_contracts()
        self.cleanup_inactive_tokens()
        
        token_files = glob.glob("tokenOHLCV/*_ohlcv.json")
        
        active_files = []
        for file_path in token_files:
            try:
                mint = os.path.basename(file_path).replace('_ohlcv.json', '')
                if not self.active_contracts or mint in self.active_contracts:
                    state = self.memory["token_states"].get(mint, "new")
                    if state in ["active", "new", "monitoring"]:
                        active_files.append(file_path)
            except:
                continue
        
        tasks = []
        for file_path in active_files:
            tasks.append(self.process_file(file_path))
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def process_file(self, file_path):
        try:
            with open(file_path, 'r') as f:
                ohlcv_data = json.load(f)
            
            mint = ohlcv_data.get('mint')
            if not mint:
                return
            
            await self.update_token_data(mint, ohlcv_data)
            
            if self.is_trained:
                prediction = await self.predict(mint, ohlcv_data)
                if prediction is not None and prediction > 0.7:
                    state = self.memory["token_states"].get(mint, "new")
                    self.logger.info(f"PUMP SIGNAL {mint[:8]}... [{self.snapshot_interval}s]: {prediction:.3f} [{state}]")
            
        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {e}")
    
    async def run(self):
        self.logger.info(f"RichardML started with {self.snapshot_interval}s intervals")
        
        while True:
            try:
                await self.scan_tokens()
                self.tokens_processed += 1
                
                confirmed_pumps = len(self.pump_memory["tokens"]) + sum(1 for token in self.memory["tokens"].values() 
                                    if token.get("pump_status") == "confirmed_pump")
                
                if not self.is_trained and confirmed_pumps >= 5:
                    self.logger.info("Starting initial training with confirmed pumps...")
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(self.executor, self.train_model_sync)
                elif self.is_trained and self.tokens_processed % 50 == 0:
                    self.logger.info("Retraining model with new data...")
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(self.executor, self.train_model_sync)
                
                if self.tokens_processed % 3 == 0:
                    self.prune_memory()
                    self.save_memory()
                    self.save_pump_memory()
                
                if self.tokens_processed % 20 == 0:
                    monitoring_tokens = sum(1 for state in self.memory["token_states"].values() if state == "monitoring")
                    active_tokens = sum(1 for state in self.memory["token_states"].values() if state == "active")
                    self.logger.info(f"Processed {len(self.memory['tokens'])} tokens, {monitoring_tokens} monitoring, {active_tokens} active, {confirmed_pumps} confirmed pumps")
                
                await asyncio.sleep(0.5)
                
            except KeyboardInterrupt:
                self.logger.info("RichardML stopped by user")
                break
            except Exception as e:
                self.logger.error(f"Main loop error: {e}")
                await asyncio.sleep(5)

if __name__ == "__main__":
    print("richardML.py must be imported!")
    exit(1)