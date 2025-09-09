#!/usr/bin/env python3
import json,os,time,logging,asyncio,numpy as np,lightgbm as lgb,joblib,glob,concurrent.futures,gc,weakref
from datetime import datetime
from pathlib import Path
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.feature_selection import SelectFromModel
from collections import deque
from logging.handlers import RotatingFileHandler
from shared_file_lock import SharedFileLock

class RichardML:
    """
    ULTRA-FAST 5-SECOND CANDLE FOCUSED MODEL
    
    Core Concept: Sub-minute decision making for rapid pump token lifecycle
    The model focuses almost exclusively on 5-second candle data for immediate decision-making,
    using 1-minute data only for high-level context.
    
    Enhanced Ultra-Fast States (optimized for 5-second timeframes with broader detection windows):
    0: Quiescent (0-15 seconds) - Token brand new/inactive, tight price range, minimal volume over 1-3 candles
    1: Ignition (1-3 minutes) - Initial pump phase with significant volume and price jump (PRIMARY ENTRY)
    2: Acceleration (30-90 seconds) - Main pump phase with consecutive large green 5s candles (HOLD PHASE)
    3: Peak Distribution (exhaustion pattern) - Multiple candles with long upper wicks or volume divergence (SELL SIGNAL)
    4: Reversal (The Dump) - First large red 5s candle erasing previous gains (URGENT EXIT)
    """
    def __init__(self,snapshot_interval=5,monitoring_period_seconds=60,max_duds=200,max_memory_tokens=250,max_pump_memory_tokens=100):
        self.setup_logging()
        self.snapshot_interval=snapshot_interval
        self.monitoring_period_seconds=monitoring_period_seconds
        self.max_duds=max_duds
        self.max_memory_tokens=max_memory_tokens
        self.max_pump_memory_tokens=max_pump_memory_tokens
        self.memory_file="richard_memory.json"
        self.pump_memory_file="richard_pump_memory.json"
        self.model_file="richard_lgb_model.pkl"
        self.scaler_file="scaler.pkl"
        self.contracts_file="contract_addresses.json"
        self.model=None
        self.scaler=StandardScaler()
        self.is_trained=False
        self.optimal_threshold=0.3  # Default threshold for imbalanced data
        self.tokens_processed=0
        self.executor=concurrent.futures.ThreadPoolExecutor(max_workers=2)  # Keep ThreadPoolExecutor to avoid pickle issues
        self.memory=self.load_memory()
        self.pump_memory=self.load_pump_memory()
        self.last_snapshot_time=self.memory.get("last_snapshot_time",{})
        self.token_order=deque()
        self.token_file_state = {}  # Track file modification times and processed candles
        self.migrate_pump_data()
        self.active_contracts=set()
        self.load_model()
        self.load_scaler()
        self.update_active_contracts()
        self._initialize_token_order()
        # Use weak references for circular reference prevention
        self._weak_refs = weakref.WeakValueDictionary()
        
        self.logger.info(f"RichardML initialized with {snapshot_interval}s intervals, {monitoring_period_seconds}s monitoring, max {max_memory_tokens} tokens in memory, max {max_pump_memory_tokens} pump tokens")

        # Add trading models
        self.trade_model_file = "trade_model.pkl"
        self.trade_scaler_file = "trade_scaler.pkl"
        self.trade_model = None
        self.trade_scaler = StandardScaler()
        self.trade_is_trained = False
        
        # Parameter adjustment system
        self.parameter_recommendations = {}
        self.last_parameter_analysis = 0
        self.parameter_analysis_interval = 75  # Analyze every 75 tokens processed
        
        self.load_trade_model()

    def log_trade_outcome(self, trade_data, ohlcv_data):
        """
        Log complete trade outcome with timing score analysis for learning.
        Performs offline hindsight analysis to generate timing scores.
        """
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Extract OHLCV data
            candles_1m = ohlcv_data.get('ohlcv', [])
            candles_5s = ohlcv_data.get('ohlcv_5s', [])
            
            # Calculate exit features using the same method as entry features
            exit_features = {}
            if candles_5s and len(candles_5s) >= 12:
                exit_features_5s = self._calculate_volume_features(candles_5s, "5s")
                exit_features_1m = self._calculate_volume_features(candles_1m, "1m")
                exit_features.update({f"exit_{k}": v for k, v in exit_features_5s.items()})
                exit_features.update({f"exit_{k}": v for k, v in exit_features_1m.items()})
            
            # Perform offline hindsight analysis for timing score
            timing_score = self._calculate_timing_score(trade_data, candles_1m, candles_5s)
            
            # Combine trade data with exit features and timing score
            complete_trade_data = {
                **trade_data,
                **exit_features,
                "exit_timestamp": timestamp,
                "timing_score": timing_score,
                "state_sequence": self._extract_state_sequence(trade_data, candles_1m, candles_5s)
            }
            
            # Write to trade_learning.json with rolling limit
            self._write_with_rolling_limit('trade_learning.json', json.dumps(complete_trade_data), 100)
            
            self.logger.info(f"Logged trade outcome for {trade_data.get('mint', 'unknown')[:20]}... (timing_score: {timing_score:.2f})")
            
        except Exception as e:
            self.logger.error(f"Failed to log trade outcome: {e}")
            import traceback
            self.logger.error(f"Trade outcome logging traceback: {traceback.format_exc()}")
    
    def _calculate_timing_score(self, trade_data, candles_1m, candles_5s):
        """
        Calculate timing score based on how close the sell was to the actual peak.
        +1.0: Sold within 5% of actual peak (Perfect)
        -1.0: Sold early, price rose another 50% (Too Early)
        -0.5: Sold late, price already dropped 20% from peak (Too Late)
        """
        try:
            entry_price = trade_data.get('entry_price', 0)
            exit_price = trade_data.get('exit_price', 0)
            
            if entry_price <= 0 or exit_price <= 0:
                return 0.0
            
            # Find actual peak price from all available data
            all_prices = []
            if candles_1m:
                all_prices.extend([float(c['high_usd']) for c in candles_1m])
            if candles_5s:
                all_prices.extend([float(c['high_usd']) for c in candles_5s])
            
            if not all_prices:
                return 0.0
            
            actual_peak = max(all_prices)
            
            # Calculate timing score
            peak_distance_percent = abs(exit_price - actual_peak) / actual_peak * 100
            
            # Perfect timing: within 5% of peak
            if peak_distance_percent <= 5:
                return 1.0
            
            # Too early: sold and price continued up significantly
            if exit_price < actual_peak:
                additional_rise = (actual_peak - exit_price) / exit_price * 100
                if additional_rise >= 50:
                    return -1.0
                elif additional_rise >= 25:
                    return -0.5
                else:
                    return 0.2  # Slightly early but reasonable
            
            # Too late: sold after significant drop from peak
            else:
                drop_from_peak = (actual_peak - exit_price) / actual_peak * 100
                if drop_from_peak >= 20:
                    return -0.5
                elif drop_from_peak >= 10:
                    return 0.0
                else:
                    return 0.5  # Good timing, near peak
            
        except Exception as e:
            self.logger.error(f"Error calculating timing score: {e}")
            return 0.0
    
    def _extract_state_sequence(self, trade_data, candles_1m, candles_5s):
        """Extract the sequence of states during the trade for analysis"""
        try:
            mint = trade_data.get('mint')
            if not mint or mint not in self.memory["tokens"]:
                return []
            
            token_data = self.memory["tokens"][mint]
            state_history = token_data.get("state_history", [])
            
            # Return recent state transitions
            return state_history[-10:] if len(state_history) > 10 else state_history
            
        except Exception as e:
            self.logger.error(f"Error extracting state sequence: {e}")
            return []

    def _calculate_volume_features(self, candles, timeframe):
        """Calculate volume features - moved from analyze.py"""
        # For 1m timeframe, require at least 1 candle
        # For 5s timeframe, require at least 12 candles (1 minute worth)
        min_candles = 12 if timeframe == '5s' else 1
        if len(candles) < min_candles:
            return {}
        
        features = {}
        current = candles[-1]
        features[f'{timeframe}_current_volume'] = float(current['volume_usd'])
        
        # Volume analysis only
        if len(candles) >= 1:
            # Use up to 3 candles, but work with whatever data is available
            recent_candles = candles[-min(3, len(candles)):]
            prices = [float(c['close_usd']) for c in recent_candles]
            if prices[0] > 0 and len(prices) > 1:
                features[f'{timeframe}_price_momentum_3'] = ((prices[-1] - prices[0]) / prices[0]) * 100
            
            volumes = [float(c['volume_usd']) for c in recent_candles]
            if len(volumes) > 1:
                features[f'{timeframe}_volume_trend'] = volumes[-1] - volumes[0]
            features[f'{timeframe}_avg_volume_3'] = sum(volumes) / len(volumes)
        
        if len(candles) >= 5:
            recent_candles = candles[-5:]
            volumes = [float(c['volume_usd']) for c in recent_candles]
            if len(volumes) >= 3:
                vol_accel = (volumes[-1] - volumes[-2]) - (volumes[-2] - volumes[-3])
                features[f'{timeframe}_volume_acceleration'] = vol_accel
        
        features[f'{timeframe}_buy_count'] = current.get('buy_count', 0)
        features[f'{timeframe}_sell_count'] = current.get('sell_count', 0)
        sell_count = max(current.get('sell_count', 1), 1)
        features[f'{timeframe}_buy_sell_ratio'] = current.get('buy_count', 0) / sell_count
        
        return features

    def _write_with_rolling_limit(self, filepath, data, max_lines):
        """Write data with rolling limit - moved from analyze.py"""
        try:
            # Use append-only approach with periodic cleanup
            with open(filepath, 'a') as f:
                f.write(data.rstrip() + '\n')
            
            # Periodically clean up old entries (every 100 writes)
            if not hasattr(self, '_cleanup_counter'):
                self._cleanup_counter = {}
            
            if filepath not in self._cleanup_counter:
                self._cleanup_counter[filepath] = 0
            
            self._cleanup_counter[filepath] += 1
                
            if self._cleanup_counter[filepath] % 100 == 0:
                self._cleanup_log_file(filepath, max_lines)
                
        except Exception as e:
            self.logger.error(f"ERROR: Writing to {filepath}: {e}")
    
    def _cleanup_log_file(self, filepath, max_lines):
        """Clean up log file to keep only recent entries"""
        try:
            with open(filepath, 'r') as f:
                lines = f.read().strip().split('\n')
            
            if len(lines) > max_lines:
                lines = lines[-max_lines:]
                with open(filepath, 'w') as f:
                    f.write('\n'.join(lines) + '\n')
        except:
            pass

    def load_trade_model(self):
        try:

            
            if os.path.exists(self.trade_model_file):
                if os.path.exists(self.trade_scaler_file):
                    try:
                        self.trade_model = joblib.load(self.trade_model_file)
                        self.trade_scaler = joblib.load(self.trade_scaler_file)
                        self.trade_is_trained = True
        
                    except Exception as e:
                        self.logger.error(f"ERROR: Failed to load trade model files: {e}")
                        self.trade_is_trained = False
                else:
                    self.logger.warning(f"ERROR: Trade scaler file not found: {self.trade_scaler_file}")
                    self.trade_is_trained = False
            else:

                self.trade_is_trained = False
        except Exception as e:
            self.logger.error(f"ERROR: Exception during trade model loading: {e}")
            self.trade_is_trained = False

    def save_trade_model(self):
        try:
            if self.trade_model is not None:

                joblib.dump(self.trade_model, self.trade_model_file)
                joblib.dump(self.trade_scaler, self.trade_scaler_file)
                self.logger.info("DEBUG: Trade model and scaler saved successfully")
            else:
                self.logger.warning("ERROR: Cannot save trade model - model is None")
        except Exception as e:
            self.logger.error(f"ERROR: Failed to save trade model: {e}")
            import traceback
            self.logger.error(f"ERROR: Trade model save traceback: {traceback.format_exc()}")

    def prepare_trade_training_data(self):
        """Prepare trade training data with unified logic for timing scores and profitability"""
        try:
            X, y, sw = [], [], []
            timing_score_samples = 0
            profitability_samples = 0
            
            if not os.path.exists('trade_learning.json'):
                return np.array([]), np.array([]), np.array([])
            
            with open('trade_learning.json', 'r') as f:
                for line in f:
                    try:
                        line = line.strip()
                        if not line or not line.startswith('{'):
                            continue
                        trade = json.loads(line)
                        mint = trade['mint']
                        
                        # Get features (either stored or extract from OHLCV)
                        features = self._get_trade_features(trade, mint)
                        if not features:
                            continue
                        
                        # Primary method: Use timing scores if available
                        if 'timing_score' in trade:
                            timing_score = trade['timing_score']
                            normalized_score = (timing_score + 1) / 2  # Convert to 0-1 scale
                            weight = 1.0 + abs(timing_score)  # Higher weight for extreme scores
                            timing_score_samples += 1
                        
                        # Fallback method: Use profitability-based scoring
                        elif 'pnl_percent' in trade:
                            pnl_percent = trade.get('pnl_percent', 0)
                            normalized_score = max(0, min(1, (pnl_percent + 20) / 40))  # -20% to +20% -> 0 to 1
                            weight = 1.0 + abs(pnl_percent) / 100
                            profitability_samples += 1
                        
                        else:
                            continue
                        
                        X.append(features)
                        y.append(normalized_score)
                        sw.append(weight)
                        
                    except (json.JSONDecodeError, Exception) as e:
                        continue
            
            total_samples = len(X)
            if total_samples < 10:
                self.logger.warning(f"Insufficient training data - Only {total_samples} samples (minimum 10 required)")
                return np.array([]), np.array([]), np.array([])
            
            method_used = "timing scores" if timing_score_samples > 0 else "profitability"
            self.logger.info(f"Prepared trade training data with {total_samples} samples using {method_used} "
                           f"(timing: {timing_score_samples}, profitability: {profitability_samples})")
            
            return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), np.array(sw, dtype=np.float32)
            
        except Exception as e:
            self.logger.error(f"Failed to prepare trade training data: {e}")
            return np.array([]), np.array([]), np.array([])
    
    def _get_trade_features(self, trade, mint):
        """Extract or load features for a trade"""
        # Use stored features if available (check for individual feature fields)
        feature_keys = [k for k in trade.keys() if k.startswith('exit_')]
        if feature_keys:
            # Extract features from the stored exit_ fields
            features = []
            feature_names = [
                'exit_5s_current_volume', 'exit_5s_price_momentum_3', 'exit_5s_volume_trend',
                'exit_5s_avg_volume_3', 'exit_5s_volume_acceleration', 'exit_5s_buy_count',
                'exit_5s_sell_count', 'exit_5s_buy_sell_ratio', 'exit_1m_current_volume',
                'exit_1m_price_momentum_3', 'exit_1m_volume_trend', 'exit_1m_avg_volume_3',
                'exit_1m_volume_acceleration', 'exit_1m_buy_count', 'exit_1m_sell_count',
                'exit_1m_buy_sell_ratio'
            ]
            
            for feature_name in feature_names:
                features.append(trade.get(feature_name, 0.0))
            
            return features
        
        # Check for legacy 'features' field
        if 'features' in trade and trade['features']:
            return trade['features']
        
        # Fallback: load OHLCV data and extract features
        candles_1m = []
        candles_5s = []
        
        try:
            if os.path.exists(f"tokenOHLCV/{mint}_ohlcv.json"):
                with open(f"tokenOHLCV/{mint}_ohlcv.json", 'r') as f1:
                    data_1m = json.load(f1)
                    candles_1m = data_1m.get("ohlcv", [])
            
            if os.path.exists(f"tokenOHLCV/{mint}_5s_ohlcv.json"):
                with open(f"tokenOHLCV/{mint}_5s_ohlcv.json", 'r') as f5:
                    data_5s = json.load(f5)
                    candles_5s = data_5s.get("ohlcv", [])
            
            if len(candles_1m) >= 3:
                return self.extract_features(candles_5s, candles_1m, mint)
        except Exception:
            pass
        
        return None
    
    def _study_memory_data(self, X, y):
        """Study stored memory data to learn trading patterns"""
        
        # Study main memory tokens
        for mint, token_data in self.memory["tokens"].items():
            candles_5s = token_data.get("candles_5s", [])
            candles_1m = token_data.get("candles_1m", [])
            pump_status = token_data.get("pump_status")
            
            if len(candles_1m) >= 3 and pump_status:
                features = self.extract_features(candles_5s, candles_1m, mint)
                if features:
                    # Generate synthetic PnL based on pump status and multiplier
                    pnl = self._estimate_pnl_from_pump_data(token_data, pump_status)
                    X.append(features)
                    y.append(pnl)
        
        # Study pump memory tokens
        for mint, token_data in self.pump_memory["tokens"].items():
            candles_5s = token_data.get("candles_5s", [])
            candles_1m = token_data.get("candles_1m", [])
            pump_status = token_data.get("pump_status")
            
            if len(candles_1m) >= 3 and pump_status:
                features = self.extract_features(candles_5s, candles_1m, mint)
                if features:
                    # Generate synthetic PnL based on pump status and multiplier
                    pnl = self._estimate_pnl_from_pump_data(token_data, pump_status)
                    X.append(features)
                    y.append(pnl)

    def _estimate_pnl_from_pump_data(self, token_data, pump_status):
        """Estimate PnL percentage based on pump status and multiplier"""
        
        if pump_status == "confirmed_pump":
            multiplier = token_data.get("pump_multiplier", 1.0)
            # Estimate positive PnL based on pump multiplier
            return min(100, (multiplier - 1) * 100 * 0.8)  # Conservative estimate
        
        elif pump_status == "pump_and_dump":
            multiplier = token_data.get("pump_multiplier", 1.0)
            # Mixed result - initial gains but eventual loss
            return max(-50, (multiplier - 1) * 100 * 0.3 - 20)
        
        elif pump_status == "dud":
            # Estimate small loss or no gain
            return np.random.normal(-5, 10)  # Small losses with variance
        
        elif pump_status == "insufficient_data":
            # Neutral to slightly negative
            return np.random.normal(-2, 5)
        
        else:
            return 0
    
    def train_trade_model_sync(self):
        """Enhanced trade model training with validation, regularization, and k-fold CV"""
        try:
            X, y, sw = self.prepare_trade_training_data()
            if len(X) < 10:
                self.logger.warning(f"Not enough trade data for training: {len(X)} samples (minimum 10 required)")
                return False
            
            # Critical data validation and cleaning
            X, y, sw = self._validate_and_clean_trade_data(X, y, sw)
            if len(X) < 10:
                self.logger.warning("Insufficient trade data after validation and cleaning")
                return False
            
            # K-fold cross-validation for trade model
            cv_scores = self._perform_trade_kfold_validation(X, y, sw, k=5)
            if cv_scores:
                self.logger.info(f"Trade model K-fold CV RMSE: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")
            
            # Split data
            Xt, Xv, yt, yv, wt, wv = train_test_split(X, y, sw, test_size=0.2, random_state=42)
            
            # Scale features
            Xts = self.trade_scaler.fit_transform(Xt)
            Xvs = self.trade_scaler.transform(Xv)
            
            # Enhanced model parameters with stronger regularization
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 15,
                'learning_rate': 0.05,
                'verbose': -1,
                # Enhanced regularization to reduce overfitting
                'reg_alpha': 0.3,      # L1 regularization (Lasso)
                'reg_lambda': 0.3,     # L2 regularization (Ridge)
                'min_gain_to_split': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.7,
                'min_child_samples': 10,
                'max_depth': 6,
            }
            
            # Create datasets
            td = lgb.Dataset(Xts, label=yt, weight=wt)
            vd = lgb.Dataset(Xvs, label=yv, weight=wv, reference=td)
            
            # Train model
            self.trade_model = lgb.train(params, td, num_boost_round=50, valid_sets=[vd], 
                                        callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)])
            
            # Evaluate model
            val_predictions = self.trade_model.predict(Xvs)
            rmse = np.sqrt(np.mean((val_predictions - yv) ** 2))
            mae = np.mean(np.abs(val_predictions - yv))
            
            self.logger.info(f"Trade model validation RMSE: {rmse:.4f}, MAE: {mae:.4f}")
            
            self.trade_is_trained = True
            self.save_trade_model()
            return True
            
        except Exception as e:
            self.logger.error(f"ERROR: Failed to train trade model: {e}")
            import traceback
            self.logger.error(f"ERROR: Trade model training traceback: {traceback.format_exc()}")
            return False
    
    def _validate_and_clean_trade_data(self, X, y, sw):
        """Validate and clean trade training data"""
        try:
            # Convert to numpy arrays
            X = np.asarray(X, dtype=np.float32)
            y = np.asarray(y, dtype=np.float32)
            sw = np.asarray(sw, dtype=np.float32)
            
            # Remove samples with invalid features
            valid_mask = np.all(np.isfinite(X), axis=1) & np.isfinite(y) & np.isfinite(sw)
            if not np.all(valid_mask):
                invalid_count = np.sum(~valid_mask)
                self.logger.warning(f"Removing {invalid_count} trade samples with invalid data")
                X, y, sw = X[valid_mask], y[valid_mask], sw[valid_mask]
            
            # Remove extreme outliers in target variable
            y_q1, y_q3 = np.percentile(y, [25, 75])
            y_iqr = y_q3 - y_q1
            y_lower = y_q1 - 3 * y_iqr
            y_upper = y_q3 + 3 * y_iqr
            
            outlier_mask = (y >= y_lower) & (y <= y_upper)
            if not np.all(outlier_mask):
                outlier_count = np.sum(~outlier_mask)
                self.logger.info(f"Removing {outlier_count} extreme outliers from trade data")
                X, y, sw = X[outlier_mask], y[outlier_mask], sw[outlier_mask]
            
            self.logger.info(f"Trade data validation complete: {len(X)} clean samples")
            return X, y, sw
            
        except Exception as e:
            self.logger.error(f"Error in trade data validation: {e}")
            return np.array([]), np.array([]), np.array([])
    
    def _perform_trade_kfold_validation(self, X, y, sw, k=5):
        """Perform k-fold cross-validation for trade model"""
        try:
            kf = KFold(n_splits=k, shuffle=True, random_state=42)
            cv_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
                try:
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    w_train, w_val = sw[train_idx], sw[val_idx]
                    
                    # Scale features for this fold
                    fold_scaler = StandardScaler()
                    X_train_scaled = fold_scaler.fit_transform(X_train)
                    X_val_scaled = fold_scaler.transform(X_val)
                    
                    # Train model for this fold
                    params = {
                        'objective': 'regression',
                        'metric': 'rmse',
                        'boosting_type': 'gbdt',
                        'num_leaves': 10,
                        'learning_rate': 0.1,
                        'verbose': -1,
                        'reg_alpha': 0.3,
                        'reg_lambda': 0.3,
                    }
                    
                    train_data = lgb.Dataset(X_train_scaled, label=y_train, weight=w_train)
                    fold_model = lgb.train(params, train_data, num_boost_round=30, callbacks=[lgb.log_evaluation(0)])
                    
                    # Evaluate on validation set
                    val_pred = fold_model.predict(X_val_scaled)
                    fold_rmse = np.sqrt(np.mean((val_pred - y_val) ** 2))
                    cv_scores.append(fold_rmse)
                    
                except Exception as e:
                    self.logger.error(f"Error in trade fold {fold + 1}: {e}")
                    continue
            
            return cv_scores
            
        except Exception as e:
            self.logger.error(f"Error in trade k-fold validation: {e}")
            return []
    
    async def predict_trade_action(self, mint, ohlcv_data):
        """
        State-driven trading logic based on Pump Model's real-time state assessment.
        
        Entry (BUY): State 0 -> State 1 transition with high confidence
        Hold: State 2 (Momentum Spike) - strongest pump phase
        Exit (SELL): State 3 (Peak Distribution) or State 4 (Downfall)
        """
        try:
            if not self.is_trained:
                return None
            
            # Get current state prediction from Pump Model
            state_prediction = await self.predict(mint, ohlcv_data)
            if not state_prediction:
                return None
            
            predicted_state = state_prediction['predicted_state']
            confidence = state_prediction['confidence']
            state_probabilities = state_prediction['state_probabilities']
            state_name = state_prediction['state_name']
            
            # Get features for trade model if available
            candles_1m = ohlcv_data.get('ohlcv', [])
            candles_5s = ohlcv_data.get('ohlcv_5s', [])
            features = self.extract_features(candles_5s, candles_1m, mint) if candles_5s else None
            
            # State-driven trading logic
            action_result = None
            
            # Entry Condition: State 1 (Early Rise) with high confidence
            if predicted_state == 1 and confidence > 0.7:
                action_result = {
                    "action": "BUY", 
                    "confidence": confidence, 
                    "reason": f"early_rise_detected_{confidence:.2f}",
                    "state": predicted_state,
                    "state_name": state_name,
                    "features": features
                }
            
            # Holding Condition: State 2 (Momentum Spike)
            elif predicted_state == 2:
                action_result = {
                    "action": "HOLD", 
                    "confidence": confidence, 
                    "reason": f"momentum_spike_{confidence:.2f}",
                    "state": predicted_state,
                    "state_name": state_name,
                    "features": features
                }
            
            # Exit Conditions
            elif predicted_state == 3 and confidence > 0.6:  # Peak Distribution
                action_result = {
                    "action": "SELL", 
                    "confidence": confidence, 
                    "reason": f"peak_distribution_{confidence:.2f}",
                    "state": predicted_state,
                    "state_name": state_name,
                    "features": features
                }
            
            elif predicted_state == 4:  # Downfall - immediate exit
                action_result = {
                    "action": "SELL", 
                    "confidence": confidence, 
                    "reason": f"downfall_detected_{confidence:.2f}",
                    "state": predicted_state,
                    "state_name": state_name,
                    "features": features
                }
            
            # Pump fizzled - exit if reverting from higher states
            elif predicted_state in [0, 1] and state_probabilities[2] < 0.3:  # Low momentum spike probability
                action_result = {
                    "action": "SELL", 
                    "confidence": 1 - state_probabilities[2], 
                    "reason": f"pump_fizzled_state_{predicted_state}",
                    "state": predicted_state,
                    "state_name": state_name,
                    "features": features
                }
            
            # Default: Hold/Wait
            else:
                action_result = {
                    "action": "HOLD", 
                    "confidence": confidence, 
                    "reason": f"state_{predicted_state}_wait",
                    "state": predicted_state,
                    "state_name": state_name,
                    "features": features
                }
            
            return action_result
                
        except Exception as e:
            self.logger.error(f"Trade prediction error for {mint}: {e}")
            return None
    
    def setup_logging(self):
        # Custom handler to maintain max 1000 lines
        class LineCountRotatingHandler(logging.FileHandler):
            def __init__(self, filename, max_lines=1000):
                super().__init__(filename, mode='a')
                self.max_lines = max_lines
                
            def emit(self, record):
                super().emit(record)
                try:
                    with open(self.baseFilename, 'r') as f:
                        lines = f.readlines()
                    if len(lines) > self.max_lines:
                        with open(self.baseFilename, 'w') as f:
                            f.writelines(lines[-self.max_lines:])
                except:
                    pass
        
        file_handler = LineCountRotatingHandler('richard.log', max_lines=1000)
        console_handler = logging.StreamHandler()
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[file_handler, console_handler]
        )
        self.logger = logging.getLogger('RichardML')
    
    def _write_prediction_log(self, pump_prob, threshold, mint, token_state, candles_count):
        """Write prediction log with rolling limit of 1000 lines"""
        log_filename = f"predictions_{self.snapshot_interval}s.log"
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "mint": mint,
            "pump_prob": pump_prob,
            "threshold": threshold,
            "prediction": "FUTURE_PUMP" if pump_prob > threshold else "FUTURE_DUD",
            "state": token_state,
            "candles_count": candles_count
        }
        
        # Write the log entry
        with open(log_filename, 'a') as log_file:
            log_file.write(f"{json.dumps(log_entry)}\n")
        
        # Maintain max 1000 lines
        try:
            with open(log_filename, 'r') as f:
                lines = f.readlines()
            if len(lines) > 1000:
                with open(log_filename, 'w') as f:
                    f.writelines(lines[-1000:])
        except:
            pass
        
    def load_memory(self):
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file,'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading memory file: {e}")
        return {"tokens":{},"token_states":{},"last_snapshot_time":{}}
    
    def load_pump_memory(self):
        if os.path.exists(self.pump_memory_file):
            try:
                with open(self.pump_memory_file,'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading pump memory file: {e}")
        return {"tokens":{},"token_states":{},"last_snapshot_time":{}}
    
    def migrate_pump_data(self):
        pumps_to_migrate={mint:data for mint,data in self.memory["tokens"].items() if data.get("pump_status")=="confirmed_pump"}
        for mint,data in pumps_to_migrate.items():
            self.pump_memory["tokens"][mint]=data
            if mint in self.memory["token_states"]:self.pump_memory["token_states"][mint]=self.memory["token_states"][mint]
            del self.memory["tokens"][mint]
            if mint in self.memory["token_states"]:del self.memory["token_states"][mint]
        if pumps_to_migrate:
            # Manage pump memory size limit after migration
            self._manage_rolling_pump_memory()
            self.logger.info(f"Migrated {len(pumps_to_migrate)} confirmed pumps to pump memory")
            self.save_memory()
            self.save_pump_memory()
    
    def save_memory(self):
        self.memory["last_snapshot_time"] = self.last_snapshot_time
        # Use a temporary file for atomic write
        temp_file = self.memory_file + ".tmp"
        try:
            # Stream write JSON to reduce memory usage
            with open(temp_file, 'w') as f:
                f.write('{\n')
                
                # Write tokens section
                f.write('  "tokens": {\n')
                token_items = list(self.memory["tokens"].items())
                for i, (mint, token_data) in enumerate(token_items):
                    f.write(f'    "{mint}": ')
                    serializable_token = self._convert_to_json_serializable(token_data)
                    json.dump(serializable_token, f, separators=(',', ':'))
                    if i < len(token_items) - 1:
                        f.write(',')
                    f.write('\n')
                f.write('  },\n')
                
                # Write token_states section
                f.write('  "token_states": {\n')
                state_items = list(self.memory["token_states"].items())
                for i, (mint, state) in enumerate(state_items):
                    f.write(f'    "{mint}": ')
                    json.dump(state, f, separators=(',', ':'))
                    if i < len(state_items) - 1:
                        f.write(',')
                    f.write('\n')
                f.write('  },\n')
                
                # Write last_snapshot_time section
                f.write('  "last_snapshot_time": ')
                serializable_snapshot_time = self._convert_to_json_serializable(self.last_snapshot_time)
                json.dump(serializable_snapshot_time, f, separators=(',', ':'))
                f.write('\n')
                
                f.write('}\n')
            
            os.replace(temp_file, self.memory_file) # Atomically replace the old file
        except Exception as e:
            self.logger.error(f"Error saving memory to {self.memory_file}: {e}")
            if os.path.exists(temp_file):
                os.remove(temp_file) # Clean up temp file if error occurs
    
    def save_pump_memory(self):
        # Use a temporary file for atomic write
        temp_file = self.pump_memory_file + ".tmp"
        try:
            # Stream write JSON to reduce memory usage
            with open(temp_file, 'w') as f:
                f.write('{\n')
                
                # Write tokens section
                f.write('  "tokens": {\n')
                token_items = list(self.pump_memory["tokens"].items())
                for i, (mint, token_data) in enumerate(token_items):
                    f.write(f'    "{mint}": ')
                    serializable_token = self._convert_to_json_serializable(token_data)
                    json.dump(serializable_token, f, separators=(',', ':'))
                    if i < len(token_items) - 1:
                        f.write(',')
                    f.write('\n')
                f.write('  },\n')
                
                # Write token_states section
                f.write('  "token_states": {\n')
                state_items = list(self.pump_memory["token_states"].items())
                for i, (mint, state) in enumerate(state_items):
                    f.write(f'    "{mint}": ')
                    json.dump(state, f, separators=(',', ':'))
                    if i < len(state_items) - 1:
                        f.write(',')
                    f.write('\n')
                f.write('  },\n')
                
                # Write last_snapshot_time section
                f.write('  "last_snapshot_time": ')
                serializable_snapshot_time = self._convert_to_json_serializable(self.pump_memory.get("last_snapshot_time", {}))
                json.dump(serializable_snapshot_time, f, separators=(',', ':'))
                f.write('\n')
                
                f.write('}\n')
            
            os.replace(temp_file, self.pump_memory_file) # Atomically replace the old file
        except Exception as e:
            self.logger.error(f"Error saving pump memory to {self.pump_memory_file}: {e}")
            if os.path.exists(temp_file):
                os.remove(temp_file) # Clean up temp file if error occurs

    def _convert_to_json_serializable(self, data):
        """Recursively convert numpy types to JSON-serializable Python types"""
        if isinstance(data, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._convert_to_json_serializable(item) for item in data]
        elif isinstance(data, (np.integer, np.floating)):
            return data.item()  # Convert numpy scalar to Python scalar
        elif isinstance(data, np.ndarray):
            return data.tolist()  # Convert numpy array to Python list
        elif isinstance(data, np.bool_):
            return bool(data)
        else:
            return data
    
    def load_model(self):
        if os.path.exists(self.model_file):
            try:
                self.model = joblib.load(self.model_file)
                self.is_trained = True
                self.logger.info(f"Real-time state classification model loaded")
            except Exception as e:
                self.logger.error(f"Failed to load model: {e}")
    
    def load_scaler(self):
        if os.path.exists(self.scaler_file):
            try:self.scaler=joblib.load(self.scaler_file)
            except Exception as e:self.logger.error(f"Failed to load scaler: {e}")
    
    def save_model(self):
        if self.model is not None:
            joblib.dump(self.model, self.model_file)
            joblib.dump(self.scaler, self.scaler_file)
    
    def update_active_contracts(self, use_async=False):
        """Load active contracts with optional async locking"""
        if use_async:
            return self._update_active_contracts_async()
        else:
            return self._update_active_contracts_sync()
    
    def _update_active_contracts_sync(self):
        """Synchronous version for __init__"""
        try:
            if os.path.exists(self.contracts_file):
                with open(self.contracts_file,'r') as f:
                    cd=json.load(f)
                    if isinstance(cd,list):self.active_contracts=set(item.get("mint_address") for item in cd if item.get("mint_address"))
                    elif isinstance(cd,dict):self.active_contracts=set(cd.keys())
                    else:self.active_contracts=set()
            else:self.active_contracts=set()
        except Exception as e:
            self.logger.error(f"Error loading contracts: {e}")
            self.active_contracts=set()
    
    async def _update_active_contracts_async(self):
        """Async version with proper locking"""
        try:
            if os.path.exists(self.contracts_file):
                async with SharedFileLock("contract_addresses.json.lock"):
                    with open(self.contracts_file,'r') as f:
                        cd=json.load(f)
                        if isinstance(cd,list):self.active_contracts=set(item.get("mint_address") for item in cd if item.get("mint_address"))
                        elif isinstance(cd,dict):self.active_contracts=set(cd.keys())
                        else:self.active_contracts=set()
            else:self.active_contracts=set()
        except Exception as e:
            self.logger.error(f"Error loading contracts: {e}")
            self.active_contracts=set()
    
    def cleanup_inactive_tokens(self):
        if not self.active_contracts:
            self.logger.info("No active contracts defined - marking all tokens as inactive")
            for mint in list(self.memory["tokens"].keys()):
                if self.memory["token_states"].get(mint)!="inactive":self.memory["token_states"][mint]="inactive"
                if self.memory["tokens"][mint].get("pump_status") is None:self.memory["tokens"][mint]["pump_status"]="dud"
            for mint in list(self.pump_memory["tokens"].keys()):
                if self.pump_memory["token_states"].get(mint)!="inactive":self.pump_memory["token_states"][mint]="inactive"
            return
        ttmi=[]
        for mint in list(self.memory["tokens"].keys()):
            if mint not in self.active_contracts and self.memory["token_states"].get(mint) not in ["inactive","dead"]:ttmi.append(mint)
        for mint in ttmi:
            self.memory["token_states"][mint]="inactive"
            if self.memory["tokens"][mint].get("pump_status") is None:self.memory["tokens"][mint]["pump_status"]="dud"
        pttmi=[]
        for mint in list(self.pump_memory["tokens"].keys()):
            if mint not in self.active_contracts and self.pump_memory["token_states"].get(mint) not in ["inactive","dead"]:pttmi.append(mint)
        for mint in pttmi:self.pump_memory["token_states"][mint]="inactive"
        tm=len(ttmi)+len(pttmi)
        if tm>0:self.logger.info(f"Marked {tm} tokens as inactive ({len(ttmi)} main, {len(pttmi)} pump)")

    def _pump_memory_data_generator(self):
        """Generator for pump memory data"""
        for mint, token_data in self.pump_memory["tokens"].items():
            yield mint, token_data

    def _memory_data_generator(self, data_dict):
        """Generator to yield memory data one item at a time"""
        for mint, token_data in data_dict.items():
            yield mint, token_data
    
    def prune_memory(self):
        dm={mint for mint,data in self.memory["tokens"].items() if data.get("pump_status") in ["dud","pump_fake","pump_and_dump"]}
        if len(dm)>self.max_duds:
            dtr=sorted(list(dm),key=lambda m:self.memory["tokens"][m].get("first_seen",""))
            ntr=len(dtr)-self.max_duds
            mtr=dtr[:ntr]
            for mint in mtr:
                del self.memory["tokens"][mint]
                if mint in self.memory["token_states"]:del self.memory["token_states"][mint]
                if mint in self.last_snapshot_time:del self.last_snapshot_time[mint]
            self.logger.info(f"Pruned {len(mtr)} old dud tokens from memory.")
    
    
    def should_take_snapshot(self, mint, current_time):
        """Now based on OHLCV data availability rather than time intervals"""
        return True  # Always process when OHLCV data is available
    
    def classify_token_state(self, candles_1m, candles_5s):
        """Classify token state based on both 1-minute and 5-second OHLCV data"""
        if len(candles_1m) < 1:
            return "new"
        
        # Use both 1m and 5s data to determine time elapsed
        first_candle = candles_1m[0]["timestamp"]
        last_candle = candles_1m[-1]["timestamp"]
        time_elapsed = last_candle - first_candle
        
        # If we have 5s data, use it for more precise timing
        if candles_5s and len(candles_5s) > 0:
            first_5s = candles_5s[0]["timestamp"]
            last_5s = candles_5s[-1]["timestamp"]
            time_elapsed_5s = last_5s - first_5s
            # Use the longer time period for classification
            time_elapsed = max(time_elapsed, time_elapsed_5s)
        
        # Reduce monitoring period requirement since we can detect pumps earlier
        reduced_monitoring_period = max(30, self.monitoring_period_seconds // 2)
        
        if time_elapsed < reduced_monitoring_period:
            return "monitoring"
        
        # Check recent volume activity from both 1m and 5s candles
        recent_volumes_1m = [float(c["volume_usd"]) for c in candles_1m[-min(5, len(candles_1m)):]]
        avg_volume_1m = np.mean(recent_volumes_1m)
        
        # If we have 5s data, also check that
        if candles_5s and len(candles_5s) >= 5:
            recent_count = min(10, len(candles_5s))
            recent_volumes_5s = [float(c["volume_usd"]) for c in candles_5s[-recent_count:]]
            avg_volume_5s = np.mean(recent_volumes_5s)
            combined_volume = (avg_volume_1m + avg_volume_5s) / 2
        else:
            combined_volume = avg_volume_1m
        
        if combined_volume < 100:
            return "inactive"
        elif len(candles_1m) > 50 or time_elapsed > 3000:  # 50 minutes
            return "mature"
        else:
            return "active"
    
    def checkForPump(self, candles_1m, candles_5s):
        """Check for pump using BOTH 1-minute and 5-second OHLCV candlestick data with validation"""
        try:
            # Critical data validation
            candles_1m = self._validate_candle_data(candles_1m, "1m_pump_check") if candles_1m else []
            candles_5s = self._validate_candle_data(candles_5s, "5s_pump_check") if candles_5s else []
            
            # Use whatever 1-minute data we have (minimum 1 candle)
            if len(candles_1m) < 1:
                return False, "insufficient_data"
            
            # If we have very few 1-minute candles, rely more heavily on 5-second data
            if len(candles_1m) < 3 and (not candles_5s or len(candles_5s) < 6):
                return False, "insufficient_data"
            
            # Analyze 1-minute candles for overall trend (use what we have)
            prices_1m = self._extract_validated_array(candles_1m, "close_usd", "pump_prices_1m")
            volumes_1m = self._extract_validated_array(candles_1m, "volume_usd", "pump_volumes_1m")
            
            if len(prices_1m) == 0 or len(volumes_1m) == 0:
                return False, "invalid_data"
            
            # Adapt baseline calculation based on available data
            if len(prices_1m) >= 3:
                baseline_price_1m = np.mean(prices_1m[:3])
            elif len(prices_1m) >= 2:
                baseline_price_1m = np.mean(prices_1m[:2])
            else:
                baseline_price_1m = prices_1m[0]
        
            if baseline_price_1m <= 1e-9:
                return False, "invalid_baseline"
            
            max_price_1m = np.max(prices_1m)
            price_multiplier_1m = max_price_1m / baseline_price_1m
            
            # Adapt volume calculation based on available data
            if len(volumes_1m) >= 3:
                baseline_volume_1m = np.mean(volumes_1m[:3])
            elif len(volumes_1m) >= 2:
                baseline_volume_1m = np.mean(volumes_1m[:2])
            else:
                baseline_volume_1m = volumes_1m[0]
            
            max_volume_1m = np.max(volumes_1m)
            volume_surge_1m = max_volume_1m / (baseline_volume_1m + 1e-9)
        
            # Analyze 5-second candles for detailed pump detection if available
            price_multiplier_5s = price_multiplier_1m  # Default to 1m data
            volume_surge_5s = volume_surge_1m
            
            if candles_5s and len(candles_5s) >= 6:  # Reduced from 15 to 6 (30 seconds of 5s data)
                prices_5s = self._extract_validated_array(candles_5s, "close_usd", "pump_prices_5s")
                volumes_5s = self._extract_validated_array(candles_5s, "volume_usd", "pump_volumes_5s")
                
                if len(prices_5s) >= 6 and len(volumes_5s) >= 6:
                    # Use first 3 candles as baseline (15 seconds)
                    baseline_count = min(3, len(prices_5s) // 2)
                    baseline_price_5s = np.mean(prices_5s[:baseline_count])
                    
                    if baseline_price_5s > 1e-9:
                        max_price_5s = np.max(prices_5s)
                        price_multiplier_5s = max_price_5s / baseline_price_5s
                        
                        baseline_volume_5s = np.mean(volumes_5s[:baseline_count])
                        max_volume_5s = np.max(volumes_5s)
                        volume_surge_5s = max_volume_5s / (baseline_volume_5s + 1e-9)
        
            # Use the higher sensitivity data (5s if available, otherwise 1m)
            price_multiplier = max(price_multiplier_1m, price_multiplier_5s)
            volume_surge = max(volume_surge_1m, volume_surge_5s)
            
            # Validate multipliers are finite
            price_multiplier = self._validate_numeric_input(price_multiplier, "price_multiplier")
            volume_surge = self._validate_numeric_input(volume_surge, "volume_surge")
            
            # Adjust pump detection criteria based on available data quality
            if len(candles_1m) >= 3:
                # Full criteria when we have enough 1-minute data
                condition1 = price_multiplier >= 1.75 and volume_surge >= 1.3
                condition2 = price_multiplier >= 1.4 and volume_surge >= 1.8
            else:
                # More sensitive criteria when relying on limited data and 5s candles
                condition1 = price_multiplier >= 1.5 and volume_surge >= 1.2
                condition2 = price_multiplier >= 1.3 and volume_surge >= 1.5
            
            if condition1 or condition2:
                return True, "confirmed_pump"
            elif price_multiplier >= 1.2 and volume_surge >= 1.3:
                return True, "pump_fake"
            else:
                return False, "dud"
                
        except Exception as e:
            self.logger.error(f"Critical error in checkForPump: {e}")
            return False, "error"
    
    def check_pump_and_dump(self, mint, current_price, candles_1m, candles_5s):
        """Check for pump and dump pattern using both 1m and 5s data with volume exhaustion signals"""
        try:
            # Critical data validation
            current_price = self._validate_numeric_input(current_price, "current_price")
            candles_1m = self._validate_candle_data(candles_1m, "1m_pump_dump") if candles_1m else []
            candles_5s = self._validate_candle_data(candles_5s, "5s_pump_dump") if candles_5s else []
            
            token_data = self.memory["tokens"].get(mint) or self.pump_memory["tokens"].get(mint)
            if not token_data or token_data.get("pump_status") == "pump_and_dump":
                return False
            
            if len(candles_1m) < 3:
                return False
            
            # Track peak price and peak volume from both 1m and 5s candles
            peak_price = self._validate_numeric_input(token_data.get("peak_price", 0), "peak_price")
            peak_volume = self._validate_numeric_input(token_data.get("peak_volume", 0), "peak_volume")
            
            # Get peak from 1-minute candles with validation
            highs_1m = self._extract_validated_array(candles_1m, "high_usd", "pump_dump_highs_1m")
            volumes_1m = self._extract_validated_array(candles_1m, "volume_usd", "pump_dump_volumes_1m")
            
            if len(highs_1m) == 0 or len(volumes_1m) == 0:
                return False
            
            current_peak_1m = np.max(highs_1m)
            current_peak_volume_1m = np.max(volumes_1m)
            
            # Get peak from 5-second candles if available
            current_peak_5s = current_peak_1m  # Default to 1m peak
            current_peak_volume_5s = current_peak_volume_1m
            if candles_5s and len(candles_5s) >= 3:
                highs_5s = self._extract_validated_array(candles_5s, "high_usd", "pump_dump_highs_5s")
                volumes_5s = self._extract_validated_array(candles_5s, "volume_usd", "pump_dump_volumes_5s")
                
                if len(highs_5s) > 0 and len(volumes_5s) > 0:
                    current_peak_5s = np.max(highs_5s)
                    current_peak_volume_5s = np.max(volumes_5s)
            
            # Use the highest peak from either timeframe
            current_peak = max(current_peak_1m, current_peak_5s)
            current_peak_vol = max(current_peak_volume_1m, current_peak_volume_5s)
            
            if current_peak > peak_price:
                token_data["peak_price"] = current_peak
                peak_price = current_peak
            
            if current_peak_vol > peak_volume:
                token_data["peak_volume"] = current_peak_vol
                peak_volume = current_peak_vol
            
            # Enhanced pump end detection with volume exhaustion and distribution signals
            pump_ending_signals = []
            
            # Signal 1: Volume exhaustion - current volume dropped to less than 30% of peak
            current_volume = self._validate_numeric_input(candles_1m[-1].get("volume_usd", 0), "current_volume") if candles_1m else 0
            if peak_volume > 0 and current_volume < peak_volume * 0.3:
                pump_ending_signals.append("volume_exhaustion")
            
            # Signal 2: High-volume bearish candle (distribution)
            if len(candles_1m) >= 2:
                latest_candle = candles_1m[-1]
                open_price = self._validate_numeric_input(latest_candle.get("open_usd", 0), "open_price")
                close_price = self._validate_numeric_input(latest_candle.get("close_usd", 0), "close_price")
                volume = self._validate_numeric_input(latest_candle.get("volume_usd", 0), "volume")
                
                # Check if it's a high-volume red candle
                recent_volumes = self._extract_validated_array(candles_1m[-5:], "volume_usd", "recent_volumes")
                if len(recent_volumes) > 0:
                    avg_volume = np.mean(recent_volumes)
                    if volume > avg_volume * 1.5 and close_price < open_price * 0.95:  # Red candle with 5%+ drop
                        pump_ending_signals.append("high_volume_distribution")
        
            # Signal 3: Consecutive candles with long upper wicks (selling pressure at highs)
            upper_wick_candles = 0
            if len(candles_1m) >= 3:
                for candle in candles_1m[-3:]:
                    high_price = self._validate_numeric_input(candle.get("high_usd", 0), "high_price")
                    open_price = self._validate_numeric_input(candle.get("open_usd", 0), "open_price")
                    close_price = self._validate_numeric_input(candle.get("close_usd", 0), "close_price")
                    
                    # Calculate upper wick size relative to body
                    body_top = max(open_price, close_price)
                    upper_wick = high_price - body_top
                    body_size = abs(close_price - open_price)
                    
                    # Long upper wick if wick is at least 2x the body size
                    if body_size > 0 and upper_wick >= body_size * 2:
                        upper_wick_candles += 1
                    elif upper_wick > (high_price * 0.02):  # Or wick is >2% of price
                        upper_wick_candles += 1
        
            if upper_wick_candles >= 2:
                pump_ending_signals.append("upper_wick_rejection")
            
            # Check for significant price drop from peak (original condition)
            price_drop_signal = False
            if peak_price > 0 and current_price <= peak_price * 0.5:
                pump_ending_signals.append("major_price_drop")
                price_drop_signal = True
            
            # Determine if pump is ending based on signals
            pump_ending = False
            new_status = None
            
            if price_drop_signal:
                # Major price drop - definitely pump and dump
                pump_ending = True
                new_status = "pump_and_dump"
            elif len(pump_ending_signals) >= 2:
                # Multiple ending signals - mark as pump ending
                pump_ending = True
                new_status = "pump_ending"
            elif "volume_exhaustion" in pump_ending_signals and "high_volume_distribution" in pump_ending_signals:
                # Strong combination of volume signals
                pump_ending = True
                new_status = "pump_ending"
            
            if pump_ending:
                original_status = token_data.get("pump_status")
                token_data["pump_status"] = new_status
                
                # Update token state
                if mint in self.memory["token_states"]:
                    self.memory["token_states"][mint] = "inactive"
                elif mint in self.pump_memory["tokens"]:
                    if "token_states" not in self.pump_memory:
                        self.pump_memory["token_states"] = {}
                    self.pump_memory["token_states"][mint] = "inactive"
                
                if peak_price > 0:
                    drop_percent = ((peak_price - current_price) / peak_price * 100)
                    self.logger.info(f"Token {mint[:8]}... pump ending detected. Status: {original_status} -> {new_status}. Signals: {pump_ending_signals}. Price dropped {drop_percent:.1f}% from peak.")
                else:
                    self.logger.info(f"Token {mint[:8]}... pump ending detected. Status: {original_status} -> {new_status}. Signals: {pump_ending_signals}.")
                
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Critical error in check_pump_and_dump: {e}")
            return False
    
    def calculate_ema(self,prices,span):
        if len(prices)<span:return np.mean(prices) if prices else 0
        alpha=2.0/(span+1)
        ema=prices[0]
        for price in prices[1:]:ema=alpha*price+(1-alpha)*ema
        return ema
    
    def calculate_macd(self,prices,fast=6,slow=13,signal=5):
        if len(prices)<slow:return 0,0,0
        ef=self.calculate_ema(prices,fast)
        es=self.calculate_ema(prices,slow)
        ml=ef-es
        sl=ml*0.8
        h=ml-sl
        return ml,sl,h
    
    def calculate_rsi(self,prices,period=7):
        if len(prices)<period+1:return 50
        d=np.diff(prices)
        g=np.where(d>0,d,0)
        l=np.where(d<0,-d,0)
        ag=np.mean(g[-period:])
        al=np.mean(l[-period:])
        if al==0:return 100
        rs=ag/al
        return 100-(100/(1+rs))
    
    def safe_divide(self, a, b, default=0):
        """Safe division with comprehensive data validation and type checking"""
        try:
            # Critical data validation - ensure inputs are numeric
            a = self._validate_numeric_input(a, "dividend")
            b = self._validate_numeric_input(b, "divisor")
            
            # Handle both scalar and array inputs
            if np.isscalar(a) and np.isscalar(b):
                if abs(b) < 1e-10:
                    return default
                return a / b
            else:
                # For arrays, use numpy operations
                a = np.asarray(a, dtype=np.float32)
                b = np.asarray(b, dtype=np.float32)
                
                # Create result array with default values
                result = np.full_like(a, default, dtype=np.float32)
                
                # Only divide where b is not close to zero
                valid_mask = np.abs(b) >= 1e-10
                if valid_mask.any():
                    result[valid_mask] = a[valid_mask] / b[valid_mask]
                
                # Return scalar if input was effectively scalar
                if result.size == 1:
                    return float(result.item())
                return result
        except Exception as e:
            self.logger.error(f"Critical error in safe_divide: {e}, a={type(a)}, b={type(b)}")
            return default
    
    def _validate_numeric_input(self, value, name="value"):
        """Critical data validation to prevent dict/float comparison errors"""
        try:
            # Handle None values
            if value is None:
                return 0.0
            
            # Handle dict/list inputs (corrupted data)
            if isinstance(value, (dict, list)):
                self.logger.warning(f"Critical data corruption detected: {name} is {type(value)}, converting to 0")
                return 0.0
            
            # Handle string inputs
            if isinstance(value, str):
                try:
                    return float(value)
                except (ValueError, TypeError):
                    self.logger.warning(f"Invalid string value for {name}: {value}, converting to 0")
                    return 0.0
            
            # Handle numpy arrays
            if isinstance(value, np.ndarray):
                # Validate all elements are numeric
                if value.size == 0:
                    return 0.0
                # Check for non-numeric elements
                try:
                    value = value.astype(np.float32)
                    # Replace any NaN or inf values
                    value = np.where(np.isfinite(value), value, 0.0)
                    return value
                except (ValueError, TypeError):
                    self.logger.warning(f"Invalid array data for {name}, converting to zeros")
                    return np.zeros_like(value, dtype=np.float32)
            
            # Convert to float and validate
            numeric_value = float(value)
            if not np.isfinite(numeric_value):
                return 0.0
            
            return numeric_value
            
        except Exception as e:
            self.logger.error(f"Critical validation error for {name}: {e}, value={value}, type={type(value)}")
            return 0.0
    
    def _validate_candle_data(self, candles, timeframe):
        """Validate and clean candle data to prevent corruption errors"""
        if not candles:
            return []
        
        validated_candles = []
        for i, candle in enumerate(candles):
            try:
                # Ensure candle is a dictionary
                if not isinstance(candle, dict):
                    self.logger.warning(f"Invalid candle data at index {i} for {timeframe}: not a dict")
                    continue
                
                # Validate required fields exist and are numeric
                required_fields = ["close_usd", "volume_usd", "high_usd", "low_usd", "open_usd"]
                validated_candle = {}
                
                for field in required_fields:
                    if field in candle:
                        validated_candle[field] = self._validate_numeric_input(candle[field], f"{timeframe}_{field}")
                    else:
                        validated_candle[field] = 0.0
                
                # Optional fields
                validated_candle["buy_count"] = self._validate_numeric_input(candle.get("buy_count", 0), f"{timeframe}_buy_count")
                validated_candle["sell_count"] = self._validate_numeric_input(candle.get("sell_count", 0), f"{timeframe}_sell_count")
                validated_candle["timestamp"] = candle.get("timestamp", 0)
                
                validated_candles.append(validated_candle)
                
            except Exception as e:
                self.logger.error(f"Critical error validating candle {i} for {timeframe}: {e}")
                continue
        
        return validated_candles
    
    def _extract_validated_array(self, candles, field, description, default_val=0.0):
        """Extract and validate numeric arrays from candle data"""
        try:
            if not candles:
                return np.array([], dtype=np.float32)
            
            values = []
            for candle in candles:
                if isinstance(candle, dict) and field in candle:
                    val = self._validate_numeric_input(candle[field], f"{description}_{field}")
                    values.append(val)
                else:
                    values.append(default_val)
            
            return np.array(values, dtype=np.float32)
            
        except Exception as e:
            self.logger.error(f"Critical error extracting {description}: {e}")
            return np.array([], dtype=np.float32)
    
    def extract_features(self, candles_5s=None, candles_1m=None, mint=None):
        """Extract features for real-time state identification with strict data validation"""
        try:
            # Critical data validation
            if not candles_5s or len(candles_5s) < 6:
                return None
            
            # Validate candle data structure
            candles_5s = self._validate_candle_data(candles_5s, "5s")
            candles_1m = self._validate_candle_data(candles_1m, "1m") if candles_1m else []
            
            if not candles_5s:
                return None
            
            # Use recent candles for feature extraction - PRIORITIZE SHORT-TERM DATA
            recent_1m = candles_1m[-6:] if len(candles_1m) >= 6 else candles_1m  # Reduced from 12 to 6
            recent_5s = candles_5s[-6:] if candles_5s and len(candles_5s) >= 6 else candles_5s or []  # Reduced from 60 to 6
            
            # Get token data for state-specific features
            token_data = None
            if mint and mint in self.memory["tokens"]:
                token_data = self.memory["tokens"][mint]
            elif mint and mint in self.pump_memory["tokens"]:
                token_data = self.pump_memory["tokens"][mint]
        
            # Extract 1-minute features with validation
            prices_1m = self._extract_validated_array(recent_1m, "close_usd", "1m prices")
            volumes_1m = self._extract_validated_array(recent_1m, "volume_usd", "1m volumes")
            highs_1m = self._extract_validated_array(recent_1m, "high_usd", "1m highs")
            lows_1m = self._extract_validated_array(recent_1m, "low_usd", "1m lows")
            opens_1m = self._extract_validated_array(recent_1m, "open_usd", "1m opens")
            buy_counts_1m = self._extract_validated_array(recent_1m, "buy_count", "1m buy_counts", default_val=0)
            sell_counts_1m = self._extract_validated_array(recent_1m, "sell_count", "1m sell_counts", default_val=0)
            
            if prices_1m.size == 0 or volumes_1m.size == 0:
                return None
            
            # Basic price and volume features
            price_change = self.safe_divide(prices_1m[-1] - prices_1m[0], prices_1m[0])
            price_volatility = self.safe_divide(np.std(prices_1m), np.mean(prices_1m))
            avg_volume = np.mean(volumes_1m)
            volume_trend = self.safe_divide(volumes_1m[-1] - volumes_1m[0], volumes_1m[0])
            
            # PRIORITY: 5-SECOND VELOCITY AND ACCELERATION FEATURES
            # Price Velocity: immediate upward thrust over last 3-4 candles
            price_velocity_5s = 0
            volume_acceleration_5s = 0
            buy_pressure_imbalance_5s = 1.0
            
            if recent_5s and len(recent_5s) >= 4:
                prices_5s = np.array([float(c["close_usd"]) for c in recent_5s], dtype=np.float32)
                volumes_5s = np.array([float(c["volume_usd"]) for c in recent_5s], dtype=np.float32)
                
                # Price Velocity: price change over last 3-4 candles (immediate thrust)
                price_velocity_5s = self.safe_divide(prices_5s[-1] - prices_5s[-4], prices_5s[-4]) if len(prices_5s) >= 4 else 0
                
                # Volume Acceleration: rate of change of volume over last 3-4 candles
                if len(volumes_5s) >= 4:
                    vol_change_recent = volumes_5s[-1] - volumes_5s[-2]
                    vol_change_prev = volumes_5s[-2] - volumes_5s[-3]
                    volume_acceleration_5s = vol_change_recent - vol_change_prev
                
                # Buy Pressure Imbalance: massive spike in buy/sell ratio over last 3-6 candles
                recent_buys = sum(float(c.get("buy_count", 0)) for c in recent_5s[-6:])
                recent_sells = sum(float(c.get("sell_count", 0)) for c in recent_5s[-6:])
                buy_pressure_imbalance_5s = self.safe_divide(recent_buys, max(recent_sells, 1), 1.0)
            
            # SECONDARY: 1-minute contextual features (deprioritized)
            momentum_1m = self.safe_divide(prices_1m[-1] - prices_1m[-2], prices_1m[-2]) if len(prices_1m) > 1 else 0
            acceleration_1m = self.safe_divide(prices_1m[-1] - 2 * prices_1m[-2] + prices_1m[-3], prices_1m[-3]) if len(prices_1m) > 2 else 0
            
            # ENHANCED: Momentum acceleration tracking for 5-second data
            momentum_acceleration_5s = 0
            price_thrust_consistency = 0
            
            if recent_5s and len(recent_5s) >= 6:
                prices_5s = np.array([float(c["close_usd"]) for c in recent_5s], dtype=np.float32)
                
                # Calculate momentum acceleration (is the pump speeding up?)
                if len(prices_5s) >= 4:
                    current_momentum = self.safe_divide(prices_5s[-1] - prices_5s[-2], prices_5s[-2])
                    prev_momentum = self.safe_divide(prices_5s[-2] - prices_5s[-3], prices_5s[-3])
                    momentum_acceleration_5s = current_momentum - prev_momentum
                
                # Price thrust consistency: are consecutive candles all green?
                consecutive_gains = 0
                for i in range(len(prices_5s) - 1, 0, -1):
                    if prices_5s[i] > prices_5s[i-1]:
                        consecutive_gains += 1
                    else:
                        break
                price_thrust_consistency = consecutive_gains
            
            # CONTEXTUAL: 1-minute momentum change (secondary importance)
            momentum_change_1m = 0
            if len(prices_1m) >= 4:
                prev_momentum = self.safe_divide(prices_1m[-2] - prices_1m[-3], prices_1m[-3])
                momentum_change_1m = momentum_1m - prev_momentum
            
            # Ratio features - comparing immediate past to recent past
            # Volume ratio: last 1 minute vs previous 3 minutes
            volume_ratio_recent = 1.0
            if len(volumes_1m) >= 4:
                last_1min_vol = volumes_1m[-1]
                prev_3min_vol = np.mean(volumes_1m[-4:-1]) if len(volumes_1m) >= 4 else volumes_1m[-1]
                volume_ratio_recent = self.safe_divide(last_1min_vol, prev_3min_vol, 1.0)
            
            # CRITICAL: Buy pressure surge detection over last 15-30 seconds (3-6 candles)
            buy_pressure_surge_15s = 1.0
            buy_pressure_surge_30s = 1.0
            volume_surge_intensity = 0
            
            if recent_5s and len(recent_5s) >= 3:
                # Last 15 seconds (3 candles) - immediate pump ignition signal
                recent_3_candles = recent_5s[-3:]
                buys_15s = sum(float(c.get("buy_count", 0)) for c in recent_3_candles)
                sells_15s = sum(float(c.get("sell_count", 0)) for c in recent_3_candles)
                buy_pressure_surge_15s = self.safe_divide(buys_15s, max(sells_15s, 1), 1.0)
                
                # Volume surge intensity in last 15 seconds
                volumes_15s = [float(c.get("volume_usd", 0)) for c in recent_3_candles]
                if len(volumes_15s) >= 2:
                    volume_surge_intensity = max(volumes_15s) / (np.mean(volumes_15s) + 1e-9)
            
            if recent_5s and len(recent_5s) >= 6:
                # Last 30 seconds (6 candles) - sustained pump confirmation
                recent_6_candles = recent_5s[-6:]
                buys_30s = sum(float(c.get("buy_count", 0)) for c in recent_6_candles)
                sells_30s = sum(float(c.get("sell_count", 0)) for c in recent_6_candles)
                buy_pressure_surge_30s = self.safe_divide(buys_30s, max(sells_30s, 1), 1.0)
            
            # Candlestick pattern features
            # Count consecutive bullish candles
            consecutive_green_candles = 0
            for i in range(len(recent_1m) - 1, -1, -1):
                candle = recent_1m[i]
                if float(candle["close_usd"]) > float(candle["open_usd"]):
                    consecutive_green_candles += 1
                else:
                    break
            
            # Average body-to-wick ratio of last few candles
            avg_body_to_wick_ratio = 0
            if len(recent_1m) >= 3:
                body_wick_ratios = []
                for candle in recent_1m[-3:]:
                    high_price = float(candle["high_usd"])
                    low_price = float(candle["low_usd"])
                    open_price = float(candle["open_usd"])
                    close_price = float(candle["close_usd"])
                    
                    body_size = abs(close_price - open_price)
                total_range = high_price - low_price
                
                if total_range > 1e-9:
                    body_to_total_ratio = self.safe_divide(body_size, total_range, 0)
                    body_wick_ratios.append(body_to_total_ratio)
            
                if body_wick_ratios:
                    avg_body_to_wick_ratio = np.mean(body_wick_ratios)
            
            # Statistical features
            price_skew = stats.skew(prices_1m) if len(prices_1m) > 2 else 0
            volume_skew = stats.skew(volumes_1m) if len(volumes_1m) > 2 else 0
            price_range = self.safe_divide(np.max(prices_1m), np.min(prices_1m), 1)
            
            # Volume-price correlation
            volume_price_corr = 0
            if len(prices_1m) > 1 and np.std(prices_1m) > 1e-9 and np.std(volumes_1m) > 1e-9:
                try:
                    volume_price_corr = np.corrcoef(volumes_1m, prices_1m)[0, 1]
                    volume_price_corr = 0 if np.isnan(volume_price_corr) else volume_price_corr
                except:
                    volume_price_corr = 0
            
            # Buy/sell pressure
            total_buys = np.sum(buy_counts_1m)
            total_sells = np.sum(sell_counts_1m)
            buy_sell_ratio = self.safe_divide(total_buys, total_sells, 1)
            net_pressure = self.safe_divide(total_buys - total_sells, total_buys + total_sells, 0)
            
            # Calculate recent average volume
            avg_vol_recent = np.mean(volumes_1m[-5:-1]) if len(volumes_1m) > 1 else 0
            
            # Is the latest volume spike mostly selling?
            sell_volume_spike = 0
            if avg_vol_recent > 0 and volumes_1m[-1] > avg_vol_recent * 2:  # If volume doubled
                if sell_counts_1m[-1] > buy_counts_1m[-1]:
                    sell_volume_spike = 1
            
            # Volume acceleration feature
            volume_acceleration = 0
            if len(volumes_1m) >= 5:
                recent_volumes = volumes_1m[-5:]
                volume_changes = np.diff(recent_volumes)
                volume_acceleration = np.mean(volume_changes) if len(volume_changes) > 0 else 0

            # 1-minute candlestick features (fast indicators only)
            candle_1m_features = [
            np.std(prices_1m),                                             # Price volatility
            np.mean(volumes_1m),                                          # Avg volume
            self.safe_divide(np.max(highs_1m), np.min(lows_1m), 1),      # High/Low ratio
            self.safe_divide(volumes_1m[-1], np.mean(volumes_1m[:-1]), 1) if len(volumes_1m) > 1 else 1  # Volume surge
            ]
            
            # 5-second candlestick features (if available)
            candle_5s_features = [0] * 4  # Default values
            if recent_5s and len(recent_5s) >= 3:
                prices_5s = np.array([float(c["close_usd"]) for c in recent_5s], dtype=np.float32)
                volumes_5s = np.array([float(c["volume_usd"]) for c in recent_5s], dtype=np.float32)
                highs_5s = np.array([float(c["high_usd"]) for c in recent_5s], dtype=np.float32)
                lows_5s = np.array([float(c["low_usd"]) for c in recent_5s], dtype=np.float32)
                opens_5s = np.array([float(c["open_usd"]) for c in recent_5s], dtype=np.float32)
                
                candle_5s_features = [
                    np.std(prices_5s),                                             # Price volatility
                    np.mean(volumes_5s),                                          # Avg volume
                    self.safe_divide(np.max(highs_5s), np.min(lows_5s), 1),      # High/Low ratio
                    self.safe_divide(volumes_5s[-1], np.mean(volumes_5s[:-1]), 1) if len(volumes_5s) > 1 else 1  # Volume surge
                ]
            
            # Core Pump Features for State Recognition
            price_change_from_initial = 0
            price_drop_from_peak = 0
            time_since_initial_rise = 0
            pump_magnitude_category = 0
            price_acceleration_feature = acceleration_1m
            volume_vs_average = self.safe_divide(volumes_1m[-1], np.mean(volumes_1m), 1) if len(volumes_1m) > 1 else 1
            
            if token_data:
                current_price = prices_1m[-1] if len(prices_1m) > 0 else 0
                initial_price = token_data.get("initial_price", current_price)
                peak_price = token_data.get("peak_price", current_price)
                
                # Core state features
                if initial_price > 0:
                    price_change_from_initial = ((current_price - initial_price) / initial_price) * 100
                    
                    # Pump magnitude categorization
                    if price_change_from_initial < 50:
                        pump_magnitude_category = 0
                    elif price_change_from_initial < 150:
                        pump_magnitude_category = 1  # Regular
                    elif price_change_from_initial < 300:
                        pump_magnitude_category = 2  # Great
                    else:
                        pump_magnitude_category = 3  # Super
                
                if peak_price > 0:
                    price_drop_from_peak = ((peak_price - current_price) / peak_price) * 100
            
            # Time since initial rise
            initial_rise_time = token_data.get("initial_rise_time") if token_data else None
            if initial_rise_time:
                try:
                    rise_time = datetime.fromisoformat(initial_rise_time)
                    time_since_initial_rise = (datetime.now() - rise_time).total_seconds() / 60  # minutes
                except:
                    time_since_initial_rise = 0
            
            # Volume vs average (last 20 candles)
            if len(volumes_1m) >= 20:
                avg_volume_20 = np.mean(volumes_1m[-20:])
                volume_vs_average = self.safe_divide(volumes_1m[-1], avg_volume_20, 1)
            
            # Combine all features - PRIORITIZING 5-SECOND TRIGGER FEATURES
            features = [
            # PRIMARY: 5-Second Trigger Features (most important for rapid detection)
            price_velocity_5s,              # Immediate price thrust
            volume_acceleration_5s,         # Volume rapidly increasing
            buy_pressure_imbalance_5s,      # Massive buy imbalance
            momentum_acceleration_5s,       # Pump speeding up
            price_thrust_consistency,       # Consecutive green candles
            buy_pressure_surge_15s,         # 15-second buy pressure
            buy_pressure_surge_30s,         # 30-second buy pressure  
            volume_surge_intensity,         # Volume spike intensity
            
            # SECONDARY: Core Pump Features for State Recognition
            price_change_from_initial,
            price_drop_from_peak,
            time_since_initial_rise,
            pump_magnitude_category,
            price_acceleration_feature,
            volume_vs_average,
            
            # CONTEXTUAL: Basic features (deprioritized)
            price_change, price_volatility, avg_volume, volume_trend, 
            # 1-minute momentum features (contextual only)
            momentum_1m, acceleration_1m, momentum_change_1m,
            # Ratio features
            volume_ratio_recent,
            # Candlestick pattern features
            consecutive_green_candles, avg_body_to_wick_ratio,
            # Statistical features
            price_skew, volume_skew, price_range, volume_price_corr,
            len(recent_1m), np.sum(volumes_1m), np.std(volumes_1m), np.mean(prices_1m),
            self.safe_divide(prices_1m[-1], prices_1m[0], 1),
            self.safe_divide(volumes_1m[-1], volumes_1m[0], 1),
            self.safe_divide(np.max(volumes_1m), np.mean(volumes_1m), 1),
            self.safe_divide(np.max(prices_1m), np.mean(prices_1m), 1),
            self.safe_divide(np.max(prices_1m) - np.min(prices_1m), np.min(prices_1m)),
            self.safe_divide(1, 1 + price_volatility, 1) if price_volatility > 0 else 1,
            np.gradient(prices_1m)[-1] if len(prices_1m) > 1 else 0,
            # Buy/sell pressure
            buy_sell_ratio, net_pressure, total_buys, total_sells,
            # NEW: Pump maturity feature
            len(candles_1m),
            # NEW: Sell-side pressure feature
            sell_volume_spike,
            # NEW: Volume acceleration feature
            volume_acceleration
        ]
        
            # Add candlestick features
            features.extend(candle_1m_features)
            features.extend(candle_5s_features)
            
            # Ensure all features are finite and validated
            features = [self._validate_numeric_input(f, f"feature_{i}") for i, f in enumerate(features)]
            
            # Final validation - ensure no corrupted data
            if len(features) == 0:
                self.logger.warning("No valid features extracted")
                return None
            
            # Check for any remaining invalid values
            valid_features = [f for f in features if np.isfinite(f)]
            if len(valid_features) != len(features):
                invalid_count = len(features) - len(valid_features)
                self.logger.warning(f"Replaced {invalid_count} invalid feature values with 0")
                features = [f if np.isfinite(f) else 0.0 for f in features]
            
            return features
            
        except Exception as e:
            self.logger.error(f"Critical error in extract_features: {e}")
            return None

    def _initialize_token_order(self):
        """Initialize token order based on existing memory, sorted by first_seen timestamp"""
        # Sort all tokens by their first_seen timestamp to create the initial order
        all_tokens = []
        for mint, token_data in self.memory["tokens"].items():
            first_seen = token_data.get("first_seen", "")
            all_tokens.append((first_seen, mint))
        
        # Sort by timestamp and add to token_order
        all_tokens.sort(key=lambda x: x[0])
        self.token_order = deque(mint for _, mint in all_tokens)

    def _manage_rolling_memory(self):
        """Conveyor belt system - continuously removes oldest tokens when memory limit exceeded"""
        while len(self.memory["tokens"]) > self.max_memory_tokens:
            if not self.token_order:
                break
            
            # Remove the oldest token from the front of the deque
            oldest_mint = self.token_order.popleft()
            
            if oldest_mint in self.memory["tokens"]:
                # Delete from all memory dictionaries
                del self.memory["tokens"][oldest_mint]
                if oldest_mint in self.memory["token_states"]:
                    del self.memory["token_states"][oldest_mint]
                if oldest_mint in self.last_snapshot_time:
                    del self.last_snapshot_time[oldest_mint]
        
    def _manage_rolling_pump_memory(self):
        """Conveyor belt system for pump memory - removes oldest pump tokens when limit exceeded"""
        while len(self.pump_memory["tokens"]) > self.max_pump_memory_tokens:
            # Find the oldest token by first_seen timestamp
            oldest_mint = None
            oldest_time = None
            
            for mint, data in self.pump_memory["tokens"].items():
                first_seen = data.get("first_seen")
                if first_seen:
                    if oldest_time is None or first_seen < oldest_time:
                        oldest_time = first_seen
                        oldest_mint = mint
            
            if oldest_mint:
                # Remove the oldest token from pump memory
                del self.pump_memory["tokens"][oldest_mint]
                if oldest_mint in self.pump_memory["token_states"]:
                    del self.pump_memory["token_states"][oldest_mint]
                if oldest_mint in self.pump_memory.get("last_snapshot_time", {}):
                    del self.pump_memory["last_snapshot_time"][oldest_mint]
            else:
                break  # No tokens with first_seen timestamp found

    def _store_feature_cache(self, mint, features, pump_status):
        """Store minimal feature data for removed tokens"""
        if not hasattr(self, 'feature_cache'):
            self.feature_cache = {}
        self.feature_cache[mint] = {
            'features': features,
            'pump_status': pump_status,
            'cached_at': time.time()
        }
        # Limit feature cache size
        if len(self.feature_cache) > self.max_memory_tokens // 2:
            oldest_cached = min(self.feature_cache.keys(), 
                            key=lambda k: self.feature_cache[k]['cached_at'])
            del self.feature_cache[oldest_cached]

    def prune_memory(self):
        """More efficient prune_memory with batch operations and rolling memory management"""
        # Identify all dud tokens that can be removed
        dud_tokens = set()
        for mint, data in self.memory["tokens"].items():
            if data.get("pump_status") in ["dud", "pump_fake", "pump_and_dump"]:
                dud_tokens.add(mint)
        
        # Remove excess duds in a single batch operation
        if len(dud_tokens) > self.max_duds:
            # Sort by first_seen timestamp to remove oldest duds first
            sorted_duds = sorted(list(dud_tokens), 
                               key=lambda m: self.memory["tokens"][m].get("first_seen", ""))
            tokens_to_remove = sorted_duds[:len(sorted_duds) - self.max_duds]
            
            # Batch delete from all dictionaries
            for mint in tokens_to_remove:
                if mint in self.memory["tokens"]:
                    del self.memory["tokens"][mint]
                if mint in self.memory["token_states"]:
                    del self.memory["token_states"][mint]
                if mint in self.last_snapshot_time:
                    del self.last_snapshot_time[mint]
            
            # Update token_order efficiently
            tokens_to_remove_set = set(tokens_to_remove)
            self.token_order = deque(mint for mint in self.token_order if mint not in tokens_to_remove_set)
            
            self.logger.info(f"Pruned {len(tokens_to_remove)} old dud tokens from memory.")
        
        # Apply rolling memory management to enforce main memory limit
        self._manage_rolling_memory()
        
        # Explicitly call garbage collector to free memory immediately
        gc.collect()

    
    async def update_token_data(self, mint, ohlcv_data):
        """Update token data using BOTH 1-minute and 5-second OHLCV candlestick data with feature pre-computation"""
        current_time = time.time()
        
        # Extract candlestick data from both timeframes first
        candles_1m = ohlcv_data.get('ohlcv', [])
        candles_5s = ohlcv_data.get('ohlcv_5s', [])
        
        # Check if token is in pump memory and handle it there
        if mint in self.pump_memory["tokens"]:
            await self._update_pump_memory_token(mint, ohlcv_data, candles_1m, candles_5s)
            return
        
        # Track new tokens in order
        if mint not in self.memory["tokens"]:
            # Get initial price from first available candle
            initial_price = 0
            if candles_1m:
                initial_price = float(candles_1m[0]["close_usd"])
            elif candles_5s:
                initial_price = float(candles_5s[0]["close_usd"])
            
            self.memory["tokens"][mint] = {
                "first_seen": datetime.now().isoformat(),
                "candles_1m": [],
                "candles_5s": [],
                "training_samples": [],  # Real-time state samples
                "pump_status": None,
                "initial_price": initial_price,
                "peak_price": initial_price,
                "initial_rise_time": None,  # When price first broke out of baseline
                "current_state": 0,  # Start in Baseline state
                "state_history": []  # Track state transitions
            }
            # Add new token to the end of token_order list for age tracking
            self.token_order.append(mint)
        
        if not candles_1m and not candles_5s:
            return
        
        # Store latest candle data (keep as lists for JSON serialization)
        MAX_CANDLES_TO_KEEP = 50
        self.memory["tokens"][mint]["candles_1m"] = candles_1m[-MAX_CANDLES_TO_KEEP:] if candles_1m else []
        self.memory["tokens"][mint]["candles_5s"] = candles_5s[-MAX_CANDLES_TO_KEEP*12:] if candles_5s else []
        
        # Get current price from latest available candle
        if candles_1m:
            current_price = float(candles_1m[-1]["close_usd"])
        elif candles_5s:
            current_price = float(candles_5s[-1]["close_usd"])
        else:
            return
        
        # Update peak price tracking
        token_data = self.memory["tokens"][mint]
        if current_price > token_data["peak_price"]:
            token_data["peak_price"] = current_price
        
        # Real-time state identification - assign state based on current data only
        current_state = self._identify_current_state(mint, candles_1m, candles_5s, current_price)
        
        # Store state transition for training
        if current_state != token_data["current_state"]:
            token_data["state_history"].append({
                "timestamp": datetime.now().isoformat(),
                "old_state": token_data["current_state"],
                "new_state": current_state,
                "price": current_price
            })
            
            # Retroactive labeling based on confirmed state transitions
            self._perform_retroactive_labeling(mint, token_data["current_state"], current_state, candles_5s, candles_1m)
            
            token_data["current_state"] = current_state
        
        # Generate training sample with current state as label
        self._generate_real_time_training_sample(mint, candles_1m, candles_5s, current_state)
        
        # Check for pump and dump using both timeframes
        self.check_pump_and_dump(mint, current_price, candles_1m, candles_5s)
        
        # Classify token state using both timeframes
        old_state = self.memory["token_states"].get(mint, "new")
        new_state = self.classify_token_state(candles_1m, candles_5s)
        self.memory["token_states"][mint] = new_state
        
        # State-based pump detection
        token_data = self.memory["tokens"][mint]
        current_model_state = token_data["current_state"]
        
        # Update pump_status based on state transitions
        if current_model_state == 2:  # Momentum Spike state
            if token_data["pump_status"] is None:
                token_data["pump_status"] = "confirmed_pump"
                
                # Calculate pump multiplier
                initial_price = token_data["initial_price"]
                peak_price = token_data["peak_price"]
                pump_multiplier = peak_price / initial_price if initial_price > 1e-9 else 1.0
                
                token_data["pump_multiplier"] = pump_multiplier
                
                # Move to pump memory
                self.pump_memory["tokens"][mint] = self.memory["tokens"][mint]
                self.pump_memory["token_states"][mint] = self.memory["token_states"][mint]
                del self.memory["tokens"][mint]
                del self.memory["token_states"][mint]
                
                if mint in self.token_order:
                    self.token_order.remove(mint)
                
                # Manage pump memory size limit
                self._manage_rolling_pump_memory()
                
                self.logger.info(f"State-based pump detected for {mint[:8]}... [State 2: Momentum Spike] (multiplier: {pump_multiplier:.2f}x) - moved to pump memory")
        
        elif current_model_state == 4:  # Downfall state
            if token_data["pump_status"] is None:
                token_data["pump_status"] = "pump_and_dump"
                self.logger.info(f"Token entered downfall state: {mint[:8]}... [State 4: Downfall]")
        
        elif current_model_state == 0 and new_state == "inactive":  # Baseline + inactive
            if token_data["pump_status"] is None:
                token_data["pump_status"] = "dud"
                self.logger.info(f"Token marked as dud: {mint[:8]}... [State 0: Baseline, inactive]")
    
    async def _update_pump_memory_token(self, mint, ohlcv_data, candles_1m, candles_5s):
        """Update token data for tokens in pump memory to continue generating training samples"""
        if not candles_1m and not candles_5s:
            return
        
        # Store latest candle data
        MAX_CANDLES_TO_KEEP = 50
        self.pump_memory["tokens"][mint]["candles_1m"] = candles_1m[-MAX_CANDLES_TO_KEEP:] if candles_1m else []
        self.pump_memory["tokens"][mint]["candles_5s"] = candles_5s[-MAX_CANDLES_TO_KEEP*12:] if candles_5s else []
        
        # Get current price from latest available candle
        if candles_1m:
            current_price = float(candles_1m[-1]["close_usd"])
        elif candles_5s:
            current_price = float(candles_5s[-1]["close_usd"])
        else:
            return
        
        # Update peak price tracking
        token_data = self.pump_memory["tokens"][mint]
        if current_price > token_data["peak_price"]:
            token_data["peak_price"] = current_price
        
        # Real-time state identification - assign state based on current data only
        current_state = self._identify_current_state_pump_memory(mint, candles_1m, candles_5s, current_price)
        
        # Store state transition for training
        if current_state != token_data["current_state"]:
            token_data["state_history"].append({
                "timestamp": datetime.now().isoformat(),
                "old_state": token_data["current_state"],
                "new_state": current_state,
                "price": current_price
            })
            
            # Retroactive labeling based on confirmed state transitions
            self._perform_retroactive_labeling_pump_memory(mint, token_data["current_state"], current_state, candles_5s, candles_1m)
            
            token_data["current_state"] = current_state
        
        # Generate training sample with current state as label
        self._generate_real_time_training_sample_pump_memory(mint, candles_1m, candles_5s, current_state)
        
        # Check for pump and dump using both timeframes
        self.check_pump_and_dump(mint, current_price, candles_1m, candles_5s)
        
        # Classify token state using both timeframes
        old_state = self.pump_memory["token_states"].get(mint, "new")
        new_state = self.classify_token_state(candles_1m, candles_5s)
        self.pump_memory["token_states"][mint] = new_state
        
        # Update pump_status based on state transitions
        current_model_state = token_data["current_state"]
        
        if current_model_state == 4:  # Downfall state
            if token_data["pump_status"] == "confirmed_pump":
                token_data["pump_status"] = "pump_and_dump"
                self.logger.info(f"Token entered downfall state: {mint[:8]}... [State 4: Downfall]")
                
    def _identify_current_state(self, mint, candles_1m, candles_5s, current_price):
        """
        ULTRA-FAST 5-SECOND CANDLE FOCUSED STATE IDENTIFICATION
        
        New Ultra-Fast States (optimized for 5-second timeframes):
        0: Quiescent (0-15 seconds) - Token brand new/inactive, insufficient data
        1: Ignition (1-3 minutes) - Initial pump phase with significant volume and price jump
        2: Acceleration (30-90 seconds) - Main pump phase with consecutive large green 5s candles
        3: Peak Distribution (exhaustion pattern) - Multiple candles with long upper wicks or volume divergence
        4: Reversal (The Dump) - First large red 5s candle erasing previous gains
        """
        token_data = self.memory["tokens"][mint]
        
        # State 0 (Quiescent): If len(candles_5s) < 4
        if not candles_5s or len(candles_5s) < 4:
            return 0
        
        # Get recent 5s candle data for analysis
        recent_5s = candles_5s[-4:]  # Last 4 candles (20 seconds)
        current_candle = candles_5s[-1]
        
        # Calculate key fast metrics
        current_volume = float(current_candle.get("volume_usd", 0))
        current_close = float(current_candle.get("close_usd", 0))
        current_open = float(current_candle.get("open_usd", 0))
        current_high = float(current_candle.get("high_usd", 0))
        
        # Calculate Volume Acceleration (change in volume momentum)
        volume_acceleration = self._calculate_volume_acceleration_fast(candles_5s)
        
        # Calculate Buy Pressure Imbalance
        buy_pressure_imbalance = self._calculate_buy_pressure_imbalance(current_candle)
        
        # Calculate Price Velocity (price momentum over recent candles)
        price_velocity = self._calculate_price_velocity_fast(recent_5s)
        
        # Check for large red candle (State 4: Reversal)
        candle_change_percent = ((current_close - current_open) / current_open * 100) if current_open > 0 else 0
        if candle_change_percent < -5:  # Large red candle erasing gains
            # Check if it's erasing recent gains
            if len(candles_5s) >= 3:
                prev_gains = self._check_recent_gains_erased(candles_5s[-3:])
                if prev_gains:
                    return 4
        
        # State 3 (Peak Distribution): Look for exhaustion patterns over multiple candles
        exhaustion_pattern = self._detect_exhaustion_pattern(candles_5s, candles_1m)
        if exhaustion_pattern:
            return 3
        
        # State 2 (Acceleration): High Price Velocity for several consecutive candles
        if price_velocity > 2.0:  # Strong positive momentum
            consecutive_green = self._count_consecutive_green_candles(recent_5s)
            if consecutive_green >= 2:  # At least 2 consecutive green candles
                return 2
        
        # State 1 (Ignition): Broader initial pump phase detection (1-3 minutes window)
        ignition_detected = self._detect_ignition_phase(candles_5s, candles_1m, token_data)
        if ignition_detected:
            return 1
        
        # Default to State 0 (Quiescent)
        return 0
    
    def _identify_current_state_pump_memory(self, mint, candles_1m, candles_5s, current_price):
        """
        ULTRA-FAST 5-SECOND CANDLE FOCUSED STATE IDENTIFICATION (Pump Memory Version)
        
        New Ultra-Fast States (optimized for 5-second timeframes):
        0: Quiescent (0-15 seconds) - Token brand new/inactive, insufficient data
        1: Ignition (1-3 minutes) - Initial pump phase with significant volume and price jump
        2: Acceleration (30-90 seconds) - Main pump phase with consecutive large green 5s candles
        3: Peak Distribution (exhaustion pattern) - Multiple candles with long upper wicks or volume divergence
        4: Reversal (The Dump) - First large red 5s candle erasing previous gains
        """
        token_data = self.pump_memory["tokens"][mint]
        
        # State 0 (Quiescent): If len(candles_5s) < 4
        if not candles_5s or len(candles_5s) < 4:
            return 0
        
        # Get recent 5s candle data for analysis
        recent_5s = candles_5s[-4:]  # Last 4 candles (20 seconds)
        current_candle = candles_5s[-1]
        
        # Calculate key fast metrics
        current_volume = float(current_candle.get("volume_usd", 0))
        current_close = float(current_candle.get("close_usd", 0))
        current_open = float(current_candle.get("open_usd", 0))
        current_high = float(current_candle.get("high_usd", 0))
        
        # Calculate Volume Acceleration (change in volume momentum)
        volume_acceleration = self._calculate_volume_acceleration_fast(candles_5s)
        
        # Calculate Buy Pressure Imbalance
        buy_pressure_imbalance = self._calculate_buy_pressure_imbalance(current_candle)
        
        # Calculate Price Velocity (price momentum over recent candles)
        price_velocity = self._calculate_price_velocity_fast(recent_5s)
        
        # Check for large red candle (State 4: Reversal)
        candle_change_percent = ((current_close - current_open) / current_open * 100) if current_open > 0 else 0
        if candle_change_percent < -5:  # Large red candle erasing gains
            # Check if it's erasing recent gains
            if len(candles_5s) >= 3:
                prev_gains = self._check_recent_gains_erased(candles_5s[-3:])
                if prev_gains:
                    return 4
        
        # State 3 (Peak Distribution): Look for exhaustion patterns over multiple candles
        exhaustion_pattern = self._detect_exhaustion_pattern(candles_5s, candles_1m)
        if exhaustion_pattern:
            return 3
        
        # State 2 (Acceleration): High Price Velocity for several consecutive candles
        if price_velocity > 2.0:  # Strong positive momentum
            consecutive_green = self._count_consecutive_green_candles(recent_5s)
            if consecutive_green >= 2:  # At least 2 consecutive green candles
                return 2
        
        # State 1 (Ignition): Broader initial pump phase detection (1-3 minutes window)
        ignition_detected = self._detect_ignition_phase(candles_5s, candles_1m, token_data)
        if ignition_detected:
            return 1
        
        # Default to State 0 (Quiescent)
        return 0
    
    def _calculate_volume_acceleration_fast(self, candles_5s):
        """Calculate volume acceleration from 5s candles"""
        if not candles_5s or len(candles_5s) < 3:
            return 0
        
        # Get last 3 candles for acceleration calculation
        recent_candles = candles_5s[-3:]
        volumes = [float(c.get("volume_usd", 0)) for c in recent_candles]
        
        if len(volumes) < 3:
            return 0
        
        # Calculate volume momentum changes
        vol_momentum_1 = volumes[-2] - volumes[-3]
        vol_momentum_2 = volumes[-1] - volumes[-2]
        
        # Volume acceleration is change in volume momentum
        return vol_momentum_2 - vol_momentum_1
    
    def _calculate_buy_pressure_imbalance(self, candle):
        """Calculate buy pressure imbalance from buy/sell counts"""
        buy_count = candle.get("buy_count", 0)
        sell_count = candle.get("sell_count", 1)  # Avoid division by zero
        
        if sell_count == 0:
            return buy_count if buy_count > 0 else 0
        
        return buy_count / sell_count
    
    def _calculate_price_velocity_fast(self, recent_candles):
        """Calculate price velocity (momentum) over recent 5s candles"""
        if not recent_candles or len(recent_candles) < 2:
            return 0
        
        prices = [float(c.get("close_usd", 0)) for c in recent_candles]
        if not all(p > 0 for p in prices):
            return 0
        
        # Calculate average price change per candle
        total_change = 0
        valid_changes = 0
        
        for i in range(1, len(prices)):
            if prices[i-1] > 0:
                change_percent = ((prices[i] - prices[i-1]) / prices[i-1]) * 100
                total_change += change_percent
                valid_changes += 1
        
        return total_change / valid_changes if valid_changes > 0 else 0
    
    def _check_recent_gains_erased(self, recent_candles):
        """Check if recent gains are being erased by current red candle"""
        if len(recent_candles) < 2:
            return False
        
        # Check if previous candles had gains
        prev_gains = 0
        for candle in recent_candles[:-1]:
            open_price = float(candle.get("open_usd", 0))
            close_price = float(candle.get("close_usd", 0))
            if open_price > 0:
                gain = ((close_price - open_price) / open_price) * 100
                prev_gains += max(0, gain)  # Only count positive gains
        
        # Check current candle loss
        current_candle = recent_candles[-1]
        current_open = float(current_candle.get("open_usd", 0))
        current_close = float(current_candle.get("close_usd", 0))
        
        if current_open > 0:
            current_loss = abs(((current_close - current_open) / current_open) * 100)
            # If current loss is significant compared to recent gains
            return current_loss > (prev_gains * 0.5)  # Erasing >50% of recent gains
        
        return False
    
    def _calculate_upper_wick_ratio(self, candle):
        """Calculate upper wick ratio as indicator of selling pressure"""
        high_price = float(candle.get("high_usd", 0))
        open_price = float(candle.get("open_usd", 0))
        close_price = float(candle.get("close_usd", 0))
        
        if high_price <= 0:
            return 0
        
        body_top = max(open_price, close_price)
        upper_wick = high_price - body_top
        candle_range = high_price - min(open_price, close_price)
        
        return upper_wick / candle_range if candle_range > 0 else 0
    
    def _count_consecutive_green_candles(self, candles):
        """Count consecutive green (positive) candles from the end"""
        if not candles:
            return 0
        
        consecutive = 0
        for candle in reversed(candles):
            open_price = float(candle.get("open_usd", 0))
            close_price = float(candle.get("close_usd", 0))
            
            if close_price > open_price:  # Green candle
                consecutive += 1
            else:
                break
        
        return consecutive
    
    def _is_first_volume_spike(self, candles_5s, token_data):
        """Check if this is the first significant volume spike for the token"""
        if not candles_5s or len(candles_5s) < 2:
            return True  # Assume first spike if insufficient data
        
        current_volume = float(candles_5s[-1].get("volume_usd", 0))
        
        # Check if we've seen similar volume before
        volume_threshold = current_volume * 0.7  # 70% of current volume
        
        for candle in candles_5s[:-1]:  # Exclude current candle
            prev_volume = float(candle.get("volume_usd", 0))
            if prev_volume >= volume_threshold:
                return False  # We've seen similar volume before
        
        return True  # This is the first significant volume spike
    
    def _detect_ignition_phase(self, candles_5s, candles_1m, token_data):
        """
        Detect ignition phase - broader initial pump phase (1-3 minutes window)
        Instead of looking for the absolute first spike, define it as the initial phase of a pump
        """
        if not candles_5s or len(candles_5s) < 12:  # Need at least 1 minute of 5s data
            return False
        
        # Check if token is in its first 1-3 minutes of life with significant activity
        token_age_minutes = len(candles_5s) * 5 / 60  # Convert 5s candles to minutes
        if token_age_minutes > 3:  # Only consider tokens in first 3 minutes
            return False
        
        # Look for significant jump in both price and volume over the recent period
        recent_candles = candles_5s[-12:]  # Last 1 minute of data
        
        # Calculate volume surge
        current_volume = float(candles_5s[-1].get("volume_usd", 0))
        avg_early_volume = sum(float(c.get("volume_usd", 0)) for c in candles_5s[:6]) / 6 if len(candles_5s) >= 6 else 0
        
        volume_surge = current_volume / max(avg_early_volume, 1) if avg_early_volume > 0 else current_volume
        
        # Calculate price momentum over recent period
        if len(recent_candles) >= 2:
            start_price = float(recent_candles[0].get("close_usd", 0))
            end_price = float(recent_candles[-1].get("close_usd", 0))
            price_change_percent = ((end_price - start_price) / start_price * 100) if start_price > 0 else 0
        else:
            price_change_percent = 0
        
        # Ignition criteria: significant volume surge AND price increase
        if volume_surge > 2.0 and price_change_percent > 10:  # 2x volume surge + 10% price increase
            return True
        
        # Alternative: Strong buy pressure with volume acceleration
        buy_pressure_imbalance = self._calculate_buy_pressure_imbalance(candles_5s[-1])
        volume_acceleration = self._calculate_volume_acceleration_fast(candles_5s)
        
        if buy_pressure_imbalance > 1.5 and volume_acceleration > 0 and price_change_percent > 5:
            return True
        
        return False
    
    def _detect_exhaustion_pattern(self, candles_5s, candles_1m):
        """
        Detect exhaustion pattern - look for patterns of exhaustion over multiple candles
        Instead of a single candle, look for 2-3 consecutive candles with long upper wicks
        or clear divergence where price makes new high but volume does not
        """
        if not candles_5s or len(candles_5s) < 6:
            return False
        
        recent_candles = candles_5s[-6:]  # Last 30 seconds of data
        
        # Pattern 1: Multiple candles with long upper wicks (selling pressure)
        upper_wick_count = 0
        for candle in recent_candles[-3:]:  # Check last 3 candles
            upper_wick_ratio = self._calculate_upper_wick_ratio(candle)
            if upper_wick_ratio > 0.25:  # Upper wick > 25% of candle range
                upper_wick_count += 1
        
        if upper_wick_count >= 2:  # At least 2 out of 3 candles have long upper wicks
            return True
        
        # Pattern 2: Price-Volume divergence (price makes new high but volume doesn't)
        if len(candles_5s) >= 10:
            # Check if recent price made new high
            recent_highs = [float(c.get("high_usd", 0)) for c in candles_5s[-5:]]
            earlier_highs = [float(c.get("high_usd", 0)) for c in candles_5s[-10:-5]]
            
            recent_max_price = max(recent_highs) if recent_highs else 0
            earlier_max_price = max(earlier_highs) if earlier_highs else 0
            
            # Check if recent volume is declining despite price increase
            recent_volumes = [float(c.get("volume_usd", 0)) for c in candles_5s[-5:]]
            earlier_volumes = [float(c.get("volume_usd", 0)) for c in candles_5s[-10:-5]]
            
            recent_avg_volume = sum(recent_volumes) / len(recent_volumes) if recent_volumes else 0
            earlier_avg_volume = sum(earlier_volumes) / len(earlier_volumes) if earlier_volumes else 0
            
            # Divergence: price higher but volume lower
            if (recent_max_price > earlier_max_price and 
                recent_avg_volume < earlier_avg_volume * 0.7):  # Volume dropped by 30%
                return True
        
        # Pattern 3: Stalling price action with declining volume acceleration
        price_velocity = self._calculate_price_velocity_fast(recent_candles)
        volume_acceleration = self._calculate_volume_acceleration_fast(candles_5s)
        
        if price_velocity <= 1.0 and volume_acceleration < -0.5:  # Stalling price + declining volume
            return True
        
        return False
    
    def _detect_selling_pressure(self, candles_1m, candles_5s):
        """Detect selling pressure through candlestick patterns (legacy function)"""
        if not candles_1m or len(candles_1m) < 3:
            return False
        
        # Check for long upper wicks in recent candles
        upper_wick_count = 0
        for candle in candles_1m[-3:]:
            high_price = float(candle["high_usd"])
            open_price = float(candle["open_usd"])
            close_price = float(candle["close_usd"])
            
            body_top = max(open_price, close_price)
            upper_wick = high_price - body_top
            body_size = abs(close_price - open_price)
            
            # Long upper wick indicates selling pressure
            if body_size > 0 and upper_wick >= body_size * 2:
                upper_wick_count += 1
            elif upper_wick > (high_price * 0.02):  # Wick >2% of price
                upper_wick_count += 1
        
        return upper_wick_count >= 2
    
    def _calculate_price_acceleration(self, candles_1m, candles_5s):
        """Calculate price acceleration (legacy function)"""
        if candles_5s and len(candles_5s) >= 6:
            # Use 5s data for more precise acceleration
            prices = [float(c["close_usd"]) for c in candles_5s[-6:]]
        elif candles_1m and len(candles_1m) >= 3:
            # Fallback to 1m data
            prices = [float(c["close_usd"]) for c in candles_1m[-3:]]
        else:
            return 0
        
        if len(prices) < 3:
            return 0
        
        # Calculate momentum changes
        momentum_1 = (prices[-2] - prices[-3]) / prices[-3] if prices[-3] > 0 else 0
        momentum_2 = (prices[-1] - prices[-2]) / prices[-2] if prices[-2] > 0 else 0
        
        # Acceleration is change in momentum
        return momentum_2 - momentum_1
    
    def _generate_real_time_training_sample(self, mint, candles_1m, candles_5s, current_state):
        """Generate training sample with current state as label and enhanced sample weights"""
        if len(candles_5s) < 12:  # Need minimum data for features
            return
        
        features = self.extract_features(candles_5s, candles_1m, mint)
        if not features:
            return
        
        # Enhanced sample weights - much higher weights for rare but critical states
        # This forces the model to pay significantly more attention to these examples
        state_weights = {
            0: 1.0,    # Quiescent - baseline weight (most common)
            1: 8.0,    # Ignition - critical entry signal, very high weight
            2: 3.0,    # Acceleration - important momentum phase
            3: 7.0,    # Peak Distribution - critical exit signal, very high weight  
            4: 10.0    # Reversal - urgent exit signal, highest weight
        }
        
        sample_weight = state_weights.get(current_state, 1.0)
        
        # Store training sample with current state as label and enhanced weight
        self.memory["tokens"][mint]["training_samples"].append({
            'features': features,
            'label': current_state,  # Current state, not future prediction
            'weight': sample_weight,  # Enhanced weight based on state criticality
            'timestamp': time.time()
        })
        
        # Limit training samples per token
        if len(self.memory["tokens"][mint]["training_samples"]) > 100:
            self.memory["tokens"][mint]["training_samples"] = \
                self.memory["tokens"][mint]["training_samples"][-100:]
    
    def _generate_real_time_training_sample_pump_memory(self, mint, candles_1m, candles_5s, current_state):
        """Generate training sample with current state as label and enhanced weights for pump memory tokens"""
        if len(candles_5s) < 12:  # Need minimum data for features
            return
        
        features = self.extract_features(candles_5s, candles_1m, mint)
        if not features:
            return
        
        # Enhanced sample weights for pump memory - same as main memory
        state_weights = {
            0: 1.0,    # Quiescent - baseline weight
            1: 8.0,    # Ignition - critical entry signal, very high weight
            2: 3.0,    # Acceleration - important momentum phase
            3: 7.0,    # Peak Distribution - critical exit signal, very high weight  
            4: 10.0    # Reversal - urgent exit signal, highest weight
        }
        
        sample_weight = state_weights.get(current_state, 1.0)
        
        # Store training sample with current state as label and enhanced weight
        self.pump_memory["tokens"][mint]["training_samples"].append({
            'features': features,
            'label': current_state,  # Current state, not future prediction
            'weight': sample_weight,  # Enhanced weight based on state criticality
            'timestamp': time.time()
        })
        
        # Limit training samples per token
        if len(self.pump_memory["tokens"][mint]["training_samples"]) > 100:
            self.pump_memory["tokens"][mint]["training_samples"] = \
                self.pump_memory["tokens"][mint]["training_samples"][-100:]

    def _perform_retroactive_labeling(self, mint, old_state, new_state, candles_5s, candles_1m):
        """
        Retroactively label previous candles based on confirmed state transitions.
        
        Logic:
        - When State 4 (Downfall) is confirmed, the preceding candles were likely State 3 (Peak Distribution)
        - When State 2 (Momentum Spike) is confirmed, the candles that kicked it off were likely State 1 (Ignition)
        """
        try:
            token_data = self.memory["tokens"][mint]
            training_samples = token_data.get("training_samples", [])
            
            if not training_samples or len(candles_5s) < 6:
                return
            
            # Case 1: Entering State 4 (Downfall) - retroactively label preceding candles as State 3
            if new_state == 4 and old_state != 4:
                self._retroactively_label_peak_distribution(mint, training_samples, candles_5s, candles_1m)
            
            # Case 2: Entering State 2 (Momentum Spike) - retroactively label preceding candles as State 1
            elif new_state == 2 and old_state != 2:
                self._retroactively_label_ignition(mint, training_samples, candles_5s, candles_1m)
                
        except Exception as e:
            self.logger.error(f"Error in retroactive labeling for {mint[:8]}: {e}")

    def _perform_retroactive_labeling_pump_memory(self, mint, old_state, new_state, candles_5s, candles_1m):
        """
        Retroactively label previous candles for pump memory tokens.
        """
        try:
            token_data = self.pump_memory["tokens"][mint]
            training_samples = token_data.get("training_samples", [])
            
            if not training_samples or len(candles_5s) < 6:
                return
            
            # Case 1: Entering State 4 (Downfall) - retroactively label preceding candles as State 3
            if new_state == 4 and old_state != 4:
                self._retroactively_label_peak_distribution_pump_memory(mint, training_samples, candles_5s, candles_1m)
            
            # Case 2: Entering State 2 (Momentum Spike) - retroactively label preceding candles as State 1
            elif new_state == 2 and old_state != 2:
                self._retroactively_label_ignition_pump_memory(mint, training_samples, candles_5s, candles_1m)
                
        except Exception as e:
            self.logger.error(f"Error in pump memory retroactive labeling for {mint[:8]}: {e}")

    def _retroactively_label_peak_distribution(self, mint, training_samples, candles_5s, candles_1m):
        """
        When State 4 (Downfall) is confirmed, retroactively label the preceding 2-4 samples as State 3.
        This gives us perfect examples of what Peak Distribution looks like.
        """
        if len(training_samples) < 2:
            return
        
        # Look back 2-4 samples (10-20 seconds of 5s candles) to find Peak Distribution period
        lookback_samples = min(4, len(training_samples))
        
        for i in range(1, lookback_samples + 1):
            sample_idx = -i - 1  # Go backwards from current
            if abs(sample_idx) <= len(training_samples):
                sample = training_samples[sample_idx]
                
                # Only relabel if it was previously labeled as State 0, 1, or 2
                if sample['label'] in [0, 1, 2]:
                    sample['label'] = 3  # Peak Distribution
                    sample['weight'] = 2.0  # Higher weight for retroactively confirmed samples
                    
                    self.logger.debug(f"Retroactively labeled sample as State 3 (Peak Distribution) for {mint[:8]}")

    def _retroactively_label_ignition(self, mint, training_samples, candles_5s, candles_1m):
        """
        When State 2 (Momentum Spike) is confirmed, retroactively label the preceding 1-3 samples as State 1.
        This gives us perfect examples of what Ignition looks like.
        """
        if len(training_samples) < 2:
            return
        
        # Look back 1-3 samples (5-15 seconds of 5s candles) to find Ignition period
        lookback_samples = min(3, len(training_samples))
        
        for i in range(1, lookback_samples + 1):
            sample_idx = -i - 1  # Go backwards from current
            if abs(sample_idx) <= len(training_samples):
                sample = training_samples[sample_idx]
                
                # Only relabel if it was previously labeled as State 0
                if sample['label'] == 0:
                    sample['label'] = 1  # Ignition
                    sample['weight'] = 2.0  # Higher weight for retroactively confirmed samples
                    
                    self.logger.debug(f"Retroactively labeled sample as State 1 (Ignition) for {mint[:8]}")

    def _retroactively_label_peak_distribution_pump_memory(self, mint, training_samples, candles_5s, candles_1m):
        """Pump memory version of peak distribution retroactive labeling"""
        if len(training_samples) < 2:
            return
        
        lookback_samples = min(4, len(training_samples))
        
        for i in range(1, lookback_samples + 1):
            sample_idx = -i - 1
            if abs(sample_idx) <= len(training_samples):
                sample = training_samples[sample_idx]
                
                if sample['label'] in [0, 1, 2]:
                    sample['label'] = 3
                    sample['weight'] = 2.0
                    
                    self.logger.debug(f"Pump memory: Retroactively labeled sample as State 3 for {mint[:8]}")

    def _retroactively_label_ignition_pump_memory(self, mint, training_samples, candles_5s, candles_1m):
        """Pump memory version of ignition retroactive labeling"""
        if len(training_samples) < 2:
            return
        
        lookback_samples = min(3, len(training_samples))
        
        for i in range(1, lookback_samples + 1):
            sample_idx = -i - 1
            if abs(sample_idx) <= len(training_samples):
                sample = training_samples[sample_idx]
                
                if sample['label'] == 0:
                    sample['label'] = 1
                    sample['weight'] = 2.0
                    
                    self.logger.debug(f"Pump memory: Retroactively labeled sample as State 1 for {mint[:8]}")

    def prepare_training_data(self):
        """Prepare training data with undersampling for balanced curriculum learning"""
        all_samples = []
        
        # Collect all real-time state training samples
        for mint, token_data in self.memory["tokens"].items():
            training_samples = token_data.get("training_samples", [])
            for sample in training_samples:
                all_samples.append({
                    'features': sample['features'],
                    'label': sample['label'],
                    'weight': sample['weight']
                })
        
        # Also collect from pump memory
        for mint, token_data in self.pump_memory["tokens"].items():
            training_samples = token_data.get("training_samples", [])
            for sample in training_samples:
                all_samples.append({
                    'features': sample['features'],
                    'label': sample['label'],
                    'weight': sample['weight']
                })
        
        if not all_samples:
            self.logger.warning("Could not generate any training samples for state classification.")
            return np.array([]), np.array([]), np.array([])
        
        # Separate samples by state (0-4)
        state_samples = {i: [] for i in range(5)}
        for sample in all_samples:
            label = sample['label']
            if 0 <= label <= 4:
                state_samples[label].append(sample)
        
        # Count samples per state
        state_counts = {i: len(state_samples[i]) for i in range(5)}
        self.logger.info(f"Original state distribution: {state_counts}")
        
        # Check if we have samples for multiple states
        non_empty_states = [i for i in range(5) if state_counts[i] > 0]
        if len(non_empty_states) < 2:
            self.logger.warning("Need samples from at least 2 different states for training")
            return np.array([]), np.array([]), np.array([])
        
        # UNDERSAMPLING: Balance the dataset by limiting State 0 samples
        # Use all rare samples (States 1-4) but subsample State 0 for balanced curriculum
        selected_samples = []
        
        # Always include ALL samples from rare but critical states (1, 3, 4)
        for state in [1, 3, 4]:
            selected_samples.extend(state_samples[state])
        
        # Include all State 2 samples (acceleration phase)
        selected_samples.extend(state_samples[2])
        
        # Undersample State 0 to create more balanced training
        state_0_samples = state_samples[0]
        if len(state_0_samples) > 0:
            # Calculate target size for State 0 based on rare state counts
            rare_state_total = sum(len(state_samples[i]) for i in [1, 3, 4])
            
            if rare_state_total > 0:
                # Limit State 0 to 3x the total of rare states for better balance
                max_state_0 = max(rare_state_total * 3, 50)  # At least 50 samples
                
                if len(state_0_samples) > max_state_0:
                    # Randomly sample from State 0 to create balanced curriculum
                    import random
                    random.shuffle(state_0_samples)
                    selected_samples.extend(state_0_samples[:max_state_0])
                    self.logger.info(f"Undersampled State 0: {len(state_0_samples)} -> {max_state_0} samples")
                else:
                    selected_samples.extend(state_0_samples)
            else:
                # If no rare states, use all State 0 samples
                selected_samples.extend(state_0_samples)
        
        # Log final distribution
        final_state_counts = {i: 0 for i in range(5)}
        for sample in selected_samples:
            final_state_counts[sample['label']] += 1
        
        self.logger.info(f"Balanced state distribution: {final_state_counts}")
        
        # Convert to arrays
        X = np.array([s['features'] for s in selected_samples], dtype=np.float32)
        y = np.array([s['label'] for s in selected_samples], dtype=np.int8)
        sw = np.array([s['weight'] for s in selected_samples], dtype=np.float32)
        
        self.logger.info(f"Generated BALANCED training data: {len(X)} samples")
        for state in range(5):
            count = np.sum(y == state)
            state_names = ["Quiescent", "Ignition", "Acceleration", "Peak Distribution", "Reversal"]
            if count > 0:
                avg_weight = np.mean(sw[y == state])
                self.logger.info(f"  State {state} ({state_names[state]}): {count} samples (avg weight: {avg_weight:.1f})")
        
        return X, y, sw

    def train_model_sync(self):
        """Enhanced training with data validation, regularization, k-fold CV, and feature selection"""
        X, y, sw = self.prepare_training_data()
        if len(X) < 10:
            self.logger.warning(f"Not enough training data: {len(X)} samples (need at least 10)")
            time.sleep(60)
            return False
        
        # Critical data validation and cleaning
        X, y, sw = self._validate_and_clean_training_data(X, y, sw)
        if len(X) < 10:
            self.logger.warning("Insufficient data after validation and cleaning")
            return False
        
        # Check for multi-class data
        unique_classes = np.unique(y)
        num_classes = len(unique_classes)
        
        if len(y) < 20 or num_classes < 2:
            self.logger.warning(f"Not enough valid data or classes: {len(y)} samples, {num_classes} classes")
            return False
        
        # Log class distribution
        for class_id in unique_classes:
            count = np.sum(y == class_id)
            self.logger.info(f"Class {int(class_id)}: {count} samples")
        
        if num_classes < 3:
            self.logger.warning(f"Need at least 3 different states for meaningful training, got {num_classes}")
            return False
        
        try:
            # K-fold cross-validation for robust model evaluation
            cv_scores = self._perform_kfold_validation(X, y, sw, k=5)
            self.logger.info(f"K-fold CV scores: {cv_scores}")
            if cv_scores:
                self.logger.info(f"Mean CV accuracy: {np.mean(cv_scores):.3f} (+/- {np.std(cv_scores) * 2:.3f})")
            else:
                self.logger.warning("No valid CV scores obtained")
            
            # Train final model on full dataset
            Xt, Xv, yt, yv, wt, wv = train_test_split(X, y, sw, test_size=0.2, random_state=42, stratify=y)
            Xts = self.scaler.fit_transform(Xt)
            Xvs = self.scaler.transform(Xv)
            
        except Exception as e:
            self.logger.error(f"Data preparation failed: {e}")
            return False
        
        # Enhanced parameters with L1 (Lasso) and L2 (Ridge) regularization
        params = {
            'objective': 'multiclass',
            'num_class': 5,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': min(31, max(10, len(Xt) // 30)),
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'max_depth': 8,
            'min_data_in_leaf': max(3, len(Xt) // 100),
            'verbose': -1,
            # Enhanced regularization to reduce overfitting
            'reg_alpha': 0.3,      # L1 regularization (Lasso) - increased for feature selection
            'reg_lambda': 0.3,     # L2 regularization (Ridge) - increased for stability
            'min_gain_to_split': 0.1,  # Higher threshold to prevent overfitting
            'subsample': 0.8,
            'colsample_bytree': 0.7,   # Reduced to add more randomness
            'max_bin': 255,
            'min_child_samples': 20,   # Prevent overfitting on small samples
        }
        
        td = lgb.Dataset(Xts, label=yt, weight=wt)
        vd = lgb.Dataset(Xvs, label=yv, weight=wv, reference=td)
        
        max_rounds = min(100, max(20, len(Xt) // 10))
        
        self.model = lgb.train(
            params, 
            td, 
            num_boost_round=max_rounds,
            valid_sets=[vd],
            callbacks=[
                lgb.early_stopping(stopping_rounds=15),
                lgb.log_evaluation(period=0)
            ]
        )
        
        # Feature importance analysis and selection
        self._analyze_and_select_features(Xts, yt, wt)
        
        # Multi-class validation accuracy
        val_predictions = self.model.predict(Xvs)
        ypv = np.argmax(val_predictions, axis=1)
        
        state_names = ["Baseline", "Early Rise", "Momentum Spike", "Peak Distribution", "Downfall"]
        for state in range(5):
            state_mask = (yv == state)
            if np.sum(state_mask) > 0:
                acc = np.mean(ypv[state_mask] == state)
                self.logger.info(f"Validation {state_names[state]} accuracy: {acc:.3f}")
        
        overall_acc = np.mean(ypv == yv)
        self.logger.info(f"Overall validation accuracy: {overall_acc:.3f}")
        
        self.is_trained = True
        self.save_model()
        
        self.logger.info("REAL-TIME STATE model training completed with enhanced regularization!")
        self.logger.info(f"Total samples: {len(X)}, Classes: {num_classes}")
        
        return True
    
    def _validate_and_clean_training_data(self, X, y, sw):
        """Critical data validation and cleaning to prevent training errors"""
        try:
            # Convert to numpy arrays with proper dtypes
            X = np.asarray(X, dtype=np.float32)
            y = np.asarray(y, dtype=np.int8)
            sw = np.asarray(sw, dtype=np.float32)
            
            # Remove samples with invalid features
            valid_mask = np.all(np.isfinite(X), axis=1)
            if not np.all(valid_mask):
                invalid_count = np.sum(~valid_mask)
                self.logger.warning(f"Removing {invalid_count} samples with invalid features")
                X, y, sw = X[valid_mask], y[valid_mask], sw[valid_mask]
            
            # Remove samples with invalid labels
            valid_labels = (y >= 0) & (y <= 4)
            if not np.all(valid_labels):
                invalid_count = np.sum(~valid_labels)
                self.logger.warning(f"Removing {invalid_count} samples with invalid labels")
                X, y, sw = X[valid_labels], y[valid_labels], sw[valid_labels]
            
            # Remove samples with invalid weights
            valid_weights = np.isfinite(sw) & (sw > 0)
            if not np.all(valid_weights):
                invalid_count = np.sum(~valid_weights)
                self.logger.warning(f"Fixing {invalid_count} samples with invalid weights")
                sw[~valid_weights] = 1.0
            
            # Check for duplicate samples (can cause overfitting)
            if len(X) > 100:  # Only check for large datasets
                unique_indices = np.unique(X, axis=0, return_index=True)[1]
                if len(unique_indices) < len(X):
                    duplicate_count = len(X) - len(unique_indices)
                    self.logger.info(f"Removing {duplicate_count} duplicate samples")
                    X, y, sw = X[unique_indices], y[unique_indices], sw[unique_indices]
            
            self.logger.info(f"Data validation complete: {len(X)} clean samples")
            return X, y, sw
            
        except Exception as e:
            self.logger.error(f"Critical error in data validation: {e}")
            return np.array([]), np.array([]), np.array([])
    
    def _perform_kfold_validation(self, X, y, sw, k=5):
        """Perform k-fold cross-validation for robust model evaluation"""
        try:
            # Check minimum samples per class to determine appropriate k
            unique_classes, class_counts = np.unique(y, return_counts=True)
            min_class_count = np.min(class_counts)
            
            # Adjust k if necessary to ensure each fold has at least one sample per class
            if min_class_count < k:
                k = max(2, min_class_count)  # Use at least 2 folds, but not more than min class count
                self.logger.warning(f"Reduced k-fold splits to {k} due to insufficient samples in smallest class ({min_class_count} samples)")
            
            # Use StratifiedKFold to maintain class distribution
            skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
            cv_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                try:
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    w_train, w_val = sw[train_idx], sw[val_idx]
                    
                    # Scale features for this fold
                    fold_scaler = StandardScaler()
                    X_train_scaled = fold_scaler.fit_transform(X_train)
                    X_val_scaled = fold_scaler.transform(X_val)
                    
                    # Train model for this fold
                    params = {
                        'objective': 'multiclass',
                        'num_class': 5,
                        'metric': 'multi_logloss',
                        'boosting_type': 'gbdt',
                        'num_leaves': 15,
                        'learning_rate': 0.1,
                        'verbose': -1,
                        'reg_alpha': 0.3,
                        'reg_lambda': 0.3,
                        'min_gain_to_split': 0.1,
                        'subsample': 0.8,
                        'colsample_bytree': 0.7,
                    }
                    
                    train_data = lgb.Dataset(X_train_scaled, label=y_train, weight=w_train)
                    fold_model = lgb.train(params, train_data, num_boost_round=50, callbacks=[lgb.log_evaluation(0)])
                    
                    # Evaluate on validation set
                    val_pred = fold_model.predict(X_val_scaled)
                    val_pred_classes = np.argmax(val_pred, axis=1)
                    fold_accuracy = np.mean(val_pred_classes == y_val)
                    cv_scores.append(fold_accuracy)
                    
                    self.logger.info(f"Fold {fold + 1} accuracy: {fold_accuracy:.3f}")
                    
                except Exception as e:
                    self.logger.error(f"Error in fold {fold + 1}: {e}")
                    continue
            
            return cv_scores
            
        except Exception as e:
            self.logger.error(f"Error in k-fold validation: {e}")
            return []
    
    def _analyze_and_select_features(self, X, y, w):
        """Analyze feature importance and select most important features"""
        try:
            if not hasattr(self, 'model') or self.model is None:
                return
            
            # Get feature importance from the trained model
            feature_importance = self.model.feature_importance(importance_type='gain')
            
            # Log top features
            top_indices = np.argsort(feature_importance)[::-1][:20]  # Top 20 features
            self.logger.info("Top 20 most important features:")
            for i, idx in enumerate(top_indices):
                self.logger.info(f"  {i+1}. Feature {idx}: importance {feature_importance[idx]:.3f}")
            
            # Store feature importance for potential feature selection
            self.feature_importance = feature_importance
            
            # Identify low-importance features (bottom 25%)
            importance_threshold = np.percentile(feature_importance, 25)
            low_importance_features = np.sum(feature_importance < importance_threshold)
            
            if low_importance_features > 0:
                self.logger.info(f"Identified {low_importance_features} low-importance features (< {importance_threshold:.3f})")
                self.logger.info("Consider feature selection for future training iterations")
            
            # Store feature selection info for next training cycle
            self.feature_selection_threshold = importance_threshold
            
        except Exception as e:
            self.logger.error(f"Error in feature analysis: {e}")
    
    def _select_important_features(self, X, threshold_percentile=25):
        """Select features based on importance threshold"""
        try:
            if not hasattr(self, 'feature_importance'):
                return X  # Return original features if no importance data
            
            # Select features above threshold
            importance_threshold = np.percentile(self.feature_importance, threshold_percentile)
            selected_features = self.feature_importance >= importance_threshold
            
            if np.sum(selected_features) < 10:  # Keep minimum 10 features
                # Select top 10 features instead
                top_indices = np.argsort(self.feature_importance)[::-1][:10]
                selected_features = np.zeros_like(self.feature_importance, dtype=bool)
                selected_features[top_indices] = True
            
            selected_count = np.sum(selected_features)
            total_count = len(selected_features)
            
            self.logger.info(f"Feature selection: using {selected_count}/{total_count} features")
            
            return X[:, selected_features]
            
        except Exception as e:
            self.logger.error(f"Error in feature selection: {e}")
            return X
    

    
    def evaluate_model_performance(self):
        """Evaluate model performance on validation data for multi-class state classification"""
        if not self.is_trained:
            self.logger.warning("Model not trained yet")
            return
        
        # Get fresh validation data
        X, y, sw = self.prepare_training_data()
        if len(X) < 20:
            self.logger.warning("Not enough data for evaluation")
            return
        
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report, confusion_matrix
        
        try:
            _, Xv, _, yv, _, _ = train_test_split(X, y, sw, test_size=0.2, random_state=42, stratify=y)
            Xvs = self.scaler.transform(Xv)
            
            # Get predictions for multi-class
            y_pred_proba = self.model.predict(Xvs)
            y_pred = np.argmax(y_pred_proba, axis=1)
            
            state_names = ["Baseline", "Early Rise", "Momentum Spike", "Peak Distribution", "Downfall"]
            
            self.logger.info("=== REAL-TIME STATE MODEL EVALUATION ===")
            
            # Overall accuracy
            overall_acc = np.mean(y_pred == yv)
            self.logger.info(f"Overall Accuracy: {overall_acc:.3f}")
            
            # Per-class accuracy
            for state in range(5):
                state_mask = (yv == state)
                if np.sum(state_mask) > 0:
                    acc = np.mean(y_pred[state_mask] == state)
                    count = np.sum(state_mask)
                    self.logger.info(f"{state_names[state]} accuracy: {acc:.3f} ({count} samples)")
            
            # Classification report
            present_classes = sorted(np.unique(yv))
            target_names = [state_names[i] for i in present_classes]
            report = classification_report(yv, y_pred, labels=present_classes, target_names=target_names, zero_division=0)
            self.logger.info(f"Classification Report:\n{report}")
            
            # Confusion matrix
            cm = confusion_matrix(yv, y_pred, labels=present_classes)
            self.logger.info(f"Confusion Matrix:\n{cm}")
            
        except Exception as e:
            self.logger.error(f"Model evaluation failed: {e}")
    
    async def predict(self, mint, ohlcv_data):
        """Predict current state probabilities using OHLCV data - REAL-TIME STATE IDENTIFICATION"""
        if not self.is_trained:
            self.logger.warning(f"ERROR: Model not trained for prediction of {mint[:20]}...")
            return None
        
        # Try to use provided OHLCV data first, fallback to memory
        if ohlcv_data:
            candles_1m = ohlcv_data.get("ohlcv", [])
            candles_5s = ohlcv_data.get("ohlcv_5s", [])
        else:
            token_data = self.memory["tokens"].get(mint, {})
            candles_1m = token_data.get("candles_1m", [])
            candles_5s = token_data.get("candles_5s", [])
        
        token_state = self.memory["token_states"].get(mint, "new")
        
        # Skip inactive tokens only if they're in our memory and marked inactive
        if token_state == "inactive" and mint in self.memory["tokens"]:
            return None
        
        # Need minimum history for state identification
        if len(candles_5s) < 12:
            return None
        
        features = self.extract_features(candles_5s, candles_1m, mint)
        if features is None:
            return None
        
        try:
            features_scaled = self.scaler.transform([features])
            state_probabilities = self.model.predict(features_scaled)[0]
            
            # Return probabilities for all 5 states
            predicted_state = np.argmax(state_probabilities)
            confidence = state_probabilities[predicted_state]
            
            state_names = ["Baseline", "Early Rise", "Momentum Spike", "Peak Distribution", "Downfall"]
            
            # Log significant state predictions
            if predicted_state > 0 and confidence > 0.6:
                self._write_prediction_log(confidence, 0.6, mint, f"State_{predicted_state}_{state_names[predicted_state]}", len(candles_1m))
            
            return {
                'state_probabilities': state_probabilities.tolist(),
                'predicted_state': int(predicted_state),
                'confidence': float(confidence),
                'state_name': state_names[predicted_state]
            }
        except Exception as e:
            self.logger.error(f"State prediction error for {mint}: {e}")
            return None
            
    async def scan_tokens(self):
        """Smart file scanning - only processes files that have been modified"""
        await self.update_active_contracts(use_async=True)
        self.cleanup_inactive_tokens()
        tf = glob.glob("tokenOHLCV/*_ohlcv.json")
        files_to_process = []
        
        for fp in tf:
            try:
                mint = os.path.basename(fp).replace('_ohlcv.json','')
                if not self.active_contracts or mint in self.active_contracts:
                    state = self.memory["token_states"].get(mint,"new")
                    if state in ["active","new","monitoring"]:
                        # Compare file's last modified time to stored record
                        current_mtime = os.path.getmtime(fp)
                        last_mtime = self.token_file_state.get(mint, {}).get('mtime', 0)
                        
                        # Only add to processing queue if file has actually been updated
                        if current_mtime > last_mtime:
                            files_to_process.append(fp)
                            # Update modification time record
                            if mint not in self.token_file_state:
                                self.token_file_state[mint] = {}
                            self.token_file_state[mint]['mtime'] = current_mtime
            except:
                continue
        
        # Process only files that have changed, saving time and resources
        tasks = []
        for fp in files_to_process:
            tasks.append(self.process_file(fp))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def process_file(self, file_path):
        """Enhanced file processing with critical data validation to prevent corruption errors"""
        try:
            mint = os.path.basename(file_path).replace('_ohlcv.json', '').replace('_5s_ohlcv.json', '')
            
            # Use token_file_state record to see where we left off
            processed_candles = self.token_file_state.get(mint, {}).get('processed_candles', 0)
            
            # Load and validate 1-minute data
            od_1m = {}
            if os.path.exists(f"tokenOHLCV/{mint}_ohlcv.json"):
                try:
                    with open(f"tokenOHLCV/{mint}_ohlcv.json", 'r') as f:
                        od_1m = json.load(f)
                    # Critical validation - ensure loaded data is a dictionary
                    if not isinstance(od_1m, dict):
                        self.logger.error(f"Critical error: 1m data for {mint} is not a dict: {type(od_1m)}")
                        return
                except (json.JSONDecodeError, Exception) as e:
                    self.logger.error(f"Critical error loading 1m data for {mint}: {e}")
                    return
            
            # Load and validate 5-second data
            od_5s = {}
            if os.path.exists(f"tokenOHLCV/{mint}_5s_ohlcv.json"):
                try:
                    with open(f"tokenOHLCV/{mint}_5s_ohlcv.json", 'r') as f:
                        od_5s = json.load(f)
                    # Critical validation - ensure loaded data is a dictionary
                    if not isinstance(od_5s, dict):
                        self.logger.error(f"Critical error: 5s data for {mint} is not a dict: {type(od_5s)}")
                        return
                except (json.JSONDecodeError, Exception) as e:
                    self.logger.error(f"Critical error loading 5s data for {mint}: {e}")
                    return
            
            # Extract and validate candle arrays
            all_candles_1m = od_1m.get('ohlcv', [])
            all_candles_5s = od_5s.get('ohlcv', [])
            
            # Critical validation - ensure candle data is a list
            if not isinstance(all_candles_1m, list):
                self.logger.error(f"Critical error: 1m candles for {mint} is not a list: {type(all_candles_1m)}")
                return
            if not isinstance(all_candles_5s, list):
                self.logger.error(f"Critical error: 5s candles for {mint} is not a list: {type(all_candles_5s)}")
                return
            
            # Only process new candles that have been added since last scan
            if len(all_candles_1m) <= processed_candles:
                return  # No new data to process
            
            # Update processed candle count to current total
            if mint not in self.token_file_state:
                self.token_file_state[mint] = {}
            self.token_file_state[mint]['processed_candles'] = len(all_candles_1m)
            
            # For token data update, provide recent context window (not just new candles)
            CONTEXT_WINDOW = 50
            context_start_1m = max(0, len(all_candles_1m) - CONTEXT_WINDOW)
            context_start_5s = max(0, len(all_candles_5s) - CONTEXT_WINDOW * 12)
            
            # Validate candle data before processing
            validated_candles_1m = self._validate_candle_data(all_candles_1m[context_start_1m:], "1m")
            validated_candles_5s = self._validate_candle_data(all_candles_5s[context_start_5s:] if all_candles_5s else [], "5s")
            
            combined_data = {
                'mint': mint,
                'ohlcv': validated_candles_1m,
                'ohlcv_5s': validated_candles_5s
            }
            
            await self.update_token_data(mint, combined_data)
            
            if self.is_trained:
                p = await self.predict(mint, combined_data)
                if p is not None and isinstance(p, dict) and p.get('confidence', 0) > 0.7:
                    state = self.memory["token_states"].get(mint, "new")
                    confidence = p.get('confidence', 0)
                    state_name = p.get('state_name', 'Unknown')
                    self.logger.info(f"PUMP SIGNAL {mint[:8]}... [{self.snapshot_interval}s]: {confidence:.3f} [{state_name}] [{state}]")
                    
        except Exception as e:
            self.logger.error(f"Critical error processing {file_path}: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
        finally:
            od_1m = None
            od_5s = None
            combined_data = None

    def analyze_performance_and_recommend_parameters(self):
        """Analyze model performance and trading data to recommend parameter adjustments"""
        try:
            self.logger.info("Analyzing performance and generating parameter recommendations...")
            
            # Initialize recommendations with current defaults
            recommendations = {
                'MIN_PRICE_CHANGE': 0.05,
                'TRADE_COOLDOWN': 20,
                'pre_pump_threshold': 1.0,
                'volume_spike_threshold': 2.0,
                'stop_loss': 10.0,
                'take_profit': 50.0,
                'volume_5s_threshold': 200,
                'volume_1m_threshold': 500,
                'buy_sell_ratio_5s': 1.2,
                'buy_sell_ratio_1m': 1.6,
                'confidence': 0.5,
                'reasoning': []
            }
            
            # Analyze trade learning data if available
            trade_performance = self._analyze_trade_learning_data()
            if trade_performance:
                recommendations.update(self._generate_trade_based_recommendations(trade_performance, recommendations))
            
            # Analyze model predictions vs actual outcomes
            model_performance = self._analyze_model_performance()
            if model_performance:
                recommendations.update(self._generate_model_based_recommendations(model_performance, recommendations))
            
            # Analyze volume patterns from successful vs failed trades
            volume_analysis = self._analyze_volume_patterns()
            if volume_analysis:
                recommendations.update(self._generate_volume_based_recommendations(volume_analysis, recommendations))
            
            # Store recommendations for retrieval by TokenAnalyzer
            self.parameter_recommendations = recommendations
            self.last_parameter_analysis = time.time()
            
            # Log key recommendations
            if recommendations['reasoning']:
                self.logger.info(f"Parameter recommendations (confidence: {recommendations['confidence']:.2f}):")
                for reason in recommendations['reasoning'][:3]:  # Show top 3 reasons
                    self.logger.info(f"  - {reason}")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error in parameter analysis: {e}")
            return {}

    def _analyze_trade_learning_data(self):
        """Analyze trade learning data to understand performance patterns"""
        try:
            if not os.path.exists('trade_learning.json'):
                return None
            
            trades = []
            with open('trade_learning.json', 'r') as f:
                for line in f:
                    try:
                        line = line.strip()
                        if not line or not line.startswith('{'):
                            continue
                        trade = json.loads(line)
                        trades.append(trade)
                    except:
                        continue
            
            if len(trades) < 10:  # Need minimum trades for analysis
                return None
            
            # Analyze recent trades (last 50)
            recent_trades = trades[-50:]
            
            performance = {
                'total_trades': len(recent_trades),
                'winning_trades': sum(1 for t in recent_trades if t.get('pnl_percent', 0) > 0),
                'losing_trades': sum(1 for t in recent_trades if t.get('pnl_percent', 0) <= 0),
                'avg_pnl': np.mean([t.get('pnl_percent', 0) for t in recent_trades]),
                'avg_hold_time': np.mean([t.get('trade_duration', 0) for t in recent_trades]),
                'win_rate': sum(1 for t in recent_trades if t.get('pnl_percent', 0) > 0) / len(recent_trades),
                'avg_winning_pnl': np.mean([t.get('pnl_percent', 0) for t in recent_trades if t.get('pnl_percent', 0) > 0]) if any(t.get('pnl_percent', 0) > 0 for t in recent_trades) else 0,
                'avg_losing_pnl': np.mean([t.get('pnl_percent', 0) for t in recent_trades if t.get('pnl_percent', 0) <= 0]) if any(t.get('pnl_percent', 0) <= 0 for t in recent_trades) else 0,
                'exit_reasons': {}
            }
            
            # Analyze exit reasons
            for trade in recent_trades:
                reason = trade.get('reason', 'unknown')
                if reason not in performance['exit_reasons']:
                    performance['exit_reasons'][reason] = {'count': 0, 'avg_pnl': 0, 'pnls': []}
                performance['exit_reasons'][reason]['count'] += 1
                performance['exit_reasons'][reason]['pnls'].append(trade.get('pnl_percent', 0))
            
            # Calculate averages for exit reasons
            for reason in performance['exit_reasons']:
                pnls = performance['exit_reasons'][reason]['pnls']
                performance['exit_reasons'][reason]['avg_pnl'] = np.mean(pnls)
            
            return performance
            
        except Exception as e:
            self.logger.error(f"Error analyzing trade learning data: {e}")
            return None

    def _generate_trade_based_recommendations(self, performance, current_recs):
        """Generate recommendations based on trade performance analysis"""
        recommendations = current_recs.copy()
        reasoning = recommendations['reasoning']
        
        try:
            # Adjust based on win rate
            if performance['win_rate'] < 0.4:  # Low win rate
                recommendations['pre_pump_threshold'] = min(3.0, current_recs['pre_pump_threshold'] * 1.2)
                recommendations['volume_spike_threshold'] = min(5.0, current_recs['volume_spike_threshold'] * 1.1)
                recommendations['volume_5s_threshold'] = min(500, current_recs['volume_5s_threshold'] * 1.3)
                reasoning.append(f"Low win rate ({performance['win_rate']:.2f}) - tightening entry criteria")
                recommendations['confidence'] = min(1.0, recommendations['confidence'] + 0.2)
            
            elif performance['win_rate'] > 0.7:  # High win rate, can be more aggressive
                recommendations['pre_pump_threshold'] = max(0.5, current_recs['pre_pump_threshold'] * 0.9)
                recommendations['volume_spike_threshold'] = max(1.5, current_recs['volume_spike_threshold'] * 0.95)
                reasoning.append(f"High win rate ({performance['win_rate']:.2f}) - loosening entry criteria")
                recommendations['confidence'] = min(1.0, recommendations['confidence'] + 0.1)
            
            # Adjust based on average PnL
            if performance['avg_pnl'] < -5:  # Losing money on average
                recommendations['stop_loss'] = max(5.0, current_recs['stop_loss'] * 0.8)  # Tighter stop loss
                recommendations['TRADE_COOLDOWN'] = min(60, current_recs['TRADE_COOLDOWN'] + 10)
                reasoning.append(f"Negative avg PnL ({performance['avg_pnl']:.1f}%) - tighter risk management")
                recommendations['confidence'] = min(1.0, recommendations['confidence'] + 0.3)
            
            # Analyze exit reasons for patterns
            exit_reasons = performance['exit_reasons']
            
            # If many trades exit due to stop loss with bad PnL, tighten stop loss
            if 'aSellSL' in exit_reasons and exit_reasons['aSellSL']['avg_pnl'] < -8:
                recommendations['stop_loss'] = max(5.0, current_recs['stop_loss'] * 0.9)
                reasoning.append("Stop loss exits showing large losses - tightening stop loss")
                recommendations['confidence'] = min(1.0, recommendations['confidence'] + 0.2)
            
            # If many trades exit due to volume decline, adjust volume thresholds
            volume_exit_reasons = [r for r in exit_reasons.keys() if 'Vol' in r or 'volume' in r.lower()]
            if volume_exit_reasons:
                total_volume_exits = sum(exit_reasons[r]['count'] for r in volume_exit_reasons)
                if total_volume_exits > performance['total_trades'] * 0.3:  # >30% of trades
                    recommendations['volume_5s_threshold'] = min(400, current_recs['volume_5s_threshold'] * 1.2)
                    recommendations['volume_1m_threshold'] = min(800, current_recs['volume_1m_threshold'] * 1.1)
                    reasoning.append("High volume-based exits - raising volume thresholds")
                    recommendations['confidence'] = min(1.0, recommendations['confidence'] + 0.15)
            
            # Adjust hold time based on performance
            if performance['avg_hold_time'] < 30 and performance['avg_pnl'] < 0:  # Quick losses
                recommendations['TRADE_COOLDOWN'] = min(45, current_recs['TRADE_COOLDOWN'] + 5)
                reasoning.append("Quick losing trades - increasing cooldown")
                recommendations['confidence'] = min(1.0, recommendations['confidence'] + 0.1)
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating trade-based recommendations: {e}")
            return current_recs

    def _analyze_model_performance(self):
        """Analyze model prediction accuracy vs actual outcomes"""
        try:
            # This would require tracking predictions vs outcomes over time
            # For now, return basic model stats if available
            if not self.is_trained:
                return None
            
            performance = {
                'model_trained': self.is_trained,
                'optimal_threshold': getattr(self, 'optimal_threshold', 0.3),
                'trade_model_trained': self.trade_is_trained
            }
            
            return performance
            
        except Exception as e:
            self.logger.error(f"Error analyzing model performance: {e}")
            return None

    def _generate_model_based_recommendations(self, performance, current_recs):
        """Generate recommendations based on model performance"""
        recommendations = current_recs.copy()
        reasoning = recommendations['reasoning']
        
        try:
            # Adjust thresholds based on model confidence
            if performance['optimal_threshold'] > 0.6:  # High threshold suggests model is conservative
                recommendations['pre_pump_threshold'] = max(0.8, current_recs['pre_pump_threshold'] * 0.95)
                reasoning.append(f"Model threshold high ({performance['optimal_threshold']:.2f}) - slightly loosening criteria")
                recommendations['confidence'] = min(1.0, recommendations['confidence'] + 0.1)
            
            elif performance['optimal_threshold'] < 0.3:  # Low threshold suggests model is aggressive
                recommendations['pre_pump_threshold'] = min(2.0, current_recs['pre_pump_threshold'] * 1.05)
                reasoning.append(f"Model threshold low ({performance['optimal_threshold']:.2f}) - slightly tightening criteria")
                recommendations['confidence'] = min(1.0, recommendations['confidence'] + 0.1)
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating model-based recommendations: {e}")
            return current_recs

    def _analyze_volume_patterns(self):
        """Analyze volume patterns from memory data"""
        try:
            volume_data = {
                'successful_pumps': [],
                'failed_tokens': [],
                'avg_successful_volume': 0,
                'avg_failed_volume': 0
            }
            
            # Analyze pump memory for successful patterns
            for mint, token_data in self.pump_memory["tokens"].items():
                candles_5s = token_data.get("candles_5s", [])
                if candles_5s and len(candles_5s) >= 3:
                    volumes = [float(c.get("volume_usd", 0)) for c in candles_5s[-3:]]
                    volume_data['successful_pumps'].extend(volumes)
            
            # Analyze regular memory for failed patterns
            for mint, token_data in self.memory["tokens"].items():
                if token_data.get("pump_status") == "dud":
                    candles_5s = token_data.get("candles_5s", [])
                    if candles_5s and len(candles_5s) >= 3:
                        volumes = [float(c.get("volume_usd", 0)) for c in candles_5s[-3:]]
                        volume_data['failed_tokens'].extend(volumes)
            
            if volume_data['successful_pumps'] and volume_data['failed_tokens']:
                volume_data['avg_successful_volume'] = np.mean(volume_data['successful_pumps'])
                volume_data['avg_failed_volume'] = np.mean(volume_data['failed_tokens'])
                return volume_data
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error analyzing volume patterns: {e}")
            return None

    def _generate_volume_based_recommendations(self, volume_analysis, current_recs):
        """Generate recommendations based on volume pattern analysis"""
        recommendations = current_recs.copy()
        reasoning = recommendations['reasoning']
        
        try:
            successful_vol = volume_analysis['avg_successful_volume']
            failed_vol = volume_analysis['avg_failed_volume']
            
            if successful_vol > failed_vol * 1.5:  # Successful pumps have significantly higher volume
                # Increase volume thresholds to filter out low-volume tokens
                new_5s_threshold = min(500, max(successful_vol * 0.7, current_recs['volume_5s_threshold']))
                new_1m_threshold = min(1000, max(successful_vol * 1.2, current_recs['volume_1m_threshold']))
                
                if new_5s_threshold > current_recs['volume_5s_threshold']:
                    recommendations['volume_5s_threshold'] = new_5s_threshold
                    reasoning.append(f"Successful pumps show higher volume ({successful_vol:.0f} vs {failed_vol:.0f}) - raising 5s threshold")
                    recommendations['confidence'] = min(1.0, recommendations['confidence'] + 0.2)
                
                if new_1m_threshold > current_recs['volume_1m_threshold']:
                    recommendations['volume_1m_threshold'] = new_1m_threshold
                    reasoning.append(f"Raising 1m volume threshold based on successful pump patterns")
                    recommendations['confidence'] = min(1.0, recommendations['confidence'] + 0.15)
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating volume-based recommendations: {e}")
            return current_recs

    def get_parameter_recommendations(self):
        """Get the latest parameter recommendations for TokenAnalyzer"""
        return self.parameter_recommendations.copy() if self.parameter_recommendations else {}
    
    async def run(self):
        self.logger.info(f"RichardML started with {self.snapshot_interval}s intervals")
        while True:
            try:
                await self.scan_tokens()
                self.tokens_processed+=1

                # Train trade model every 50 processed tokens
                if not self.trade_is_trained and self.tokens_processed % 50 == 0:
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(self.executor, self.train_trade_model_sync)
                elif self.trade_is_trained and self.tokens_processed % 50 == 0:
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(self.executor, self.train_trade_model_sync)
                
                # Analyze performance and recommend parameter adjustments every 75 tokens
                if self.tokens_processed % self.parameter_analysis_interval == 0:
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(self.executor, self.analyze_performance_and_recommend_parameters)
                
                cp=len(self.pump_memory["tokens"])+sum(1 for token in self.memory["tokens"].values() if token.get("pump_status")=="confirmed_pump")
                
                # Train pump model every 75 processed tokens (same as trade model)
                if not self.is_trained and cp>=15:
                    self.logger.info("Starting initial training with confirmed pumps...")
                    loop=asyncio.get_running_loop()
                    await loop.run_in_executor(self.executor,self.train_model_sync)
                elif self.is_trained and self.tokens_processed%75==0:
                    self.logger.info("Retraining model with new data...")
                    loop=asyncio.get_running_loop()
                    await loop.run_in_executor(self.executor,self.train_model_sync)
                    
                if self.tokens_processed%3==0:
                    self.prune_memory()
                    self.save_memory()
                    self.save_pump_memory()
                if self.tokens_processed%20==0:
                    mt=sum(1 for state in self.memory["token_states"].values() if state=="monitoring")
                    at=sum(1 for state in self.memory["token_states"].values() if state=="active")
                    self.logger.info(f"Processed {len(self.memory['tokens'])} tokens, {mt} monitoring, {at} active, {cp} confirmed pumps")
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