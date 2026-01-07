"""
Radical Trading Strategy with Deep Reinforcement Learning
激進交易策略 - 深度強化學習驅動
Strategy 2: Radical Trading Strategy with Deep Reinforcement Learning
This code implements an advanced algorithmic trading strategy that combines deep reinforcement learning (DRL) with Transformer attention mechanisms and adaptive risk management. The strategy employs a sophisticated Double DQN architecture with a 4-layer neural network (24->128->64->32->9 actions), integrated Transformer-based attention for temporal feature enhancement, and prioritized experience replay for efficient learning.
Architecture: Double DQN (4-layer: 24->128->64->32->9) + Transformer Attention + Prioritized ER.
References: Mnih et al. (2015), Van Hasselt et al. (2016), Vaswani et al. (2017).
"""

from AlgoAPI import AlgoAPIUtil, AlgoAPI_Backtest
import numpy as np
from datetime import datetime, timezone
import time
import random

class AlgoEvent:
    """Radical Trading Strategy with Deep Reinforcement Learning / 激進交易策略 - 深度強化學習"""
    def __init__(self):
        # Data Storage / 資料儲存
        self.prices = []
        self.returns = []
        self.volumes = []
        self.highs = []
        self.lows = []
        self.bar_count = 0
        
        # Position Management / 倉位管理
        self.position = 0
        self.entry_price = 0
        self.entry_bar = 0
        self.trailing_stop = 0
        self.max_profit = 0
        
        # Strategy Parameters / 策略參數
        self.initial_capital = 100000000
        self.base_position_pct = 0.40  # Base position 40% / 基礎倉位40%
        self.max_position_pct = 0.70   # Max position 70% / 最大倉位70%
        self.min_gap = 3  # Min trade interval / 最小交易間隔
        self.min_bars = 30  # Min data points / 最小資料量
        
        # DQN Parameters / DQN參數
        self.state_dim = 24  # State dimension / 狀態維度
        self.action_dim = 9  # Action space: 9 actions / 動作空間：9個動作
        self.learning_rate = 0.005
        self.gamma = 0.97  # Discount factor / 折扣因子
        self.epsilon = 0.25  # Exploration rate / 探索率
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.998
        
        # Neural Network / 神經網路
        self.q_network = self.init_deep_network()
        self.target_network = self.init_deep_network()
        self.update_target_freq = 50
        self.update_count = 0
        
        # Prioritized Experience Replay / 優先級經驗回放
        self.replay_buffer = []
        self.priorities = []
        self.buffer_size = 2000
        self.batch_size = 64
        self.alpha = 0.6  # Priority exponent / 優先級指數
        self.beta = 0.4  # Importance sampling / 重要性採樣
        self.beta_increment = 0.0001
        
        # Transformer Attention / Transformer注意力
        self.attention_dim = 24
        self.attention_weights_q = np.random.randn(self.attention_dim, self.attention_dim) * 0.01
        self.attention_weights_k = np.random.randn(self.attention_dim, self.attention_dim) * 0.01
        self.attention_weights_v = np.random.randn(self.attention_dim, self.attention_dim) * 0.01
        
        # Risk Control / 風險控制
        self.current_volatility = 0.20
        self.max_drawdown = 0.0
        self.peak_equity = self.initial_capital
        self.risk_budget = 1.0
        
        # Trading Hour Cache / 交易時間緩存
        self.last_trade_time = None
        self.trading_hour_cache = {}
        
        # Performance Tracking / 性能追蹤
        self.recent_pnls = []
        self.win_rate = 0.5
        self.profit_factor = 1.0
        
    def start(self, mEvt):
        self.instrument = mEvt['subscribeList'][0]
        self.evt = AlgoAPI_Backtest.AlgoEvtHandler(self, mEvt)
        
        # Set lot size / 設置交易單位
        if 'HK' in self.instrument:
            if '00005' in self.instrument:
                self.lot_size = 400
            elif '00700' in self.instrument:
                self.lot_size = 100
            elif '01810' in self.instrument:
                self.lot_size = 100
            else:
                self.lot_size = 100
        else:
            self.lot_size = 1
        
        self.evt.start()
    
    def init_deep_network(self):
        """Initialize 4-layer network / 初始化4層網路"""
        w1 = np.random.randn(self.state_dim, 128) * 0.01
        w2 = np.random.randn(128, 64) * 0.01
        w3 = np.random.randn(64, 32) * 0.01
        w4 = np.random.randn(32, self.action_dim) * 0.01
        return {'w1': w1, 'w2': w2, 'w3': w3, 'w4': w4}
    
    def is_trading_hour(self, md=None):
        """Check trading hours / 檢查交易時間"""
        try:
            if md and hasattr(md, 'timestamp'):
                timestamp = md.timestamp
            elif md and hasattr(md, 'time'):
                timestamp = md.time
            else:
                timestamp = time.time()
            
            if isinstance(timestamp, (int, float)):
                dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
            else:
                dt = timestamp
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
            
            hour = dt.hour
            minute = dt.minute
            weekday = dt.weekday()
            
            if weekday >= 5:
                return False
            
            cache_key = (hour, minute, weekday)
            if cache_key in self.trading_hour_cache:
                return self.trading_hour_cache[cache_key]
            
            result = False
            
            if 'HK' in self.instrument:
                if (hour == 1 and minute >= 30) or (2 <= hour <= 3) or (hour == 4 and minute == 0):
                    result = True
                elif (hour == 5) or (6 <= hour <= 7) or (hour == 8 and minute == 0):
                    result = True
            elif 'US' in self.instrument or 'NVDA' in self.instrument or 'AAPL' in self.instrument or 'MCD' in self.instrument:
                if (hour == 13 and minute >= 30) or (14 <= hour <= 21) or (hour == 22 and minute == 0):
                    result = True
            else:
                result = True
            
            self.trading_hour_cache[cache_key] = result
            return result
        except:
            return True
    
    def calc_atr(self, period=14):
        """Calculate ATR / 計算ATR"""
        if len(self.highs) < period + 1:
            return 0.025
        
        trs = []
        for i in range(len(self.highs) - period, len(self.highs)):
            tr = max(
                self.highs[i] - self.lows[i],
                abs(self.highs[i] - self.prices[i-1]) if i > 0 else 0,
                abs(self.lows[i] - self.prices[i-1]) if i > 0 else 0
            )
            trs.append(tr)
        
        return np.mean(trs) / (self.prices[-1] + 1e-8)
    
    def apply_attention(self, features):
        """Transformer attention mechanism / Transformer注意力機制"""
        Q = np.dot(features, self.attention_weights_q)
        K = np.dot(features, self.attention_weights_k)
        V = np.dot(features, self.attention_weights_v)
        
        attention_scores = np.dot(Q, K.T) / np.sqrt(self.attention_dim)
        attention_scores = np.tanh(attention_scores)
        
        exp_scores = np.exp(attention_scores - np.max(attention_scores))
        attention_probs = exp_scores / (np.sum(exp_scores) + 1e-8)
        
        attention_output = np.dot(attention_probs, V)
        output = features + attention_output * 0.5
        
        return output
    
    def extract_features(self):
        """Extract 24-dimensional feature vector / 提取24維特徵向量"""
        if len(self.prices) < 60:
            return np.zeros(self.state_dim)
        
        features = []
        p = np.array(self.prices[-60:])
        r = np.array(self.returns[-40:]) if len(self.returns) >= 40 else np.array([0])
        v = np.array(self.volumes[-30:]) if len(self.volumes) >= 30 else np.array([1])
        
        # 1-6: Multi-timeframe momentum / 多時間框架動量
        for period in [2, 3, 5, 8, 13, 21]:
            if len(p) >= period:
                momentum = (p[-1] - p[-period]) / (p[-period] + 1e-8)
                features.append(np.tanh(momentum * 20))
            else:
                features.append(0)
        
        # 7-10: Moving average position / 均線位置
        for period in [5, 10, 20, 40]:
            if len(p) >= period:
                ma = np.mean(p[-period:])
                features.append((p[-1] - ma) / (ma + 1e-8))
            else:
                features.append(0)
        
        # 11-14: Technical indicators / 技術指標
        if len(r) >= 20:
            features.append(np.std(r) * 60)  # Volatility / 波動率
            
            # RSI
            gains = [x for x in r if x > 0]
            losses = [-x for x in r if x < 0]
            avg_gain = np.mean(gains) if gains else 0.001
            avg_loss = np.mean(losses) if losses else 0.001
            rsi = 100 - (100 / (1 + avg_gain / avg_loss))
            features.append((rsi - 50) / 50)
            
            # MACD
            if len(p) >= 26:
                ema12 = np.mean(p[-12:])
                ema26 = np.mean(p[-26:])
                macd = (ema12 - ema26) / (ema26 + 1e-8)
                features.append(macd * 30)
            else:
                features.append(0)
            
            # CCI
            if len(self.highs) >= 20:
                typical_price = (np.array(self.highs[-20:]) + 
                               np.array(self.lows[-20:]) + 
                               np.array(p[-20:])) / 3
                sma_tp = np.mean(typical_price)
                mad = np.mean(np.abs(typical_price - sma_tp))
                cci = (typical_price[-1] - sma_tp) / (0.015 * mad + 1e-8)
                features.append(np.tanh(cci / 100))
            else:
                features.append(0)
        else:
            features.extend([0.02, 0, 0, 0])
        
        # 15-18: Volume features / 成交量特徵
        if len(v) >= 15:
            vol_ratio = v[-1] / (np.mean(v[-15:]) + 1e-8)
            features.append(min(vol_ratio - 1, 3))
            vol_trend = (np.mean(v[-5:]) - np.mean(v[-10:-5])) / (np.mean(v[-10:-5]) + 1e-8)
            features.append(min(vol_trend, 2))
            
            if len(r) > 0:
                price_vol_corr = 1 if (r[-1] > 0 and vol_ratio > 1) or (r[-1] < 0 and vol_ratio > 1) else -0.5
                features.append(price_vol_corr * min(vol_ratio, 1.5))
            else:
                features.append(0)
            
            vol_vol = np.std(v[-10:]) / (np.mean(v[-10:]) + 1e-8)
            features.append(min(vol_vol, 1))
        else:
            features.extend([0, 0, 0, 0])
        
        # 19-21: Breakout and trend strength / 突破和趨勢強度
        if len(p) >= 30:
            high_30 = max(p[-30:])
            low_30 = min(p[-30:])
            features.append((p[-1] - high_30) / (high_30 + 1e-8))
            features.append((p[-1] - low_30) / (low_30 + 1e-8))
            ma_fast = np.mean(p[-10:])
            ma_slow = np.mean(p[-20:])
            trend_strength = abs(ma_fast - ma_slow) / (ma_slow + 1e-8)
            features.append(trend_strength * 15)
        else:
            features.extend([0, 0, 0])
        
        # 22-24: Price acceleration, volatility change, position / 價格加速度、波動率變化、持倉
        if len(r) >= 5:
            accel = r[-1] - r[-3] if len(r) >= 3 else 0
            features.append(accel * 150)
            if len(r) >= 10:
                vol_change = np.std(r[-5:]) - np.std(r[-10:-5])
                features.append(vol_change * 80)
            else:
                features.append(0)
        else:
            features.extend([0, 0])
        
        if self.position != 0:
            pnl_pct = (p[-1] - self.entry_price) / (self.entry_price + 1e-8)
            if self.position < 0:
                pnl_pct = -pnl_pct
            features.append(np.tanh(pnl_pct * 10))
        else:
            features.append(0)
        
        features = np.array(features)
        features = self.apply_attention(features)
        
        return features
    
    def forward_pass(self, state, network):
        """Forward propagation / 前向傳播"""
        h1 = np.tanh(np.dot(state, network['w1']))
        h2 = np.tanh(np.dot(h1, network['w2']))
        h3 = np.tanh(np.dot(h2, network['w3']))
        output = np.dot(h3, network['w4'])
        return output
    
    def select_action(self, state, training=True):
        """ε-greedy action selection / ε-貪婪動作選擇"""
        if training and random.random() < self.epsilon:
            # Exploration / 探索
            actions = list(range(self.action_dim))
            weights = [0.10, 0.10, 0.08, 0.07, 0.30, 0.07, 0.08, 0.10, 0.10]
            return np.random.choice(actions, p=weights)
        else:
            # Exploitation / 利用
            q_values = self.forward_pass(state, self.q_network)
            if q_values.ndim > 1:
                q_values = q_values.flatten()
            if len(q_values) != self.action_dim:
                return 4  # Hold / 持有
            return np.argmax(q_values)
    
    def action_to_signal(self, action):
        """Convert action to trading signal / 將動作轉換為交易信號"""
        action_map = {
            0: (-4, 0.55), 1: (-3, 0.45), 2: (-2, 0.35), 3: (-1, 0.25),
            4: (0, 0.0),   5: (1, 0.25),  6: (2, 0.35),  7: (3, 0.45), 8: (4, 0.55)
        }
        return action_map.get(action, (0, 0.0))
    
    def compute_reward(self, price, prev_price, prev_action, position):
        """Compute reward / 計算獎勵"""
        if prev_action == 4:
            return 0
        
        price_change = (price - prev_price) / (prev_price + 1e-8)
        
        # Base reward / 基礎獎勵
        if prev_action in [0, 1, 2, 3]:  # Short / 做空
            reward = -price_change * 300
        else:  # Long / 做多
            reward = price_change * 300
        
        # Position reward / 持倉獎勵
        if position != 0:
            if (position > 0 and price_change > 0) or (position < 0 and price_change < 0):
                reward += abs(price_change) * 150
        
        # Risk penalty / 風險懲罰
        if len(self.returns) >= 10:
            vol = np.std(self.returns[-10:])
            reward -= vol * 30
        
        # Time penalty / 時間懲罰
        if self.position != 0:
            bars_held = self.bar_count - self.entry_bar
            if bars_held > 40:
                reward -= 0.2
        
        return reward
    
    def train_network(self):
        """Train network (Double DQN + Prioritized ER) / 訓練網路(Double DQN + 優先級經驗回放)"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Calculate priorities / 計算優先級
        priorities = np.array(self.priorities)
        priorities = np.power(priorities + 1e-6, self.alpha)
        probs = priorities / np.sum(priorities)
        
        # Sampling / 採樣
        indices = np.random.choice(len(self.replay_buffer), self.batch_size, p=probs)
        batch = [self.replay_buffer[i] for i in indices]
        
        # Importance sampling weights / 重要性採樣權重
        weights = np.power(len(self.replay_buffer) * probs[indices], -self.beta)
        weights = weights / np.max(weights)
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Training / 訓練
        for idx, (state, action, reward, next_state, done) in enumerate(batch):
            weight = weights[idx]
            current_q = self.forward_pass(state, self.q_network)
            if current_q.ndim > 1:
                current_q = current_q.flatten()
            
            # Target Q-value (Double DQN) / 目標Q值
            if done:
                target_q = reward
            else:
                next_q = self.forward_pass(next_state, self.q_network)
                if next_q.ndim > 1:
                    next_q = next_q.flatten()
                best_action = np.argmax(next_q)
                
                target_q_values = self.forward_pass(next_state, self.target_network)
                if target_q_values.ndim > 1:
                    target_q_values = target_q_values.flatten()
                target_q = reward + self.gamma * target_q_values[best_action]
            
            # Update Q-network / 更新Q網路
            td_error = abs(target_q - current_q[action])
            self.priorities[indices[idx]] = td_error
            
            if action < len(current_q):
                error = (target_q - current_q[action]) * weight * self.learning_rate
                h1 = np.tanh(np.dot(state, self.q_network['w1']))
                h2 = np.tanh(np.dot(h1, self.q_network['w2']))
                h3 = np.tanh(np.dot(h2, self.q_network['w3']))
                
                grad_w4 = np.outer(h3, np.zeros(self.action_dim))
                grad_w4[:, action] = error
                self.q_network['w4'] += grad_w4 * 0.01
        
        # Update target network / 更新目標網路
        self.update_count += 1
        if self.update_count % self.update_target_freq == 0:
            self.target_network = {
                'w1': self.q_network['w1'].copy(),
                'w2': self.q_network['w2'].copy(),
                'w3': self.q_network['w3'].copy(),
                'w4': self.q_network['w4'].copy()
            }
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def calc_position_size(self, price, available, signal_strength, signal_direction):
        """Calculate position size / 計算倉位大小"""
        base_pct = self.base_position_pct
        is_hk = 'HK' in self.instrument
        
        if is_hk:
            base_pct *= 0.15  # HK stocks / 港股
        
        # Signal strength adjustment / 信號強度調整
        strength_multiplier = 0.9 + signal_strength * 0.8
        target_pct = base_pct * strength_multiplier
        
        # Dynamic risk adjustment / 動態風險調整
        if len(self.returns) >= 20:
            volatility = np.std(self.returns[-20:]) * np.sqrt(252 * 24)
            self.current_volatility = volatility
            if volatility > 0.35:
                target_pct *= 0.35 / volatility
        
        # Drawdown protection / 回撤保護
        current_equity = available
        if current_equity < self.peak_equity:
            drawdown = (self.peak_equity - current_equity) / self.peak_equity
            if drawdown > 0.20:
                target_pct *= 0.4
            elif drawdown > 0.15:
                target_pct *= 0.6
        else:
            self.peak_equity = current_equity
        
        # Limit range / 限制範圍
        if is_hk:
            target_pct = max(0.10, min(target_pct, 0.20))
        else:
            target_pct = max(0.25, min(target_pct, self.max_position_pct))
        
        # Calculate target amount / 計算目標金額
        target = available * target_pct
        lots = max(1, int(target / (price * self.lot_size)))
        shares = lots * self.lot_size
        
        # Verify capital / 驗證資金
        required = shares * price * 1.25
        max_allowed_ratio = 0.75 if is_hk else 0.95
        max_allowed = available * max_allowed_ratio
        
        # Gradually reduce / 逐步減少
        max_iterations = 20
        iteration = 0
        while required > max_allowed and lots > 1 and iteration < max_iterations:
            lots = max(1, lots - 1)
            shares = lots * self.lot_size
            required = shares * price * 1.25
            iteration += 1
        
        final_max_ratio = 0.70 if is_hk else 0.90
        if required <= available * final_max_ratio:
            return shares
        
        target = available * (final_max_ratio * 0.9)
        lots = max(1, int(target / (price * self.lot_size)))
        shares = lots * self.lot_size
        return shares
    
    def on_marketdatafeed(self, md, ab):
        # Check trading hours / 檢查交易時間
        if not self.is_trading_hour(md):
            return
        
        # Get price data / 獲取價格資料
        price = md.lastPrice
        high = getattr(md, 'high', price)
        low = getattr(md, 'low', price)
        volume = getattr(md, 'volume', 1000000)
        
        # Get available capital / 獲取可用資金
        try:
            if isinstance(ab, dict):
                available = float(ab.get('availableBalance', self.initial_capital * 0.9))
            else:
                available = float(getattr(ab, 'availableBalance', self.initial_capital * 0.9))
            
            if available <= 0 or available > self.initial_capital * 10:
                available = self.initial_capital * 0.9
        except:
            available = self.initial_capital * 0.9
        
        available *= 0.90
        
        # Store data / 儲存資料
        prev_price = self.prices[-1] if self.prices else price
        self.prices.append(price)
        self.highs.append(high)
        self.lows.append(low)
        self.volumes.append(volume)
        self.bar_count += 1
        
        # Calculate return rate / 計算收益率
        if len(self.prices) > 1:
            ret = (price - self.prices[-2]) / (self.prices[-2] + 1e-8)
            self.returns.append(ret)
        
        # Limit data length / 限制資料長度
        if len(self.prices) > 500:
            self.prices = self.prices[-500:]
            self.highs = self.highs[-500:]
            self.lows = self.lows[-500:]
            self.returns = self.returns[-500:]
            self.volumes = self.volumes[-500:]
        
        # Insufficient data / 資料不足
        if len(self.prices) < self.min_bars:
            return
        
        # Extract features / 提取特徵
        state = self.extract_features()
        
        # Calculate ATR / 計算ATR
        atr = self.calc_atr(14)
        
        # Position management / 持倉管理
        if self.position != 0:
            bars_held = self.bar_count - self.entry_bar
            
            # Calculate P&L / 計算盈虧
            pnl_pct = (price - self.entry_price) / (self.entry_price + 1e-8)
            if self.position < 0:
                pnl_pct = -pnl_pct
            
            # Update maximum profit / 更新最大盈利
            if pnl_pct > self.max_profit:
                self.max_profit = pnl_pct
            
            # Dynamic stop loss / 動態止損
            should_exit = False
            if self.position > 0:
                new_stop = price * (1 - 1.8 * max(atr, 0.02))
                if new_stop > self.trailing_stop or self.trailing_stop == 0:
                    self.trailing_stop = new_stop
                
                # Trailing take profit / 移動止盈
                if pnl_pct > 0.025 and self.max_profit > 0.025:
                    trailing_profit = self.entry_price * (1 + (self.max_profit * 0.70))
                    if trailing_profit > self.trailing_stop:
                        self.trailing_stop = trailing_profit
                
                # Take profit / 止盈
                if pnl_pct > 5.0 * max(atr, 0.02):
                    should_exit = True
                elif price <= self.trailing_stop or bars_held >= 60:
                    should_exit = True
            else:
                new_stop = price * (1 + 1.8 * max(atr, 0.02))
                if new_stop < self.trailing_stop or self.trailing_stop == 0:
                    self.trailing_stop = new_stop
                
                if pnl_pct > 0.025 and self.max_profit > 0.025:
                    trailing_profit = self.entry_price * (1 - (self.max_profit * 0.70))
                    if trailing_profit < self.trailing_stop:
                        self.trailing_stop = trailing_profit
                
                profit_pct = (self.entry_price - price) / self.entry_price
                if profit_pct > 5.0 * max(atr, 0.02):
                    should_exit = True
                elif price >= self.trailing_stop or bars_held >= 60:
                    should_exit = True
            
            if should_exit:
                # Record trade results / 記錄交易結果
                if pnl_pct != 0:
                    self.recent_pnls.append(pnl_pct)
                    if len(self.recent_pnls) > 100:
                        self.recent_pnls.pop(0)
                    
                    if len(self.recent_pnls) >= 10:
                        wins = [r for r in self.recent_pnls if r > 0]
                        losses = [r for r in self.recent_pnls if r < 0]
                        self.win_rate = len(wins) / len(self.recent_pnls)
                        if losses:
                            avg_win = np.mean(wins) if wins else 0
                            avg_loss = abs(np.mean(losses))
                            self.profit_factor = avg_win / avg_loss if avg_loss > 0 else 1.0
                
                self.close_position(price)
            
            # Record experience / 記錄經驗
            if hasattr(self, 'last_state') and hasattr(self, 'last_action'):
                prev_state = self.last_state
                prev_action = self.last_action
                reward = self.compute_reward(price, prev_price, prev_action, self.position)
                done = (self.position == 0)
                
                # Add to replay buffer / 添加到回放緩衝區
                if len(self.replay_buffer) >= self.buffer_size:
                    self.replay_buffer.pop(0)
                    self.priorities.pop(0)
                
                td_error = abs(reward) + 0.1
                self.replay_buffer.append((prev_state, prev_action, reward, state, done))
                self.priorities.append(td_error)
                
                # Train network / 訓練網路
                if len(self.replay_buffer) >= self.batch_size:
                    self.train_network()
        
        # Entry logic / 開倉邏輯
        else:
            # Check trade interval / 檢查交易間隔
            if self.last_trade_time is not None:
                if self.bar_count - self.last_trade_time < self.min_gap:
                    return
            
            # DQN decision / DQN決策
            action = self.select_action(state, training=True)
            signal, signal_strength = self.action_to_signal(action)
            
            # Signal strength filter / 信號強度過濾
            if signal == 0 or signal_strength < 0.30:
                return
            
            # Volatility filter / 波動率過濾
            if len(self.returns) >= 20:
                vol = np.std(self.returns[-20:])
                if vol > 0.25 or vol < 0.003:
                    return
            
            # Volume confirmation / 成交量確認
            if len(self.volumes) >= 15:
                vol_ma = np.mean(self.volumes[-15:])
                if self.volumes[-1] < vol_ma * 0.45:
                    return
            
            # Calculate position / 計算倉位
            size = self.calc_position_size(price, available, signal_strength, signal)
            if size > 0:
                self.open_position(signal, price, size, atr)
                self.last_action = action
                self.last_state = state
                self.last_trade_time = self.bar_count
    
    def open_position(self, direction, price, size, atr):
        """Open position / 開倉"""
        if size == 0:
            return
        
        actual_direction = 1 if direction > 0 else -1
        
        order = AlgoAPIUtil.OrderObject()
        order.instrument = self.instrument
        order.orderRef = f"R_{self.bar_count}"
        order.volume = int(size)
        order.openclose = 'open'
        order.buysell = actual_direction
        order.ordertype = 0
        
        try:
            self.evt.sendOrder(order)
            self.position = size if actual_direction > 0 else -size
            self.entry_price = price
            self.entry_bar = self.bar_count
            self.max_profit = 0
            
            # Dynamic stop loss / 動態止損
            atr_adj = max(atr, 0.02)
            if actual_direction > 0:
                self.trailing_stop = price * (1 - 1.8 * atr_adj)
            else:
                self.trailing_stop = price * (1 + 1.8 * atr_adj)
        except Exception as e:
            pass
    
    def close_position(self, price):
        """Close position / 平倉"""
        if self.position == 0:
            return
        
        # Record trade results / 記錄交易結果
        pnl_pct = (price - self.entry_price) / (self.entry_price + 1e-8)
        if self.position < 0:
            pnl_pct = -pnl_pct
        
        if pnl_pct != 0:
            self.recent_pnls.append(pnl_pct)
            if len(self.recent_pnls) > 100:
                self.recent_pnls.pop(0)
        
        order = AlgoAPIUtil.OrderObject()
        order.instrument = self.instrument
        order.orderRef = f"RX_{self.bar_count}"
        order.volume = int(abs(self.position))
        order.openclose = 'open'
        order.buysell = -1 if self.position > 0 else 1
        order.ordertype = 0
        
        try:
            self.evt.sendOrder(order)
            self.position = 0
            self.entry_price = 0
            self.trailing_stop = 0
            self.max_profit = 0
        except Exception as e:
            pass
    
    def on_bulkdatafeed(self, isSync, bd, ab):
        pass
    
    def on_orderfeed(self, of):
        pass
    
    def on_dailyPLfeed(self, pl):
        pass
    
    def on_openPositionfeed(self, op, oo, uo):
        pass


