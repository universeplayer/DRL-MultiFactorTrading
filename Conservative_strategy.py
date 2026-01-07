"""
Conservative Trading Strategy with Multi-Factor Model
Strategy 1: Conservative Trading Strategy with Multi-Factor Model

This code implements a sophisticated conservative algorithmic trading strategy based on rigorous academic multi-factor models, combining trend analysis, momentum indicators, and classical technical analysis with dynamic risk management. The strategy synthesizes six independent factors (Trend 35%, Momentum 25%, RSI 20%, MACD 15%, Bollinger Bands 5%, Volume confirmation) through weighted aggregation to generate robust trading signals.
Architecture: Multi-Factor Model (Trend, Momentum, RSI, MACD, Bollinger Bands) + Volatility Targeting
References: Fama & French (1993), Kim et al. (2016), Ang & Timmermann (2012).

參考文獻：Fama & French (1993), Kim et al. (2016), Ang & Timmermann (2012)
"""

from AlgoAPI import AlgoAPIUtil, AlgoAPI_Backtest
import numpy as np
from datetime import datetime, timezone
import time

class AlgoEvent:
    """Conservative Trading Strategy with Multi-Factor Model / 保守交易策略 - 多因子模型驅動"""
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
        self.stop_loss = 0
        self.take_profit = 0
        self.highest_price = 0
        self.lowest_price = 0
        self.max_profit_pct = 0
        
        # Strategy Parameters / 策略參數
        self.initial_capital = 100000000
        self.base_position_pct = 0.35  # Base position 35% / 基礎倉位35%
        self.max_position_pct = 0.55   # Max position 55% / 最大倉位55%
        self.min_gap = 8  # Min trade interval / 最小交易間隔
        self.min_bars = 50  # Min data points / 最小資料量
        
        # Risk Control Parameters / 風險控制參數
        self.target_volatility = 0.15  # Target volatility 15% / 目標波動率15%
        self.max_drawdown_pct = 0.15   # Max drawdown 15% / 最大回撤15%
        self.atr_period = 14
        self.stop_loss_atr = 2.0       # Stop loss: 2x ATR / 止損：2倍ATR
        self.take_profit_atr = 4.0     # Take profit: 4x ATR / 止盈：4倍ATR
        
        # Trading Hour Cache / 交易時間緩存
        self.last_trade_time = None
        self.trading_hour_cache = {}
        
        # Performance Tracking / 性能追蹤
        self.recent_returns = []
        self.recent_pnls = []
        self.win_rate = 0.5
        self.profit_factor = 1.0
        self.peak_equity = self.initial_capital
        
        # Market State / 市場狀態
        self.market_state = 'trending'
        self.volatility_regime = 'normal'
        
    def start(self, mEvt):
        self.instrument = mEvt['subscribeList'][0]
        self.evt = AlgoAPI_Backtest.AlgoEvtHandler(self, mEvt)
        
        # Set lot size / 設置交易單位
        if 'HK' in self.instrument:
            if '00005' in self.instrument:
                self.lot_size = 400
            elif '00700' in self.instrument:
                self.lot_size = 400
            elif '01810' in self.instrument:
                self.lot_size = 400
            else:
                self.lot_size = 400
        else:
            self.lot_size = 1
        
        self.evt.start()
    
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
                # HK stocks: 09:30-12:00, 13:00-16:00 HKT = UTC+8 / 港股：09:30-12:00, 13:00-16:00 HKT = UTC+8
                if (hour == 1 and minute >= 30) or (2 <= hour <= 3) or (hour == 4 and minute == 0):
                    result = True
                elif (hour == 5) or (6 <= hour <= 7) or (hour == 8 and minute == 0):
                    result = True
            elif 'US' in self.instrument or 'NVDA' in self.instrument or 'AAPL' in self.instrument or 'MCD' in self.instrument:
                # US stocks: 09:30-16:00 ET / 美股：09:30-16:00 ET
                if (hour == 13 and minute >= 30) or (14 <= hour <= 21) or (hour == 22 and minute == 0):
                    result = True
            else:
                result = True
            
            self.trading_hour_cache[cache_key] = result
            return result
        except:
            return True
    
    def calc_atr(self, period=None):
        """Calculate ATR / 計算ATR"""
        if period is None:
            period = self.atr_period
        
        if len(self.highs) < period + 1:
            return 0.02
        
        trs = []
        for i in range(len(self.highs) - period, len(self.highs)):
            tr = max(
                self.highs[i] - self.lows[i],
                abs(self.highs[i] - self.prices[i-1]) if i > 0 else 0,
                abs(self.lows[i] - self.prices[i-1]) if i > 0 else 0
            )
            trs.append(tr)
        
        return np.mean(trs)
    
    def calc_rsi(self, period=14):
        """Calculate RSI / 計算RSI"""
        if len(self.returns) < period:
            return 50.0
        
        gains = [r for r in self.returns[-period:] if r > 0]
        losses = [-r for r in self.returns[-period:] if r < 0]
        
        avg_gain = np.mean(gains) if gains else 0.001
        avg_loss = np.mean(losses) if losses else 0.001
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calc_macd(self, fast=12, slow=26, signal=9):
        """Calculate MACD / 計算MACD"""
        if len(self.prices) < slow:
            return 0.0, 0.0, 0.0
        
        p = np.array(self.prices[-slow:])
        
        # EMA calculation / EMA計算
        ema_fast = np.mean(p[-fast:])
        ema_slow = np.mean(p)
        macd_line = ema_fast - ema_slow
        
        # Signal line / 信號線
        if len(self.prices) >= slow + signal:
            signal_line = np.mean([np.mean(self.prices[-i-fast:-i]) - np.mean(self.prices[-i-slow:-i]) 
                                  for i in range(signal)])
        else:
            signal_line = macd_line * 0.8
        
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def calc_bollinger_bands(self, period=20, std_dev=2):
        """Calculate Bollinger Bands / 計算布林帶"""
        if len(self.prices) < period:
            price = self.prices[-1]
            return price, price, price
        
        p = np.array(self.prices[-period:])
        ma = np.mean(p)
        std = np.std(p)
        
        upper_band = ma + std_dev * std
        lower_band = ma - std_dev * std
        
        return upper_band, ma, lower_band
    
    def identify_market_state(self):
        """Identify market state / 識別市場狀態"""
        if len(self.prices) < 50 or len(self.returns) < 30:
            return 'trending', 'normal'
        
        p = np.array(self.prices[-50:])
        r = np.array(self.returns[-30:])
        
        # Calculate volatility / 計算波動率
        volatility = np.std(r) * np.sqrt(252 * 24)
        if volatility < 0.10:
            vol_regime = 'low'
        elif volatility > 0.30:
            vol_regime = 'high'
        else:
            vol_regime = 'normal'
        
        # Trend strength / 趨勢強度
        ma_fast = np.mean(p[-10:])
        ma_medium = np.mean(p[-20:])
        ma_slow = np.mean(p[-40:])
        trend_up = (ma_fast > ma_medium > ma_slow)
        trend_down = (ma_fast < ma_medium < ma_slow)
        trend_strength = abs(ma_fast - ma_slow) / (ma_slow + 1e-8)
        
        # Market state / 市場狀態
        if trend_strength > 0.08 and (trend_up or trend_down):
            market_state = 'trending'
        elif volatility > 0.25:
            market_state = 'volatile'
        else:
            market_state = 'ranging'
        
        return market_state, vol_regime
    
    def generate_signal(self):
        """Generate trading signal / 生成交易信號"""
        if len(self.prices) < self.min_bars:
            return 0, 0.0
        
        price = self.prices[-1]
        p = np.array(self.prices[-50:])
        r = np.array(self.returns[-30:]) if len(self.returns) >= 30 else np.array([0])
        
        # 1. Trend signal / 趨勢信號
        ma_fast = np.mean(p[-8:])
        ma_medium = np.mean(p[-20:])
        ma_slow = np.mean(p[-40:])
        trend_signal = 0
        trend_strength = 0.0
        
        if ma_fast > ma_medium > ma_slow:
            trend_signal = 1
            trend_strength = min((ma_fast - ma_slow) / (ma_slow + 1e-8) * 15, 1.0)
        elif ma_fast < ma_medium < ma_slow:
            trend_signal = -1
            trend_strength = min((ma_slow - ma_fast) / (ma_fast + 1e-8) * 15, 1.0)
        
        # 2. Momentum signal / 動量信號
        momentum_5 = (p[-1] - p[-5]) / (p[-5] + 1e-8) if len(p) >= 5 else 0
        momentum_10 = (p[-1] - p[-10]) / (p[-10] + 1e-8) if len(p) >= 10 else 0
        momentum = (momentum_5 * 0.6 + momentum_10 * 0.4)
        momentum_signal = 1 if momentum > 0.008 else (-1 if momentum < -0.008 else 0)
        momentum_strength = min(abs(momentum) * 50, 1.0)
        
        # 3. RSI signal / RSI信號
        rsi = self.calc_rsi(14)
        rsi_signal = 0
        rsi_strength = 0.0
        if rsi < 35:  # Oversold / 超賣
            rsi_signal = 1
            rsi_strength = (35 - rsi) / 35
        elif rsi > 65:  # Overbought / 超買
            rsi_signal = -1
            rsi_strength = (rsi - 65) / 35
        
        # 4. MACD signal / MACD信號
        macd, signal_line, histogram = self.calc_macd()
        macd_signal = 0
        macd_strength = 0.0
        if histogram > 0 and macd > signal_line:
            macd_signal = 1
            macd_strength = min(abs(histogram) * 100, 1.0)
        elif histogram < 0 and macd < signal_line:
            macd_signal = -1
            macd_strength = min(abs(histogram) * 100, 1.0)
        
        # 5. Bollinger Bands signal / 布林帶信號
        upper_band, middle_band, lower_band = self.calc_bollinger_bands()
        bb_signal = 0
        bb_strength = 0.0
        if price < lower_band:
            bb_signal = 1
            bb_strength = min((lower_band - price) / (lower_band + 1e-8), 1.0)
        elif price > upper_band:
            bb_signal = -1
            bb_strength = min((price - upper_band) / (upper_band + 1e-8), 1.0)
        
        # 6. Volume confirmation / 成交量確認
        volume_signal = 0
        if len(self.volumes) >= 20:
            vol_ratio = self.volumes[-1] / (np.mean(self.volumes[-20:]) + 1e-8)
            if vol_ratio > 1.3:
                volume_signal = 1 if rsi < 50 else -1
            elif vol_ratio < 0.7:
                volume_signal = 0
        
        # Comprehensive signal / 綜合信號
        signals = []
        strengths = []
        if trend_signal != 0:
            signals.append(trend_signal)
            strengths.append(trend_strength * 0.35)
        if momentum_signal != 0:
            signals.append(momentum_signal)
            strengths.append(momentum_strength * 0.25)
        if rsi_signal != 0:
            signals.append(rsi_signal)
            strengths.append(rsi_strength * 0.20)
        if macd_signal != 0:
            signals.append(macd_signal)
            strengths.append(macd_strength * 0.15)
        if bb_signal != 0:
            signals.append(bb_signal)
            strengths.append(bb_strength * 0.05)
        
        if not signals:
            return 0, 0.0
        
        # Weighted average / 加權平均
        signal_sum = sum(s * st for s, st in zip(signals, strengths))
        strength_sum = sum(strengths)
        
        if strength_sum < 0.25:
            return 0, 0.0
        
        final_signal = 1 if signal_sum > 0.15 else (-1 if signal_sum < -0.15 else 0)
        final_strength = min(strength_sum, 1.0)
        
        # Volume confirmation / 成交量確認
        if volume_signal != 0 and volume_signal != final_signal:
            final_strength *= 0.7
        
        return final_signal, final_strength
    
    def calc_position_size(self, price, available, signal_strength, signal_direction):
        """Calculate position size / 計算倉位大小"""
        is_hk = 'HK' in self.instrument
        
        # Hong Kong stock special handling / 港股特殊處理
        if is_hk:
            one_lot_cost = price * self.lot_size * 1.25
            if one_lot_cost > available * 0.8:
                return 0
            
            max_cost = available * self.base_position_pct * (0.8 + signal_strength * 0.4)
            max_lots = int(max_cost / (price * self.lot_size * 1.25))
            
            # Grade based on signal strength / 根據信號強度分級
            if signal_strength > 0.6:
                lots = min(25, max_lots)
            elif signal_strength > 0.4:
                lots = min(20, max_lots)
            elif signal_strength > 0.25:
                lots = min(15, max_lots)
            else:
                lots = min(10, max_lots)
            
            lots = max(3, lots)
            shares = lots * self.lot_size
            
            required = shares * price * 1.25
            if required <= available * 0.75:
                return shares
            else:
                # Gradually reduce / 逐步減少
                for test_lots in [20, 15, 10, 8, 5, 3]:
                    test_shares = test_lots * self.lot_size
                    test_required = test_shares * price * 1.25
                    if test_required <= available * 0.75:
                        return test_shares
                return 0
        
        # Non-HK stocks / 非港股
        base_pct = self.base_position_pct
        strength_multiplier = 0.85 + signal_strength * 0.5
        target_pct = base_pct * strength_multiplier
        
        # Volatility adjustment / 波動率調整
        if len(self.returns) >= 20:
            volatility = np.std(self.returns[-20:]) * np.sqrt(252 * 24)
            if volatility > 0:
                vol_adjustment = min(self.target_volatility / volatility, 1.2)
                target_pct *= vol_adjustment
        
        # Drawdown protection / 回撤保護
        current_equity = available
        if current_equity < self.peak_equity:
            drawdown = (self.peak_equity - current_equity) / self.peak_equity
            if drawdown > 0.10:
                target_pct *= (1 - drawdown * 0.6)
        else:
            self.peak_equity = current_equity
        
        # Limit range / 限制範圍
        target_pct = max(0.20, min(target_pct, self.max_position_pct))
        
        # Calculate target amount / 計算目標金額
        target = available * target_pct
        lots = max(1, int(target / (price * self.lot_size)))
        shares = lots * self.lot_size
        
        # Verify capital / 驗證資金
        required = shares * price * 1.20
        max_allowed = available * 0.85
        if required <= max_allowed:
            return shares
        
        target = available * 0.80
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
        
        # Calculate ATR / 計算ATR
        atr = self.calc_atr()
        atr_pct = atr / (price + 1e-8)
        
        # Position management / 持倉管理
        if self.position != 0:
            # Update highest and lowest prices / 更新最高價和最低價
            if self.position > 0:
                if price > self.highest_price:
                    self.highest_price = price
                pnl_pct = (price - self.entry_price) / (self.entry_price + 1e-8)
            else:
                if price < self.lowest_price or self.lowest_price == 0:
                    self.lowest_price = price
                pnl_pct = (self.entry_price - price) / (self.entry_price + 1e-8)
            
            # Update maximum profit / 更新最大盈利
            if pnl_pct > self.max_profit_pct:
                self.max_profit_pct = pnl_pct
            
            bars_held = self.bar_count - self.entry_bar
            should_exit = False
            
            # Stop loss and take profit / 止損止盈
            if self.position > 0:
                # Stop loss / 止損
                stop_loss_price = self.entry_price * (1 - self.stop_loss_atr * atr_pct)
                if price <= stop_loss_price:
                    should_exit = True
                
                # Take profit / 止盈
                take_profit_price = self.entry_price * (1 + self.take_profit_atr * atr_pct)
                if price >= take_profit_price:
                    should_exit = True
                
                # Trailing stop / 移動止盈
                if pnl_pct > 0.02:
                    trailing_stop = self.entry_price * (1 + self.max_profit_pct * 0.5)
                    if price < trailing_stop:
                        should_exit = True
            else:
                # Short position / 做空
                stop_loss_price = self.entry_price * (1 + self.stop_loss_atr * atr_pct)
                if price >= stop_loss_price:
                    should_exit = True
                
                take_profit_price = self.entry_price * (1 - self.take_profit_atr * atr_pct)
                if price <= take_profit_price:
                    should_exit = True
                
                if pnl_pct > 0.02:
                    trailing_stop = self.entry_price * (1 - self.max_profit_pct * 0.5)
                    if price > trailing_stop:
                        should_exit = True
            
            # Time stop loss / 時間止損
            if bars_held >= 150:
                should_exit = True
            
            # Signal reversal / 信號反轉
            signal_dir, signal_strength = self.generate_signal()
            if (self.position > 0 and signal_dir < -0.3 and signal_strength > 0.35) or \
               (self.position < 0 and signal_dir > 0.3 and signal_strength > 0.35):
                should_exit = True
            
            if should_exit:
                # Record trade results / 記錄交易結果
                if pnl_pct != 0:
                    self.recent_returns.append(pnl_pct)
                    self.recent_pnls.append(pnl_pct)
                    if len(self.recent_returns) > 50:
                        self.recent_returns.pop(0)
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
        
        # Entry logic / 開倉邏輯
        else:
            # Check trade interval / 檢查交易間隔
            if self.last_trade_time is not None:
                if self.bar_count - self.last_trade_time < self.min_gap:
                    return
            
            # Generate signal / 生成信號
            signal_dir, signal_strength = self.generate_signal()
            
            # Signal strength filter / 信號強度過濾
            min_signal_strength = 0.22
            if signal_dir == 0 or signal_strength < min_signal_strength:
                return
            
            # Volatility filter / 波動率過濾
            if len(self.returns) >= 20:
                vol = np.std(self.returns[-20:])
                max_vol = 0.20
                min_vol = 0.002
                if vol > max_vol or vol < min_vol:
                    return
            
            # Volume confirmation / 成交量確認
            if len(self.volumes) >= 20:
                vol_ma = np.mean(self.volumes[-20:])
                min_vol_ratio = 0.60
                if self.volumes[-1] < vol_ma * min_vol_ratio:
                    return
            
            # Calculate position / 計算倉位
            size = self.calc_position_size(price, available, signal_strength, signal_dir)
            if size > 0:
                self.open_position(signal_dir, price, size, atr)
                self.last_trade_time = self.bar_count
    
    def open_position(self, direction, price, size, atr):
        """Open position / 開倉"""
        if size == 0:
            return
        
        order = AlgoAPIUtil.OrderObject()
        order.instrument = self.instrument
        order.orderRef = f"C_{self.bar_count}"
        order.volume = int(size)
        order.openclose = 'open'
        order.buysell = 1 if direction > 0 else -1
        order.ordertype = 0
        
        try:
            self.evt.sendOrder(order)
            self.position = size if direction > 0 else -size
            self.entry_price = price
            self.entry_bar = self.bar_count
            self.highest_price = price
            self.lowest_price = price
            self.max_profit_pct = 0
            
            # Dynamic stop loss and take profit / 動態止損止盈
            atr_pct = atr / (price + 1e-8)
            atr_multiplier = max(atr_pct, 0.015)
            
            if direction > 0:
                self.stop_loss = price * (1 - self.stop_loss_atr * atr_multiplier)
                self.take_profit = price * (1 + self.take_profit_atr * atr_multiplier)
            else:
                self.stop_loss = price * (1 + self.stop_loss_atr * atr_multiplier)
                self.take_profit = price * (1 - self.take_profit_atr * atr_multiplier)
        except Exception as e:
            pass
    
    def close_position(self, price):
        """Close position / 平倉"""
        if self.position == 0:
            return
        
        order = AlgoAPIUtil.OrderObject()
        order.instrument = self.instrument
        order.orderRef = f"CX_{self.bar_count}"
        order.volume = int(abs(self.position))
        order.openclose = 'open'
        order.buysell = -1 if self.position > 0 else 1
        order.ordertype = 0
        
        try:
            self.evt.sendOrder(order)
            self.position = 0
            self.entry_price = 0
            self.stop_loss = 0
            self.take_profit = 0
            self.highest_price = 0
            self.lowest_price = 0
            self.max_profit_pct = 0
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
