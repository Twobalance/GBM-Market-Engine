
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime
import uuid

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
st.set_page_config(layout="wide", page_title="GBM Market Framework", page_icon=" ")

INITIAL_BALANCE = 10000.0
SYMBOL = "XYZ/USDT"
DT = 1/365/24/60/60  # 1 second time step in years (approx) for GBM
MU = 0.1  # Drift
SIGMA = 2.0  # Volatility (Increased for more action)
JUMP_INTENSITY = 0.05 # Higher probability of jump
JUMP_MEAN = 0.0 
JUMP_STD = 0.005 # ~0.5% jumps (approx $250 moves at $50k)

# ==========================================
# CORE CLASSES
# ==========================================

class MarketEngine:
    def __init__(self):
        if 'market_data' not in st.session_state:
            # Initialize with some history
            self.price = 500.0
            st.session_state.market_data = pd.DataFrame(columns=['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Buy_Volume', 'Sell_Volume'])
            self.generate_history(100) # seeded history
        else:
            self.price = st.session_state.market_data.iloc[-1]['Close'] if not st.session_state.market_data.empty else 500.0
            if not st.session_state.market_data.empty:
                self.last_time = st.session_state.market_data.iloc[-1]['Time']
            else:
                 self.last_time = datetime.now()

    def generate_history(self, periods):
        data = []
        current_p = self.price
        now = datetime.now()
        
        # Go back in time
        start_time = now - pd.Timedelta(seconds=periods)
        
        for i in range(periods):
            # Previous Close is Open
            open_p = current_p
            
            # Simulate "Micro-ticks" within the candle to form High/Low
            high_p = open_p
            low_p = open_p
            temp_p = open_p
            
            # 10 micro-steps per candle
            for _ in range(10):
                drift = (MU - 0.5 * SIGMA**2) * (DT/10)
                shock = SIGMA * np.sqrt(DT/10) * np.random.normal()
                
                # Jump (lower prob per micro-step)
                if np.random.random() < JUMP_INTENSITY/10:
                    jump = np.random.normal(JUMP_MEAN, JUMP_STD)
                    temp_p *= np.exp(jump)
                
                temp_p *= np.exp(drift + shock)
                high_p = max(high_p, temp_p)
                low_p = min(low_p, temp_p)
            
            close_p = temp_p
            current_p = close_p
            
            # Volume based on volatility (range)
            candle_range = abs(high_p - low_p) / open_p
            base_vol = 100 + (candle_range * 1000000) # Arbitrary scale
            volume = base_vol * np.random.uniform(0.8, 1.2)
            
            # Split Volume (Buy vs Sell)
            # If price went up, favor buy volume
            price_change_pct = (close_p - open_p) / open_p
            # Sigmoid-ish split around 0.5
            buy_bias = 0.5 + (price_change_pct * 100) # amplify small moves
            buy_bias = max(0.1, min(0.9, buy_bias)) # Cap between 10% and 90%
            
            # Add some randomness to the bias
            buy_bias += np.random.uniform(-0.1, 0.1)
            buy_bias = max(0.1, min(0.9, buy_bias))
            
            buy_vol = volume * buy_bias
            sell_vol = volume - buy_vol
            
            data.append({
                'Time': start_time + pd.Timedelta(seconds=i),
                'Open': open_p, 'High': high_p, 'Low': low_p, 'Close': close_p,
                'Volume': volume,
                'Buy_Volume': buy_vol,
                'Sell_Volume': sell_vol
            })
            
        df = pd.DataFrame(data)
        st.session_state.market_data = df # Replace init
        self.price = current_p
        self.last_time = data[-1]['Time']

    def tick(self):
        # Generate next candle with micro-ticks
        last_close = self.price
        open_p = last_close
        
        # Get base volatility from session state
        base_sigma = st.session_state.get('volatility', SIGMA)
        
        # Dynamic Volatility: Increase when price is going up (momentum effect)
        # Calculate recent trend from last few candles
        if len(st.session_state.market_data) >= 5:
            recent_prices = st.session_state.market_data['Close'].tail(5).values
            price_trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]  # % change over 5 candles
            
            # Volatility multiplier: increases with upward movement
            # When price goes up, volatility increases (momentum/FOMO effect)
            # When price goes down, volatility stays normal or slightly lower
            if price_trend > 0:
                # Upward trend: volatility increases proportionally (up to 2x)
                vol_multiplier = 1.0 + min(abs(price_trend) * 50, 1.0)  # Max 2x volatility
            else:
                # Downward trend: slight volatility decrease (fear/caution)
                vol_multiplier = max(0.7, 1.0 - abs(price_trend) * 20)  # Min 0.7x volatility
            
            current_sigma = base_sigma * vol_multiplier
        else:
            current_sigma = base_sigma
        
        # Store current dynamic volatility for display
        st.session_state['current_dynamic_vol'] = current_sigma
        
        high_p = open_p
        low_p = open_p
        temp_p = open_p
        
        # 10 micro-steps
        for _ in range(10):
            drift = (MU - 0.5 * current_sigma**2) * (DT/10)
            shock = current_sigma * np.sqrt(DT/10) * np.random.normal()
            
            if np.random.random() < JUMP_INTENSITY/10:
                jump = np.random.normal(JUMP_MEAN, JUMP_STD)
                temp_p *= np.exp(jump)
                
            temp_p *= np.exp(drift + shock)
            high_p = max(high_p, temp_p)
            low_p = min(low_p, temp_p)
            
        new_close = temp_p
        self.price = new_close
        
        # Strict Timestamping
        new_time = self.last_time + pd.Timedelta(seconds=1)
        self.last_time = new_time
        
        # Volume
        candle_range = abs(high_p - low_p) / open_p
        base_vol = 100 + (candle_range * 1000000)
        volume = base_vol * np.random.uniform(0.8, 1.2)
        
        # Split Volume
        price_change_pct = (new_close - open_p) / open_p
        buy_bias = 0.5 + (price_change_pct * 100) 
        buy_bias = max(0.1, min(0.9, buy_bias))
        buy_bias += np.random.uniform(-0.1, 0.1)
        buy_bias = max(0.1, min(0.9, buy_bias))
        
        buy_vol = volume * buy_bias
        sell_vol = volume - buy_vol
        
        new_row = {
            'Time': new_time,
            'Open': open_p,
            'High': high_p,
            'Low': low_p,
            'Close': new_close,
            'Volume': volume,
            'Buy_Volume': buy_vol,
            'Sell_Volume': sell_vol
        }
        
        df = pd.DataFrame([new_row])
        st.session_state.market_data = pd.concat([st.session_state.market_data, df], ignore_index=True)
        
        if len(st.session_state.market_data) > 300: # Increase buffer for MA
             st.session_state.market_data = st.session_state.market_data.iloc[-300:]
             
        return self.price

class TradingEngine:
    def __init__(self):
        if 'wallet' not in st.session_state:
            st.session_state.wallet = {
                'balance': INITIAL_BALANCE,
                'used_margin': 0.0,
                'pnl': 0.0,
                'equity': INITIAL_BALANCE
            }
        if 'positions' not in st.session_state:
            st.session_state.positions = []
        
        if 'orders' not in st.session_state:
            st.session_state.orders = []
        
        if 'trade_counter' not in st.session_state:
            st.session_state.trade_counter = 0

    def place_order(self, side, amount, leverage, margin_mode, current_price):
        margin_needed = (amount * current_price) / leverage
        
        # Basic check
        available = st.session_state.wallet['balance'] - st.session_state.wallet['used_margin'] 
        # Note: Cross margin logic simplifies availability check to Equity, we stick to balance for simplification or use specific cross logic
        
        # Adjust implementation for Cross/Isolated
        if margin_mode == 'Isolated':
            if margin_needed > available:
                return False, "Insufficient Margin"
        else: # Cross
             # In cross, we look at total equity
             if margin_needed > (st.session_state.wallet['equity'] - st.session_state.wallet['used_margin']):
                  return False, "Insufficient Equity"

        st.session_state.trade_counter += 1
        trade_id = f"Trade {st.session_state.trade_counter}"
        
        position = {
            'id': trade_id,
            'uuid': str(uuid.uuid4())[:8],
            'symbol': f"{st.session_state.get('symbol_name', 'XYZ')}/USDT",
            'side': side,
            'size': amount,
            'entry_price': current_price,
            'leverage': leverage,
            'margin_mode': margin_mode,
            'initial_margin': margin_needed,
            'liq_price': 0.0,
            'pnl': 0.0,
            'active': True
        }
        
        # Calculate Liquidation Price
        # Long: Entry * (1 - 1/Lev + MMR)
        # Short: Entry * (1 + 1/Lev - MMR)
        mmr = 0.005 # 0.5% Maintenance Margin
        
        if margin_mode == 'Isolated':
            if side == 'Long':
                position['liq_price'] = current_price * (1 - 1/leverage + mmr)
            else:
                position['liq_price'] = current_price * (1 + 1/leverage - mmr)
        else:
            # Cross Liq is dynamic based on account balance, simplified here to static estimation or updated in loop
            # For this MVP, we will calculate a static approximation for Cross or update it in `update_positions`
            # Approximate for simplicity: share the total balance
            position['liq_price'] = 0.0 # Will calculate dynamically

        st.session_state.positions.append(position)
        st.session_state.wallet['used_margin'] += margin_needed
        return True, "Order Filled"

    def update_positions(self, current_price):
        total_pnl = 0.0
        active_positions = []
        
        wallet = st.session_state.wallet
        equity = wallet['balance'] # Start with static balance
        
        # First Calc PnL
        for pos in st.session_state.positions:
            if not pos['active']: continue
            
            diff = current_price - pos['entry_price']
            if pos['side'] == 'Short':
                diff = -diff
            
            pnl = diff * pos['size']
            pos['pnl'] = pnl
            total_pnl += pnl
        
        # Update Equity
        wallet['pnl'] = total_pnl
        wallet['equity'] = wallet['balance'] + total_pnl
        
        # Check Liquidation
        mmr = 0.005 # Maintenance Margin Requirement

        for pos in st.session_state.positions:
            if not pos['active']: continue
            
            liquidated = False
            
            if pos['margin_mode'] == 'Isolated':
                # Margin Balance = Initial Margin + PnL
                margin_balance = pos['initial_margin'] + pos['pnl']
                # Maintenance Margin Requirement = Position Value * MMR
                maint_margin_req = (pos['size'] * current_price) * mmr
                
                if margin_balance < maint_margin_req:
                    liquidated = True
                    
            else: # Cross
                # Cross Margin Ratio = Equity / Total Maint Margin
                # If Equity < Total Maint Margin -> Liquidate ALL (Simplified) or liquidate largest loser
                # Here we check if Equity is effectively gone or below maintenance for this pos
                # Simplified: If Equity < 0, boom. Or stricter:
                # If Equity < Total Positions Value * MMR
                
                # Check if specific position is dragging account under
                # We often treat Cross as "Account Level Liquidation"
                pass 

        # Global Cross Check
        total_position_value = sum([p['size'] * current_price for p in st.session_state.positions if p['active'] and p['margin_mode'] == 'Cross'])
        total_maint_margin = total_position_value * mmr
        
        # If Cross positions exist and Equity < Total Maint Margin (for cross positions) -> Liquidate Cross
        cross_positions = [p for p in st.session_state.positions if p['active'] and p['margin_mode'] == 'Cross']
        if cross_positions and wallet['equity'] < total_maint_margin:
             for pos in cross_positions:
                 self.liquidate(pos, current_price)
                 pos['active'] = False
                 # Actually need to capture the realized loss
        
        # Handle Isolated deletions
        for pos in st.session_state.positions:
            if pos['active']:
                if pos['margin_mode'] == 'Isolated':
                    # Re-check updated PnL logic above
                    margin_balance = pos['initial_margin'] + pos['pnl']
                    maint_margin_req = (pos['size'] * current_price) * mmr
                    if margin_balance < maint_margin_req:
                        self.liquidate(pos, current_price)
                        pos['active'] = False
                active_positions.append(pos)
                
        st.session_state.positions = active_positions

    def liquidate(self, pos, price):
        # Realize the loss (Position Margin is lost)
        # For Isolated: Loss is capped at Margin.
        # For Cross: Loss is taken from Balance.
        loss = 0
        if pos['margin_mode'] == 'Isolated':
            loss = -pos['initial_margin'] # Lose the margin
            # In update, we already removed it from balance if we finalize? No, balance is static.
            # We must subtract from Balance.
            st.session_state.wallet['balance'] += (pos['pnl'] if pos['pnl'] > -pos['initial_margin'] else -pos['initial_margin'])
            st.session_state.wallet['used_margin'] -= pos['initial_margin']
            
        else: # Cross
            st.session_state.wallet['balance'] += pos['pnl'] 
            st.session_state.wallet['used_margin'] -= pos['initial_margin']
            
        st.warning(f"Position {pos['id']} LIQUIDATED at {price:.2f}")

    def close_position(self, pos_id, current_price):
        for pos in st.session_state.positions:
            if pos['id'] == pos_id:
                # Realize PnL
                st.session_state.wallet['balance'] += pos['pnl']
                st.session_state.wallet['used_margin'] -= pos['initial_margin']
                pos['active'] = False
                st.session_state.positions.remove(pos)
                return

    def close_all_positions(self):
        for pos in st.session_state.positions[:]:
             # Realize PnL
             st.session_state.wallet['balance'] += pos['pnl']
             st.session_state.wallet['used_margin'] -= pos['initial_margin']
             pos['active'] = False
             st.session_state.positions.remove(pos)

class OrderBookEngine:
    def __init__(self):
        self.last_trade_time = datetime.now()
        if 'recent_trades' not in st.session_state:
            st.session_state.recent_trades = []

    def generate_depth(self, current_price):
        # Generate synthetic order book depth
        # 10 levels of bids and asks
        
        bids = []
        asks = []
        
        # Bids (Prices below current)
        current_bid = current_price - (current_price * 0.0001) # Spread
        sum_size = 0.0
        for i in range(12):
            price = current_bid - (i * 0.1) # 10 cent steps approx
            # Add some randomness to steps
            price -= np.random.uniform(0, 0.5)
            
            size = np.random.uniform(0.001, 2.0)
            
            # Flash order size (occasional whale)
            if np.random.random() < 0.05:
                size *= 10
                
            sum_size += size
            bids.append({
                'price': price,
                'size': size,
                'sum': sum_size
            })
            
        # Asks (Prices above current)
        current_ask = current_price + (current_price * 0.0001) # Spread
        sum_size = 0.0
        for i in range(12):
            price = current_ask + (i * 0.1) 
            price += np.random.uniform(0, 0.5)
            
            size = np.random.uniform(0.001, 2.0)
            if np.random.random() < 0.05:
                size *= 10
                
            sum_size += size
            asks.append({
                'price': price,
                'size': size,
                'sum': sum_size
            })
        
        # Asks should be reversed for display (High to Low) but we generate Low to High
        # For display we want Lowest Ask at the bottom (closest to price)
        # So sorting: Asks (ASC) is correct for generation, but UI needs DESC (Highest on top).
        
        return pd.DataFrame(bids), pd.DataFrame(asks)

    def generate_recent_trades(self, current_price, price_movement_detected=False):
        # Only add a trade if there was a price update or randomly
        if price_movement_detected or np.random.random() < 0.2:
            now = datetime.now()
            
            # Determine side based on price direction or random
            side = 'Buy' if np.random.random() > 0.5 else 'Sell'
            
            # Trade Price closely matches current
            trade_price = current_price
            
            amount = np.random.uniform(0.0005, 0.5)
            
            new_trade = {
                'price': trade_price,
                'amount': amount,
                'time': now.strftime("%H:%M:%S"),
                'side': side
            }
            
            st.session_state.recent_trades.insert(0, new_trade)
            if len(st.session_state.recent_trades) > 25:
                st.session_state.recent_trades.pop()

# ==========================================
# UI & MAIN LOOP
# ==========================================

def main():
    market = MarketEngine()
    trader = TradingEngine()
    ob_engine = OrderBookEngine()
    
    # --- SIDEBAR ---
    st.sidebar.title("GBM Market Framework")
    
    # Settings (Volatility)
    with st.sidebar.expander("⚙️ Market Settings", expanded=True):
        # Symbol Name
        if 'symbol_name' not in st.session_state:
            st.session_state.symbol_name = "XYZ"
        
        st.session_state.symbol_name = st.text_input(
            "Symbol Name",
            value=st.session_state.symbol_name,
            max_chars=10,
            help="Enter your custom symbol name (e.g., BTC, ETH, AAPL)"
        )
        
        if 'volatility' not in st.session_state:
            st.session_state.volatility = SIGMA
        
        # Two options: Slider for quick adjustment, Number input for precise
        vol_col1, vol_col2 = st.columns([2, 1])
        
        with vol_col1:
            st.session_state.volatility = st.slider(
                "Volatility (σ)", 
                min_value=0.01, max_value=10.0, value=st.session_state.volatility, step=0.01,
                help="Higher = more chaotic. Low (0.1-0.5) = stable, High (3-10) = wild swings"
            )
        
        with vol_col2:
            precise_vol = st.number_input(
                "Exact σ",
                min_value=0.01, max_value=50.0, value=st.session_state.volatility, step=0.01,
                help="Enter any value 0.01 to 50"
            )
            if precise_vol != st.session_state.volatility:
                st.session_state.volatility = precise_vol
        
        # Quick presets - use selectbox instead of buttons for compact layout
        preset = st.selectbox(
            "Quick Presets",
            options=["Custom", "Low (0.3)", "Medium (1.0)", "High (3.0)", "Extreme (8.0)"],
            index=0,
            label_visibility="collapsed"
        )
        if preset == "Low (0.3)":
            st.session_state.volatility = 0.3
            st.rerun()
        elif preset == "Medium (1.0)":
            st.session_state.volatility = 1.0
            st.rerun()
        elif preset == "High (3.0)":
            st.session_state.volatility = 3.0
            st.rerun()
        elif preset == "Extreme (8.0)":
            st.session_state.volatility = 8.0
            st.rerun()
        
        col1, col2 = st.columns(2)
        if col1.button("FLASH CRASH 📉"):
             st.session_state.market_data.loc[len(st.session_state.market_data)-1, 'Close'] *= 0.95
             market.price *= 0.95
             st.toast("Flash Crash! -5%")
             
        if col2.button("PUMP IT 🚀"):
             st.session_state.market_data.loc[len(st.session_state.market_data)-1, 'Close'] *= 1.05
             market.price *= 1.05
             st.toast("Pump Triggered! +5%")

    # Moving Averages Settings
    with st.sidebar.expander("📈 Indicators", expanded=False):
        ma_input = st.text_input("Moving Averages (comma sep)", "20, 50", help="e.g. 20, 50, 200")
        try:
            ma_periods = [int(p.strip()) for p in ma_input.split(',') if p.strip().isdigit()]
        except:
            ma_periods = [20, 50]
            
    # Wallet Info
    w = st.session_state.wallet
    st.sidebar.metric("Equity (Est)", f"${w['equity']:.2f}", delta=f"{w['pnl']:.2f}")
    st.sidebar.text(f"Wallet Balance: ${w['balance']:.2f}")
    st.sidebar.text(f"Used Margin: ${w['used_margin']:.2f}")
    st.sidebar.text(f"Available: ${w['equity'] - w['used_margin']:.2f}")
    
    st.sidebar.markdown("---")
    
    # Order Entry
    st.sidebar.subheader("Place Order")
    side = st.sidebar.radio("Side", ["Long", "Short"], horizontal=True)
    margin_mode = st.sidebar.selectbox("Margin Mode", ["Isolated", "Cross"])
    leverage = st.sidebar.slider("Leverage", 1, 50, 10)
    
    # Size Input Type
    size_type = st.sidebar.radio("Size Type", ["XYZ", "USD", "% Balance"], horizontal=True)
    
    current_price = market.price
    
    amount = 0.0
    final_size_xyz = 0.0
    
    if size_type == "XYZ":
        amount = st.sidebar.number_input("Size (XYZ)", min_value=0.001, max_value=100.0, value=0.1, step=0.01)
        final_size_xyz = amount
    elif size_type == "USD":
        amount = st.sidebar.number_input("Size (USD)", min_value=10.0, max_value=1000000.0, value=1000.0, step=100.0)
        final_size_xyz = amount / current_price
    else:
        pct = st.sidebar.slider("Use % of Available", 1, 100, 50)
        # Available for margin
        available = st.session_state.wallet['equity'] - st.session_state.wallet['used_margin']
        margin_to_use = available * (pct / 100.0)
        # Position Size = Margin * Leverage
        pos_size_usd = margin_to_use * leverage
        final_size_xyz = pos_size_usd / current_price
        
        st.sidebar.caption(f"Margin: ${margin_to_use:.2f}")
        st.sidebar.caption(f"Pos Size: ${pos_size_usd:.2f} ({final_size_xyz:.4f} XYZ)")

    if st.sidebar.button("Execute Order"):
        success, msg = trader.place_order(side, final_size_xyz, leverage, margin_mode, current_price)
        if success:
            st.sidebar.success(msg)
        else:
            st.sidebar.error(msg)
            
    # --- MAIN AREA ---
    
    # Real-time Updater
    current_price = market.tick()
    trader.update_positions(current_price)
    ob_engine.generate_recent_trades(current_price, price_movement_detected=True)
    
    # View Settings
    st.sidebar.markdown("---")
    st.sidebar.subheader("👁️ View")
    full_screen = st.sidebar.checkbox("Full Screen Chart", value=False, help="Hide Order Book for a larger chart")

    # Layout: Coin Chart Left, OrderBook Right
    if full_screen:
        col_chart = st.container()
        col_ob = None
    else:
        col_chart, col_ob = st.columns([0.75, 0.25])
    
    with col_chart:
        # 1. CHART
        df = st.session_state.market_data
        
        # Calculate Dynamic MAs
        for p in ma_periods:
            df[f'SMA_{p}'] = df['Close'].rolling(window=p).mean()

        # Create Subplots: Row 1 = Price, Row 2 = Volume (TradingView Style)
        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.03, 
            row_heights=[0.75, 0.25],
            subplot_titles=('', '')
        )

        # TradingView Color Scheme
        TV_BG_COLOR = '#131722'
        TV_GRID_COLOR = '#1e222d'
        TV_TEXT_COLOR = '#787b86'
        TV_GREEN = '#26a69a'
        TV_RED = '#ef5350'
        TV_BORDER_COLOR = '#2a2e39'

        # Candlesticks - TradingView style (thin wicks, clean look)
        fig.add_trace(go.Candlestick(
            x=df['Time'],
            open=df['Open'], 
            high=df['High'],
            low=df['Low'], 
            close=df['Close'],
            name='',
            increasing=dict(line=dict(color=TV_GREEN, width=1), fillcolor=TV_GREEN),
            decreasing=dict(line=dict(color=TV_RED, width=1), fillcolor=TV_RED),
            whiskerwidth=0,
            showlegend=False
        ), row=1, col=1)
        
        # Add SMA Traces with TradingView-like colors
        tv_ma_colors = ['#f7a21b', '#2962ff', '#e91e63', '#9c27b0', '#00bcd4', '#ff5722']
        for i, p in enumerate(ma_periods):
            color = tv_ma_colors[i % len(tv_ma_colors)]
            fig.add_trace(go.Scatter(
                x=df['Time'], 
                y=df[f'SMA_{p}'], 
                mode='lines', 
                name=f'MA {p}', 
                line=dict(color=color, width=1),
                hoverinfo='skip'
            ), row=1, col=1)
        
        # Volume Bars - TradingView style (colored by candle direction)
        colors = [TV_GREEN if df.iloc[i]['Close'] >= df.iloc[i]['Open'] else TV_RED 
                  for i in range(len(df))]
        
        fig.add_trace(go.Bar(
            x=df['Time'], 
            y=df['Volume'], 
            name='Volume',
            marker=dict(
                color=colors,
                opacity=0.5
            ),
            showlegend=False
        ), row=2, col=1)
        
        # TradingView Layout Styling
        fig.update_layout(
            # Remove title for cleaner look (price shown in separate element)
            title=None,
            
            # Dark theme matching TradingView
            paper_bgcolor=TV_BG_COLOR,
            plot_bgcolor=TV_BG_COLOR,
            
            height=650,
            margin=dict(l=0, r=60, t=10, b=30),
            
            # No range slider
            xaxis_rangeslider_visible=False,
            
            # Legend styling
            showlegend=True,
            legend=dict(
                yanchor="top", 
                y=0.99, 
                xanchor="left", 
                x=0.01,
                bgcolor='rgba(19, 23, 34, 0.8)',
                bordercolor=TV_BORDER_COLOR,
                borderwidth=1,
                font=dict(color=TV_TEXT_COLOR, size=10)
            ),
            
            # Hover mode - TradingView uses crosshair
            hovermode='x unified',
            
            # Preserve zoom/pan state
            uirevision='tradingview_chart',
            
            # Font styling
            font=dict(family='Trebuchet MS, sans-serif', color=TV_TEXT_COLOR),
            
            # Spike lines (crosshair effect)
            hoverlabel=dict(
                bgcolor=TV_BORDER_COLOR,
                bordercolor=TV_BORDER_COLOR,
                font=dict(color='white', size=11)
            )
        )
        
        # X-Axis styling (shared)
        for xaxis in ['xaxis', 'xaxis2']:
            fig.update_layout(**{
                xaxis: dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor=TV_GRID_COLOR,
                    showline=True,
                    linewidth=1,
                    linecolor=TV_BORDER_COLOR,
                    tickfont=dict(color=TV_TEXT_COLOR, size=10),
                    tickformat='%H:%M:%S',
                    spikemode='across',
                    spikesnap='cursor',
                    spikecolor=TV_TEXT_COLOR,
                    spikethickness=0.5,
                    spikedash='solid',
                    showspikes=True,
                    zeroline=False
                )
            })
        
        # Y-Axis styling - Price (Row 1)
        fig.update_layout(
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor=TV_GRID_COLOR,
                showline=True,
                linewidth=1,
                linecolor=TV_BORDER_COLOR,
                tickfont=dict(color=TV_TEXT_COLOR, size=10),
                side='right',
                tickformat='.2f',
                spikemode='across',
                spikesnap='cursor',
                spikecolor=TV_TEXT_COLOR,
                spikethickness=0.5,
                spikedash='solid',
                showspikes=True,
                zeroline=False,
                fixedrange=False
            ),
            # Y-Axis styling - Volume (Row 2)
            yaxis2=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor=TV_GRID_COLOR,
                showline=True,
                linewidth=1,
                linecolor=TV_BORDER_COLOR,
                tickfont=dict(color=TV_TEXT_COLOR, size=9),
                side='right',
                tickformat='.0f',
                zeroline=False,
                showticklabels=True,
                fixedrange=True
            )
        )
        
        # Add "Vol" label for volume pane
        fig.add_annotation(
            text="Vol",
            xref="paper", yref="paper",
            x=0.01, y=0.22,
            showarrow=False,
            font=dict(color=TV_TEXT_COLOR, size=10),
            bgcolor=TV_BG_COLOR
        )
        
        # Add positions entry lines (TradingView style)
        for pos in st.session_state.positions:
            line_color = '#089981' if pos['side'] == 'Long' else '#f23645'
            fig.add_hline(
                y=pos['entry_price'], 
                line_dash="dot", 
                line_color=line_color,
                line_width=1,
                annotation=dict(
                    text=f"⬤ {pos['side']} @ {pos['entry_price']:.2f}",
                    font=dict(color=line_color, size=10),
                    bgcolor='rgba(19, 23, 34, 0.8)',
                    bordercolor=line_color,
                    borderwidth=1,
                    borderpad=3,
                    xanchor='right',
                    x=0.99
                ),
                row=1, col=1
            )

        # Symbol and Price Header (TradingView style) - Change % per MINUTE (60 candles)
        # Calculate price change over last 60 seconds (1 minute)
        lookback = min(60, len(df) - 1)  # 60 seconds or available data
        if lookback > 0:
            price_1min_ago = df.iloc[-(lookback + 1)]['Close']
            price_change = df.iloc[-1]['Close'] - price_1min_ago
            price_change_pct = (price_change / price_1min_ago * 100) if price_1min_ago != 0 else 0
        else:
            price_change = 0
            price_change_pct = 0
            
        price_color = '#26a69a' if price_change >= 0 else '#ef5350'
        price_arrow = '▲' if price_change >= 0 else '▼'
        
        # Get dynamic volatility indicator
        dynamic_vol = st.session_state.get('current_dynamic_vol', SIGMA)
        base_vol = st.session_state.get('volatility', SIGMA)
        vol_ratio = dynamic_vol / base_vol if base_vol > 0 else 1.0
        vol_indicator = '🔥' if vol_ratio > 1.2 else ('❄️' if vol_ratio < 0.85 else '➖')
        
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 15px; padding: 8px 12px; background-color: #131722; border-bottom: 1px solid #2a2e39; font-family: 'Trebuchet MS', sans-serif;">
            <span style="font-size: 16px; font-weight: 600; color: white;">{st.session_state.get('symbol_name', 'XYZ')}/USDT</span>
            <span style="font-size: 18px; font-weight: 600; color: {price_color};">{current_price:.2f}</span>
            <span style="font-size: 13px; color: {price_color};">{price_arrow} {abs(price_change):.2f} ({abs(price_change_pct):.2f}%/min)</span>
            <span style="font-size: 11px; color: #787b86;" title="Dynamic Volatility: {vol_ratio:.2f}x">{vol_indicator} Vol: {vol_ratio:.1f}x</span>
            <span style="font-size: 11px; color: #787b86; margin-left: auto;">O <span style="color: white;">{df.iloc[-1]['Open']:.2f}</span></span>
            <span style="font-size: 11px; color: #787b86;">H <span style="color: white;">{df.iloc[-1]['High']:.2f}</span></span>
            <span style="font-size: 11px; color: #787b86;">L <span style="color: white;">{df.iloc[-1]['Low']:.2f}</span></span>
            <span style="font-size: 11px; color: #787b86;">C <span style="color: white;">{df.iloc[-1]['Close']:.2f}</span></span>
        </div>
        """, unsafe_allow_html=True)

        st.plotly_chart(fig, use_container_width=True)
        
        # 2. POSITIONS TABLE (Bottom of Chart)
        if st.session_state.positions:
            st.markdown("### Open Positions")
            pos_df = pd.DataFrame(st.session_state.positions)
            
            # Simple Table
            # Create a styled dataframe
            def color_pnl(val):
                if isinstance(val, (int, float)):
                    color = '#26a69a' if val >= 0 else '#ef5350'
                else:
                    color = 'white'
                return f'color: {color}'

            # We need to prep the display DF
            disp = pos_df[['id', 'side', 'size', 'entry_price', 'leverage', 'pnl']].copy()
            st.dataframe(disp.style.format({'entry_price': '{:.2f}', 'pnl': '{:.2f}'}).applymap(color_pnl, subset=['pnl']))
            
            # Close Actions
            col_actions, col_close_all = st.columns([0.8, 0.2])
            with col_close_all:
                if st.button("🔴 CLOSE ALL", use_container_width=True):
                    trader.close_all_positions()
                    st.rerun()
            
            st.caption("Individual Actions:")
            cols = st.columns(len(pos_df))
            for idx, (i, row) in enumerate(pos_df.iterrows()):
                 if cols[idx].button(f"Close {row['id']}", key=f"btn_close_{row['id']}"):
                     trader.close_position(row['id'], current_price)
                     st.rerun()

    if col_ob:
        with col_ob:
            # 3. ORDER BOOK & TRADES
            st.subheader("Order Book")
            
            bids_df, asks_df = ob_engine.generate_depth(current_price)
            
            # Sort Asks DESC (Highest price at top) for visualization
            asks_df = asks_df.sort_values('price', ascending=False)
            bids_df = bids_df.sort_values('price', ascending=False) # Highest bid at top (closest to spread)

            # Style Config
            html_code = """
    <style>
        .ob-container {
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
            background-color: #0e1117;
            padding: 5px;
        }
        .ob-row {
            display: flex;
            justify-content: space-between;
            line-height: 1.5;
        }
        .ob-header {
            color: #888;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .ask { color: #ef5350; }
        .bid { color: #26a69a; }
        .price-current {
            font-size: 18px;
            font-weight: bold;
            text-align: center;
            margin: 10px 0;
            color: #4caf50; /* Or vary based on tick */
        }
        .trades-container {
            margin-top: 20px;
        }
    </style>
    <div class="ob-container">
        <div class="ob-row ob-header">
            <span>Price</span>
            <span>Size</span>
            <span>Sum</span>
        </div>
    """
            
            # ASKS
            for i, row in asks_df.iterrows():
                html_code += f"""
    <div class="ob-row ask">
        <span>{row['price']:,.1f}</span>
        <span>{row['size']:.3f}</span>
        <span>{row['sum']:.3f}</span>
    </div>
    """
                
            # CURRENT PRICE
            # Determine color of current price (Green if Close > Open)
            last_candle = st.session_state.market_data.iloc[-1]
            price_color = '#26a69a' if last_candle['Close'] >= last_candle['Open'] else '#ef5350'
            
            html_code += f"""
    <div class="price-current" style="color: {price_color}">
        {current_price:,.1f} <span style="font-size: 10px; color: #888;">USD</span>
    </div>
    """
            
            # BIDS
            for i, row in bids_df.iterrows():
                html_code += f"""
    <div class="ob-row bid">
        <span>{row['price']:,.1f}</span>
        <span>{row['size']:.3f}</span>
        <span>{row['sum']:.3f}</span>
    </div>
    """
                
            html_code += "</div>"
            st.markdown(html_code, unsafe_allow_html=True)
            
            # RECENT TRADES custom standard
            st.subheader("Trades")
            
            trades_html = """
    <div class="ob-container trades-container">
        <div class="ob-row ob-header">
            <span>Price</span>
            <span>Amt</span>
            <span>Time</span>
        </div>
    """
            
            for trade in st.session_state.recent_trades[:15]:
                color = '#26a69a' if trade['side'] == 'Buy' else '#ef5350'
                trades_html += f"""
    <div class="ob-row">
        <span style="color: {color}">{trade['price']:,.1f}</span>
        <span>{trade['amount']:.4f}</span>
        <span style="color: #888">{trade['time']}</span>
    </div>
    """
                
            trades_html += "</div>"
            st.markdown(trades_html, unsafe_allow_html=True)

    # Auto-refresh loop
    time.sleep(1) 
    st.rerun()

if __name__ == "__main__":
    main()
