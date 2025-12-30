import asyncio
from typing import Dict, List, Optional
from datetime import datetime
from loguru import logger
import yaml
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from ib_insync import IB, Stock, MarketOrder, LimitOrder
import pandas as pd


class AlpacaExecutor:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        api_key = self.config['api']['alpaca_key']
        secret_key = self.config['api']['alpaca_secret']
        base_url = self.config['api']['alpaca_base_url']
        
        self.client = TradingClient(api_key, secret_key, paper=True)
        self.execution_config = self.config['execution']
        
        logger.info("Alpaca executor initialized")
    
    def get_account(self) -> Dict:
        account = self.client.get_account()
        
        account_info = {
            'cash': float(account.cash),
            'portfolio_value': float(account.portfolio_value),
            'buying_power': float(account.buying_power),
            'equity': float(account.equity)
        }
        
        logger.info(f"Account value: ${account_info['portfolio_value']:,.2f}")
        
        return account_info
    
    def get_positions(self) -> pd.DataFrame:
        positions = self.client.get_all_positions()
        
        if not positions:
            return pd.DataFrame()
        
        positions_data = []
        for pos in positions:
            positions_data.append({
                'symbol': pos.symbol,
                'qty': float(pos.qty),
                'market_value': float(pos.market_value),
                'avg_entry_price': float(pos.avg_entry_price),
                'current_price': float(pos.current_price),
                'unrealized_pl': float(pos.unrealized_pl),
                'unrealized_plpc': float(pos.unrealized_plpc)
            })
        
        return pd.DataFrame(positions_data)
    
    async def place_market_order(self, symbol: str, qty: float, side: str) -> Optional[str]:
        try:
            order_side = OrderSide.BUY if side.upper() == 'BUY' else OrderSide.SELL
            
            order_request = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.DAY
            )
            
            order = self.client.submit_order(order_request)
            
            logger.success(f"Market order placed: {side} {qty} {symbol}, Order ID: {order.id}")
            
            return order.id
            
        except Exception as e:
            logger.error(f"Failed to place market order for {symbol}: {e}")
            return None
    
    async def place_limit_order(self, symbol: str, qty: float, side: str, limit_price: float) -> Optional[str]:
        try:
            order_side = OrderSide.BUY if side.upper() == 'BUY' else OrderSide.SELL
            
            order_request = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.DAY,
                limit_price=limit_price
            )
            
            order = self.client.submit_order(order_request)
            
            logger.success(f"Limit order placed: {side} {qty} {symbol} @ ${limit_price}, Order ID: {order.id}")
            
            return order.id
            
        except Exception as e:
            logger.error(f"Failed to place limit order for {symbol}: {e}")
            return None
    
    async def execute_trades(self, trades: Dict[str, float], current_prices: Dict[str, float]):
        for symbol, target_qty in trades.items():
            if abs(target_qty) < 0.01:
                continue
            
            side = 'BUY' if target_qty > 0 else 'SELL'
            qty = abs(target_qty)
            
            await self.place_market_order(symbol, qty, side)
            await asyncio.sleep(0.5)


class IBKRExecutor:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.ib = IB()
        self.connected = False
        self.execution_config = self.config['execution']
        
    async def connect(self):
        try:
            host = self.config['api']['ibkr_host']
            port = self.config['api']['ibkr_port']
            client_id = self.config['api']['ibkr_client_id']
            
            await self.ib.connectAsync(host, port, clientId=client_id)
            self.connected = True
            logger.success(f"Connected to IBKR at {host}:{port}")
            
        except Exception as e:
            logger.error(f"Failed to connect to IBKR: {e}")
            self.connected = False
    
    def disconnect(self):
        if self.connected:
            self.ib.disconnect()
            self.connected = False
            logger.info("Disconnected from IBKR")
    
    def get_account_summary(self) -> Dict:
        if not self.connected:
            logger.warning("Not connected to IBKR")
            return {}
        
        account_values = self.ib.accountSummary()
        
        account_dict = {}
        for item in account_values:
            account_dict[item.tag] = item.value
        
        return account_dict
    
    def get_positions(self) -> pd.DataFrame:
        if not self.connected:
            logger.warning("Not connected to IBKR")
            return pd.DataFrame()
        
        positions = self.ib.positions()
        
        if not positions:
            return pd.DataFrame()
        
        positions_data = []
        for pos in positions:
            positions_data.append({
                'symbol': pos.contract.symbol,
                'position': pos.position,
                'avg_cost': pos.avgCost
            })
        
        return pd.DataFrame(positions_data)
    
    async def place_market_order(self, symbol: str, qty: float, action: str, exchange: str = 'SMART'):
        if not self.connected:
            logger.warning("Not connected to IBKR")
            return None
        
        try:
            contract = Stock(symbol, exchange, 'USD')
            self.ib.qualifyContracts(contract)
            
            order = MarketOrder(action, qty)
            
            trade = self.ib.placeOrder(contract, order)
            
            logger.success(f"IBKR market order placed: {action} {qty} {symbol}")
            
            return trade
            
        except Exception as e:
            logger.error(f"Failed to place IBKR market order for {symbol}: {e}")
            return None
    
    async def execute_trades(self, trades: Dict[str, float]):
        if not self.connected:
            await self.connect()
        
        for symbol, target_qty in trades.items():
            if abs(target_qty) < 1:
                continue
            
            action = 'BUY' if target_qty > 0 else 'SELL'
            qty = abs(int(target_qty))
            
            await self.place_market_order(symbol, qty, action)
            await asyncio.sleep(0.5)


class ExecutionManager:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        broker = self.config['execution']['broker']
        
        if broker == 'alpaca':
            self.executor = AlpacaExecutor(config_path)
        elif broker == 'ibkr':
            self.executor = IBKRExecutor(config_path)
        else:
            self.executor = AlpacaExecutor(config_path)
        
        logger.info(f"Execution manager initialized with {broker}")
    
    async def execute_portfolio_rebalance(self, target_weights: Dict[str, float], 
                                         current_prices: Dict[str, float], 
                                         total_value: float):
        
        current_positions = self.executor.get_positions()
        
        trades = {}
        
        for symbol, target_weight in target_weights.items():
            target_value = total_value * target_weight
            target_qty = target_value / current_prices[symbol]
            
            current_qty = 0
            if not current_positions.empty:
                pos = current_positions[current_positions['symbol'] == symbol]
                if not pos.empty:
                    current_qty = pos.iloc[0]['qty']
            
            trade_qty = target_qty - current_qty
            
            if abs(trade_qty) > 0.01:
                trades[symbol] = trade_qty
        
        logger.info(f"Executing {len(trades)} trades for rebalance")
        
        await self.executor.execute_trades(trades, current_prices)
