
from typing import Dict, Any, Optional, List
import uuid

class MockKalshiClient:
    def __init__(self):
        self.balance = 10000.0  # $10,000 start balance
        self.positions = []
        self.orders = []
        self.fills = []
        self.markets = []
        
    async def get_balance(self) -> Dict[str, Any]:
        return {"balance": int(self.balance * 100)}  # Returns in cents
        
    async def get_positions(self, ticker: Optional[str] = None) -> Dict[str, Any]:
        if ticker:
            return {"positions": [p for p in self.positions if p['ticker'] == ticker]}
        return {"positions": self.positions}
        
    async def place_order(
        self,
        ticker: str,
        client_order_id: str,
        side: str,
        action: str,
        count: int,
        type_: str = "market",
        yes_price: Optional[int] = None,
        no_price: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        
        # Simulate validation
        if not ticker:
            raise Exception("Invalid ticker")
            
        order = {
            "order_id": str(uuid.uuid4()),
            "client_order_id": client_order_id,
            "ticker": ticker,
            "side": side,
            "action": action,
            "count": count,
            "type": type_,
            "status": "executed",
            "created_time": "2025-01-01T00:00:00Z"
        }
        
        self.orders.append(order)
        
        # Simulate immediate execution for market orders
        if type_ == "market":
            fill_price = 50  # Default 50 cents
            cost = (count * fill_price) / 100
            
            if action == 'buy':
                if self.balance < cost:
                    return {"error": "Insufficient balance"}
                self.balance -= cost
                
                # Add/update position
                existing = next((p for p in self.positions if p['ticker'] == ticker and p['side'] == side), None)
                if existing:
                    existing['position'] += count
                    existing['fees_paid'] += 1 # dummy fee
                else:
                    self.positions.append({
                        "ticker": ticker,
                        "side": side,
                        "position": count,
                        "fees_paid": 0,
                        "cost_basis": cost
                    })
            elif action == 'sell':
                existing = next((p for p in self.positions if p['ticker'] == ticker and p['side'] == side), None)
                if not existing or existing['position'] < count:
                    return {"error": "Insufficient position"}
                
                self.balance += cost
                existing['position'] -= count
                if existing['position'] <= 0:
                    self.positions.remove(existing)
                    
        return {"order": order}

    async def get_market(self, ticker: str) -> Dict[str, Any]:
        return {
            "market": {
                "ticker": ticker,
                "status": "active",
                "yes_price": 50,
                "no_price": 50,
                "liquidity": 10000,
                "volume": 5000
            }
        }
