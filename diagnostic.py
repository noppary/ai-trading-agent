
import asyncio
import os
import json
from dotenv import load_dotenv
import sys
import pathlib

# Add src to path
sys.path.append(str(pathlib.Path(__file__).parent))
from src.trading.hyperliquid_api import HyperliquidAPI

async def diagnostic():
    load_dotenv()
    api = HyperliquidAPI()
    
    print("--- User State Diagnostic ---")
    state = await api.get_user_state()
    print(f"Balance: {state['balance']}")
    print(f"Total Value: {state['total_value']}")
    print(f"Positions Count: {len(state['positions'])}")
    
    for i, pos in enumerate(state['positions']):
        print(f"Position [{i}]: {pos}")
        coin = pos.get('coin')
        szi = pos.get('szi')
        print(f"  coin: {coin}, szi: {szi} (type: {type(szi)})")
    
    print("\n--- Open Orders Diagnostic ---")
    orders = await api.get_open_orders()
    print(f"Orders Count: {len(orders)}")
    for i, o in enumerate(orders):
        print(f"Order [{i}]: {o}")

if __name__ == "__main__":
    asyncio.run(diagnostic())
