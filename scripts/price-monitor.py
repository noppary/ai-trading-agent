"""Price monitor one-shot: prints current price + funding for each configured asset."""
import asyncio
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.trading.hyperliquid_api import HyperliquidAPI
from src.config_loader import CONFIG


async def main():
    raw = CONFIG.get("assets", "BTC ETH SOL")
    assets = [a.strip() for a in raw.replace(",", " ").split() if a.strip()]

    hl = HyperliquidAPI()
    for asset in assets:
        try:
            price = await hl.get_current_price(asset)
            funding = await hl.get_funding_rate(asset)
            annualized = round(funding * 24 * 365 * 100, 2) if funding else None
            print(f"{asset}: price={price}, funding={funding}, annualized={annualized}%")
        except Exception as e:
            print(f"{asset}: ERROR {e}")


asyncio.run(main())
