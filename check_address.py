
from eth_account import Account
import os
from dotenv import load_dotenv

load_dotenv()
pk = os.getenv("HYPERLIQUID_PRIVATE_KEY")
if pk:
    acc = Account.from_key(pk)
    print(f"Address: {acc.address}")
else:
    print("No PK found")
