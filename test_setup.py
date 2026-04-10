import sys
import pandas
import numpy
import pydantic
import aiohttp
import websockets
from dotenv import load_dotenv
from rich import print

load_dotenv()

print("[green]R2 local environment is working.[/green]")
print(f"Python version: {sys.version}")
print(f"pandas version: {pandas.__version__}")
print(f"numpy version: {numpy.__version__}")
print(f"pydantic version: {pydantic.__version__}")
