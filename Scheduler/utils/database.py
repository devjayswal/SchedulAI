from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_DB_URI = os.getenv("MONGO_DB_URI")
client = AsyncIOMotorClient(MONGO_DB_URI)
db = client["scheduler"]
