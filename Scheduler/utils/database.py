from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv
import logging

# Load environment variables from .env file
load_dotenv()

MONGO_DB_URI = os.getenv("MONGO_DB_URI")

# Add logging to debug connection issues
logger = logging.getLogger(__name__)

# Debug: Print all environment variables that start with MONGO
mongo_vars = {k: v for k, v in os.environ.items() if 'MONGO' in k.upper()}
logger.info(f"MongoDB environment variables: {mongo_vars}")

if not MONGO_DB_URI:
    logger.error("MONGO_DB_URI not found in environment variables!")
    logger.error(f"Available env vars with 'MONGO': {mongo_vars}")
    raise ValueError("MONGO_DB_URI environment variable is required")

logger.info(f"Connecting to MongoDB with URI: {MONGO_DB_URI[:50]}...")

try:
    client = AsyncIOMotorClient(MONGO_DB_URI)
    db = client["scheduler"]
    logger.info("MongoDB client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize MongoDB client: {e}")
    raise

# Test connection function
async def test_connection():
    """Test MongoDB connection"""
    try:
        # Test the connection
        await client.admin.command('ping')
        logger.info("MongoDB connection test successful")
        return True
    except Exception as e:
        logger.error(f"MongoDB connection test failed: {e}")
        return False
