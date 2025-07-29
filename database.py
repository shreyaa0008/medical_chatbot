from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv
import logging
from typing import Optional

logger = logging.getLogger(__name__)

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB = os.getenv("MONGODB_DB")

if not MONGODB_URI or not MONGODB_DB:
    raise ValueError("MONGODB_URI and MONGODB_DB must be set in .env file")

# Global variables
client = None
db = None
users_collection = None

async def init_db():
    global client, db, users_collection
    try:
        client = AsyncIOMotorClient(MONGODB_URI)
        # Test the connection
        await client.admin.command('ping')
        logger.info("Successfully connected to MongoDB")
        
        db = client[MONGODB_DB]
        users_collection = db.get_collection("users")
        
        # Create indexes
        await users_collection.create_index("username", unique=True)
        await users_collection.create_index("email", unique=True)
        logger.info("Database indexes created successfully")
        
        return client, db, users_collection
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")
        raise

async def close_db():
    global client
    if client:
        client.close()
        logger.info("Database connection closed")

# Initialize the basic connection (without async operations)
client = AsyncIOMotorClient(MONGODB_URI)
db = client[MONGODB_DB]
users_collection = db.get_collection("users")
doctors_collection = db.get_collection("doctors")