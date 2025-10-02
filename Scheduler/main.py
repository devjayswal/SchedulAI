import os
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
# from routes.users import router as user_router
# from routes.faculty import router as faculty_router
# from routes.classroom import router as classroom_router
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from utils.log_generator import log_generator
import time
from utils.job_manager import create_job, get_queue, set_status, get_status
from routes.timetable import router as timetable_router
from routes.status import router as status_router
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PORT = int(os.getenv("PORT", 8000))  # Default to 8000 if not found

app = FastAPI(title="Timetable Scheduler API", version="1.0.0")
app.mount("/static", StaticFiles(directory="public"), name="static")

@app.on_event("startup")
async def startup_event():
    """Test database connection on startup"""
    try:
        from utils.database import test_connection
        connection_ok = await test_connection()
        if connection_ok:
            logger.info("✅ Database connection successful")
        else:
            logger.error("❌ Database connection failed")
    except Exception as e:
        logger.error(f"❌ Database connection error: {e}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
        "*"  # Allow all origins for development
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# app.include_router(user_router)
# app.include_router(faculty_router)
# app.include_router(classroom_router)
app.include_router(timetable_router)
app.include_router(status_router)

@app.get("/")
async def root():
    return FileResponse(os.path.join("public", "index.html"))

@app.get("/health")
async def health_check():
    """Health check endpoint with database status"""
    try:
        from utils.database import test_connection
        db_status = await test_connection()
        return {
            "status": "healthy" if db_status else "unhealthy",
            "database": "connected" if db_status else "disconnected",
            "timestamp": time.time()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "database": "error",
            "error": str(e),
            "timestamp": time.time()
        }

@app.get("/cors-test")
async def cors_test():
    """Simple CORS test endpoint"""
    return {
        "message": "CORS is working!",
        "timestamp": time.time(),
        "origin_allowed": True
    }

@app.get("/logs/{job_id}")
async def stream_logs(job_id: str):
    try:
        queue = get_queue(job_id)
    except KeyError:
        raise HTTPException(404, "Job not found")

    async def event_generator():
        # First send status
        yield f"data: STATUS:{get_status(job_id)}\n\n"
        # Then stream log lines until "DONE"
        while True:
            msg = await queue.get()
            yield f"data: {msg}\n\n"
            if msg == "DONE":
                break

    return StreamingResponse(event_generator(), media_type="text/event-stream")
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
