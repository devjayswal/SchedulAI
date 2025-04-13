# utils/job_manager.py
import asyncio
from uuid import uuid4

# Each job ID maps to an asyncio.Queue of log strings
_job_queues: dict[str, asyncio.Queue[str]] = {}
_job_status: dict[str, str] = {}
_job_metadata: dict[str, dict] = {}  # Store job details (optional)

def create_job(job_type: str, metadata: dict = None) -> str:
    job_id = uuid4().hex
    _job_queues[job_id] = asyncio.Queue()
    _job_status[job_id] = "pending"
    _job_metadata[job_id] = {"type": job_type, "data": metadata}
    return job_id

def get_queue(job_id: str) -> asyncio.Queue[str]:
    return _job_queues[job_id]

def set_status(job_id: str, status: str):
    _job_status[job_id] = status

def get_status(job_id: str) -> str:
    return _job_status[job_id]

def get_metadata(job_id: str) -> dict:
    return _job_metadata.get(job_id, {})
