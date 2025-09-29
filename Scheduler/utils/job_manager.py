# utils/job_manager.py
import asyncio
import time
from uuid import uuid4
from datetime import datetime
from typing import Dict, List, Optional

# Job tracking dictionaries
_job_queues: Dict[str, asyncio.Queue[str]] = {}
_job_status: Dict[str, str] = {}
_job_metadata: Dict[str, dict] = {}
_job_progress: Dict[str, dict] = {}  # Progress tracking
_job_logs: Dict[str, List[str]] = {}  # Store recent logs
_job_start_time: Dict[str, float] = {}  # Job start timestamps

def create_job(job_type: str, metadata: dict = None) -> str:
    """Create a new job with unique ID and initialize tracking."""
    job_id = uuid4().hex
    _job_queues[job_id] = asyncio.Queue()
    _job_status[job_id] = "pending"
    _job_metadata[job_id] = {
        "type": job_type, 
        "data": metadata,
        "created_at": datetime.now().isoformat()
    }
    _job_progress[job_id] = {
        "current_step": 0,
        "total_steps": 0,
        "percentage": 0,
        "current_phase": "Initializing",
        "estimated_completion": None
    }
    _job_logs[job_id] = []
    _job_start_time[job_id] = time.time()
    return job_id

def get_queue(job_id: str) -> asyncio.Queue[str]:
    """Get the log queue for a job."""
    return _job_queues[job_id]

def set_status(job_id: str, status: str):
    """Set the status of a job."""
    _job_status[job_id] = status
    if status == "running" and job_id not in _job_start_time:
        _job_start_time[job_id] = time.time()

def get_status(job_id: str) -> str:
    """Get the current status of a job."""
    return _job_status.get(job_id, "not_found")

def get_metadata(job_id: str) -> dict:
    """Get metadata for a job."""
    return _job_metadata.get(job_id, {})

def update_progress(job_id: str, current_step: int = None, total_steps: int = None, 
                   phase: str = None, percentage: float = None):
    """Update progress information for a job."""
    if job_id not in _job_progress:
        _job_progress[job_id] = {
            "current_step": 0,
            "total_steps": 0,
            "percentage": 0,
            "current_phase": "Initializing",
            "estimated_completion": None
        }
    
    progress = _job_progress[job_id]
    
    if current_step is not None:
        progress["current_step"] = current_step
    if total_steps is not None:
        progress["total_steps"] = total_steps
    if phase is not None:
        progress["current_phase"] = phase
    if percentage is not None:
        progress["percentage"] = min(100, max(0, percentage))
    elif current_step is not None and total_steps is not None and total_steps > 0:
        progress["percentage"] = min(100, (current_step / total_steps) * 100)
    
    # Estimate completion time
    if progress["percentage"] > 0 and _job_start_time.get(job_id):
        elapsed = time.time() - _job_start_time[job_id]
        estimated_total = elapsed / (progress["percentage"] / 100)
        remaining = estimated_total - elapsed
        progress["estimated_completion"] = datetime.fromtimestamp(
            time.time() + remaining
        ).isoformat()

def get_progress(job_id: str) -> dict:
    """Get progress information for a job."""
    return _job_progress.get(job_id, {})

def add_log(job_id: str, message: str):
    """Add a log message to a job."""
    if job_id not in _job_logs:
        _job_logs[job_id] = []
    
    timestamp = datetime.now().isoformat()
    log_entry = f"[{timestamp}] {message}"
    _job_logs[job_id].append(log_entry)
    
    # Keep only last 100 log entries
    if len(_job_logs[job_id]) > 100:
        _job_logs[job_id] = _job_logs[job_id][-100:]

def get_logs(job_id: str, limit: int = 50) -> List[str]:
    """Get recent log entries for a job."""
    logs = _job_logs.get(job_id, [])
    return logs[-limit:] if limit else logs

def get_job_summary(job_id: str) -> dict:
    """Get complete job summary including status, progress, and metadata."""
    if job_id not in _job_status:
        return {"error": "Job not found"}
    
    start_time = _job_start_time.get(job_id)
    elapsed_time = None
    if start_time:
        elapsed_time = time.time() - start_time
    
    return {
        "job_id": job_id,
        "status": _job_status[job_id],
        "metadata": _job_metadata.get(job_id, {}),
        "progress": _job_progress.get(job_id, {}),
        "elapsed_time": elapsed_time,
        "recent_logs": get_logs(job_id, 10)
    }

def get_all_jobs() -> List[dict]:
    """Get summary of all jobs."""
    return [get_job_summary(job_id) for job_id in _job_status.keys()]

def cleanup_job(job_id: str):
    """Clean up job data (call when job is completed/failed)."""
    if job_id in _job_queues:
        del _job_queues[job_id]
    if job_id in _job_status:
        del _job_status[job_id]
    if job_id in _job_metadata:
        del _job_metadata[job_id]
    if job_id in _job_progress:
        del _job_progress[job_id]
    if job_id in _job_logs:
        del _job_logs[job_id]
    if job_id in _job_start_time:
        del _job_start_time[job_id]
