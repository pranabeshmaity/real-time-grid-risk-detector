from fastapi import APIRouter
import psutil
import platform
from datetime import datetime

router = APIRouter()

@router.get("/info")
async def system_info():
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": psutil.cpu_count(),
        "memory_total": psutil.virtual_memory().total,
        "timestamp": datetime.now().isoformat()
    }

@router.get("/status")
async def system_status():
    return {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_usage": psutil.disk_usage('/').percent,
        "timestamp": datetime.now().isoformat()
    }
