from fastapi import APIRouter, HTTPException
from typing import List, Optional
from datetime import datetime

router = APIRouter()

@router.get("/pmu/{bus_id}")
async def get_pmu_data(bus_id: int, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None):
    return {"bus_id": bus_id, "voltage": 1.0, "frequency": 60.0, "timestamp": datetime.now().isoformat()}

@router.get("/pmu/batch")
async def get_batch_pmu_data(bus_ids: List[int]):
    return [{"bus_id": bid, "voltage": 1.0, "frequency": 60.0} for bid in bus_ids]

@router.post("/ingest")
async def ingest_pmu_data(data: dict):
    return {"status": "ingested", "timestamp": datetime.now().isoformat()}
