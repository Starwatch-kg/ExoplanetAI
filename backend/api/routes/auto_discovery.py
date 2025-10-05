"""
API endpoints for automated exoplanet discovery
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field

from services.auto_discovery import AutoDiscoveryService, DiscoveryCandidate

router = APIRouter(prefix="/auto-discovery", tags=["Auto Discovery"])

# Global service instance (will be initialized on startup)
discovery_service: Optional[AutoDiscoveryService] = None


class DiscoveryConfig(BaseModel):
    """Configuration for automated discovery"""
    confidence_threshold: float = Field(0.85, ge=0.0, le=1.0, description="Minimum confidence for candidates")
    check_interval_hours: int = Field(6, ge=1, le=24, description="Hours between checks")
    max_concurrent_tasks: int = Field(5, ge=1, le=20, description="Max parallel processing tasks")


class CandidateResponse(BaseModel):
    """Response model for a discovery candidate"""
    target_name: str
    tic_id: Optional[str]
    mission: str
    confidence: float
    predicted_class: str
    period: Optional[float]
    depth: Optional[float]
    snr: Optional[float]
    discovery_time: str
    lightcurve_path: Optional[str]


class DiscoveryStats(BaseModel):
    """Discovery statistics"""
    total_processed: int
    total_candidates: int
    high_confidence_candidates: int
    last_check_time: Optional[str]
    is_running: bool
    confidence_threshold: float


class ManualDiscoveryRequest(BaseModel):
    """Request to manually trigger discovery on specific targets"""
    targets: List[str] = Field(..., description="List of TIC IDs or target names")
    mission: str = Field("TESS", description="Mission name")


@router.post("/start", status_code=200)
async def start_discovery(
    background_tasks: BackgroundTasks,
    config: Optional[DiscoveryConfig] = None
):
    """
    Start the automated discovery service
    
    The service will run in the background and continuously monitor
    for new exoplanet candidates.
    """
    global discovery_service
    
    if discovery_service and discovery_service.is_running:
        return {
            "status": "already_running",
            "message": "Discovery service is already running"
        }
    
    # Initialize service with config
    if config:
        discovery_service = AutoDiscoveryService(
            confidence_threshold=config.confidence_threshold,
            check_interval_hours=config.check_interval_hours,
            max_concurrent_tasks=config.max_concurrent_tasks
        )
    else:
        discovery_service = AutoDiscoveryService()
    
    # Start in background
    background_tasks.add_task(discovery_service.start)
    
    return {
        "status": "started",
        "message": "Automated discovery service started successfully",
        "config": {
            "confidence_threshold": discovery_service.confidence_threshold,
            "check_interval_hours": discovery_service.check_interval.total_seconds() / 3600,
            "max_concurrent_tasks": discovery_service.max_concurrent
        }
    }


@router.post("/stop", status_code=200)
async def stop_discovery():
    """Stop the automated discovery service"""
    global discovery_service
    
    if not discovery_service or not discovery_service.is_running:
        return {
            "status": "not_running",
            "message": "Discovery service is not running"
        }
    
    discovery_service.stop()
    
    return {
        "status": "stopped",
        "message": "Discovery service stopped successfully"
    }


@router.get("/status", response_model=DiscoveryStats)
async def get_status():
    """Get current status and statistics of the discovery service"""
    global discovery_service
    
    if not discovery_service:
        raise HTTPException(
            status_code=404,
            detail="Discovery service not initialized. Call /start first."
        )
    
    stats = await discovery_service.get_statistics()
    return stats


@router.get("/candidates", response_model=List[CandidateResponse])
async def get_candidates(
    hours: int = Query(24, ge=1, le=168, description="Get candidates from last N hours"),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0, description="Minimum confidence threshold")
):
    """
    Get recently discovered candidates
    
    Returns candidates discovered in the specified time window
    with confidence above the threshold.
    """
    global discovery_service
    
    if not discovery_service:
        raise HTTPException(
            status_code=404,
            detail="Discovery service not initialized"
        )
    
    candidates = await discovery_service.get_recent_candidates(
        hours=hours,
        min_confidence=min_confidence
    )
    
    return [
        CandidateResponse(
            target_name=c.target_name,
            tic_id=c.tic_id,
            mission=c.mission,
            confidence=c.confidence,
            predicted_class=c.predicted_class,
            period=c.period,
            depth=c.depth,
            snr=c.snr,
            discovery_time=c.discovery_time.isoformat(),
            lightcurve_path=c.lightcurve_path
        )
        for c in candidates
    ]


@router.get("/candidates/top", response_model=List[CandidateResponse])
async def get_top_candidates(
    limit: int = Query(10, ge=1, le=100, description="Number of top candidates to return")
):
    """Get top N candidates by confidence score"""
    global discovery_service
    
    if not discovery_service:
        raise HTTPException(
            status_code=404,
            detail="Discovery service not initialized"
        )
    
    # Get all candidates and sort by confidence
    all_candidates = await discovery_service.get_recent_candidates(
        hours=168,  # Last week
        min_confidence=0.0
    )
    
    top_candidates = sorted(
        all_candidates,
        key=lambda x: x.confidence,
        reverse=True
    )[:limit]
    
    return [
        CandidateResponse(
            target_name=c.target_name,
            tic_id=c.tic_id,
            mission=c.mission,
            confidence=c.confidence,
            predicted_class=c.predicted_class,
            period=c.period,
            depth=c.depth,
            snr=c.snr,
            discovery_time=c.discovery_time.isoformat(),
            lightcurve_path=c.lightcurve_path
        )
        for c in top_candidates
    ]


@router.post("/trigger", status_code=202)
async def trigger_manual_discovery(
    background_tasks: BackgroundTasks,
    request: ManualDiscoveryRequest
):
    """
    Manually trigger discovery on specific targets
    
    Useful for testing or analyzing specific objects without
    waiting for the automated cycle.
    """
    global discovery_service
    
    if not discovery_service:
        discovery_service = AutoDiscoveryService()
    
    # Process targets in background
    async def process_manual_targets():
        targets = [
            {"tic_id": tid, "name": tid, "mission": request.mission}
            for tid in request.targets
        ]
        candidates = await discovery_service._process_targets_batch(targets)
        await discovery_service._save_candidates(candidates)
    
    background_tasks.add_task(process_manual_targets)
    
    return {
        "status": "processing",
        "message": f"Processing {len(request.targets)} targets in background",
        "targets": request.targets
    }


@router.get("/reports/latest")
async def get_latest_report():
    """Get the latest discovery report"""
    global discovery_service
    
    if not discovery_service:
        raise HTTPException(
            status_code=404,
            detail="Discovery service not initialized"
        )
    
    # Find latest report file
    reports_dir = discovery_service.data_dir / "reports"
    if not reports_dir.exists():
        return {"message": "No reports available yet"}
    
    report_files = sorted(reports_dir.glob("report_*.json"), reverse=True)
    if not report_files:
        return {"message": "No reports available yet"}
    
    # Read latest report
    import json
    with open(report_files[0], 'r') as f:
        report = json.load(f)
    
    return report


@router.get("/config")
async def get_config():
    """Get current discovery service configuration"""
    global discovery_service
    
    if not discovery_service:
        raise HTTPException(
            status_code=404,
            detail="Discovery service not initialized"
        )
    
    return {
        "confidence_threshold": discovery_service.confidence_threshold,
        "check_interval_hours": discovery_service.check_interval.total_seconds() / 3600,
        "max_concurrent_tasks": discovery_service.max_concurrent,
        "data_directory": str(discovery_service.data_dir)
    }


@router.put("/config")
async def update_config(config: DiscoveryConfig):
    """Update discovery service configuration (requires restart)"""
    global discovery_service
    
    if not discovery_service:
        raise HTTPException(
            status_code=404,
            detail="Discovery service not initialized"
        )
    
    if discovery_service.is_running:
        raise HTTPException(
            status_code=400,
            detail="Cannot update config while service is running. Stop the service first."
        )
    
    # Update configuration
    discovery_service.confidence_threshold = config.confidence_threshold
    discovery_service.check_interval = timedelta(hours=config.check_interval_hours)
    discovery_service.max_concurrent = config.max_concurrent_tasks
    
    return {
        "status": "updated",
        "message": "Configuration updated successfully",
        "config": {
            "confidence_threshold": config.confidence_threshold,
            "check_interval_hours": config.check_interval_hours,
            "max_concurrent_tasks": config.max_concurrent_tasks
        }
    }


@router.delete("/candidates/{target_name}")
async def delete_candidate(target_name: str):
    """Delete a specific candidate from the list"""
    global discovery_service
    
    if not discovery_service:
        raise HTTPException(
            status_code=404,
            detail="Discovery service not initialized"
        )
    
    # Find and remove candidate
    initial_count = len(discovery_service.candidates)
    discovery_service.candidates = [
        c for c in discovery_service.candidates
        if c.target_name != target_name
    ]
    
    if len(discovery_service.candidates) == initial_count:
        raise HTTPException(
            status_code=404,
            detail=f"Candidate {target_name} not found"
        )
    
    return {
        "status": "deleted",
        "message": f"Candidate {target_name} deleted successfully"
    }


@router.post("/export")
async def export_candidates(
    format: str = Query("json", regex="^(json|csv)$"),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0)
):
    """Export candidates to JSON or CSV format"""
    global discovery_service
    
    if not discovery_service:
        raise HTTPException(
            status_code=404,
            detail="Discovery service not initialized"
        )
    
    candidates = [
        c for c in discovery_service.candidates
        if c.confidence >= min_confidence
    ]
    
    if format == "json":
        return {
            "format": "json",
            "count": len(candidates),
            "candidates": [c.to_dict() for c in candidates]
        }
    else:  # CSV
        import csv
        from io import StringIO
        
        output = StringIO()
        if candidates:
            fieldnames = ["target_name", "tic_id", "mission", "confidence", 
                         "predicted_class", "period", "depth", "snr", "discovery_time"]
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            
            for c in candidates:
                writer.writerow({
                    "target_name": c.target_name,
                    "tic_id": c.tic_id or "",
                    "mission": c.mission,
                    "confidence": c.confidence,
                    "predicted_class": c.predicted_class,
                    "period": c.period or "",
                    "depth": c.depth or "",
                    "snr": c.snr or "",
                    "discovery_time": c.discovery_time.isoformat()
                })
        
        return {
            "format": "csv",
            "count": len(candidates),
            "data": output.getvalue()
        }
