#!/usr/bin/env python3
"""
CLI tool for managing ExoplanetAI automated discovery pipeline
"""

import asyncio
import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.data_ingest import get_ingest_service
from services.auto_discovery import AutoDiscoveryService
from services.model_registry import get_model_registry
from core.logging import get_logger, configure_structlog

# Configure logging
configure_structlog(
    service_name="auto-discovery-cli",
    environment="development",
    log_level="INFO",
    enable_json=False,
    enable_console=True
)

logger = get_logger(__name__)


class AutoDiscoveryCLI:
    """CLI interface for automated discovery management"""
    
    def __init__(self):
        self.ingest_service = get_ingest_service()
        self.discovery_service = None
        self.model_registry = get_model_registry()
    
    async def status(self) -> Dict[str, Any]:
        """Get pipeline status"""
        try:
            # Ingestion status
            ingestion_stats = self.ingest_service.get_ingestion_stats()
            
            # Discovery status
            discovery_stats = {}
            if self.discovery_service:
                discovery_stats = {
                    "is_running": self.discovery_service.is_running,
                    "last_check": self.discovery_service.last_check_time.isoformat() if self.discovery_service.last_check_time else None,
                    "stats": self.discovery_service.discovery_stats
                }
            
            # Model registry status
            registry_stats = self.model_registry.get_registry_stats()
            active_models = self.model_registry.get_active_models()
            
            status = {
                "timestamp": datetime.now().isoformat(),
                "ingestion": {
                    "is_running": ingestion_stats.get("is_running", False),
                    "total_ingested": ingestion_stats.get("total_ingested", 0),
                    "recent_ingested": ingestion_stats.get("recent_ingested", 0),
                    "last_ingestion": ingestion_stats.get("last_ingestion"),
                    "average_quality": ingestion_stats.get("average_quality", 0.0)
                },
                "discovery": discovery_stats,
                "models": {
                    "registry_stats": registry_stats,
                    "active_models": active_models
                }
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return {"error": str(e)}
    
    async def start_ingestion(self) -> Dict[str, str]:
        """Start data ingestion"""
        try:
            if self.ingest_service.is_running:
                return {"status": "already_running", "message": "Data ingestion is already running"}
            
            # Start ingestion in background
            task = asyncio.create_task(self.ingest_service.start_continuous_ingestion())
            
            # Wait a bit to see if it starts successfully
            await asyncio.sleep(2)
            
            if self.ingest_service.is_running:
                return {"status": "started", "message": "Data ingestion started successfully"}
            else:
                return {"status": "failed", "message": "Failed to start data ingestion"}
                
        except Exception as e:
            logger.error(f"Error starting ingestion: {e}")
            return {"status": "error", "message": str(e)}
    
    async def stop_ingestion(self) -> Dict[str, str]:
        """Stop data ingestion"""
        try:
            self.ingest_service.stop_ingestion()
            return {"status": "stopped", "message": "Data ingestion stopped"}
        except Exception as e:
            logger.error(f"Error stopping ingestion: {e}")
            return {"status": "error", "message": str(e)}
    
    async def start_discovery(self, confidence_threshold: float = 0.85) -> Dict[str, str]:
        """Start automated discovery"""
        try:
            if not self.discovery_service:
                self.discovery_service = AutoDiscoveryService(confidence_threshold=confidence_threshold)
            
            if self.discovery_service.is_running:
                return {"status": "already_running", "message": "Discovery service is already running"}
            
            # Start discovery in background
            task = asyncio.create_task(self.discovery_service.start())
            
            # Wait a bit to see if it starts successfully
            await asyncio.sleep(2)
            
            if self.discovery_service.is_running:
                return {"status": "started", "message": "Discovery service started successfully"}
            else:
                return {"status": "failed", "message": "Failed to start discovery service"}
                
        except Exception as e:
            logger.error(f"Error starting discovery: {e}")
            return {"status": "error", "message": str(e)}
    
    async def stop_discovery(self) -> Dict[str, str]:
        """Stop automated discovery"""
        try:
            if self.discovery_service:
                self.discovery_service.stop()
                return {"status": "stopped", "message": "Discovery service stopped"}
            else:
                return {"status": "not_running", "message": "Discovery service was not running"}
        except Exception as e:
            logger.error(f"Error stopping discovery: {e}")
            return {"status": "error", "message": str(e)}
    
    async def run_single_cycle(self, confidence_threshold: float = 0.85) -> Dict[str, Any]:
        """Run a single discovery cycle"""
        try:
            if not self.discovery_service:
                self.discovery_service = AutoDiscoveryService(confidence_threshold=confidence_threshold)
            
            logger.info("Running single discovery cycle...")
            
            # Run one cycle
            await self.discovery_service._discovery_cycle()
            
            # Get results
            stats = self.discovery_service.discovery_stats
            candidates = [c.to_dict() for c in self.discovery_service.candidates[-10:]]  # Last 10 candidates
            
            return {
                "status": "completed",
                "stats": stats,
                "recent_candidates": candidates,
                "message": f"Discovery cycle completed. Found {len(candidates)} recent candidates."
            }
            
        except Exception as e:
            logger.error(f"Error running discovery cycle: {e}")
            return {"status": "error", "message": str(e)}
    
    async def ingest_targets(self, targets: list) -> Dict[str, Any]:
        """Manually ingest specific targets"""
        try:
            results = []
            
            for target in targets:
                if isinstance(target, str):
                    # Assume it's a TIC ID
                    target_dict = {
                        "target_name": f"TIC-{target}",
                        "tic_id": target,
                        "mission": "TESS"
                    }
                else:
                    target_dict = target
                
                result = await self.ingest_service._ingest_single_observation(target_dict)
                if result:
                    results.append(result.to_dict())
                    logger.info(f"✅ Ingested {target_dict['target_name']}")
                else:
                    logger.warning(f"❌ Failed to ingest {target_dict['target_name']}")
            
            # Save ingestion history after all ingestions
            if results:
                self.ingest_service._save_ingestion_history()
            
            return {
                "status": "completed",
                "ingested_count": len(results),
                "results": results,
                "message": f"Ingested {len(results)} out of {len(targets)} targets"
            }
            
        except Exception as e:
            logger.error(f"Error ingesting targets: {e}")
            return {"status": "error", "message": str(e)}
    
    def list_models(self, model_name: str = None) -> Dict[str, Any]:
        """List available models"""
        try:
            models = self.model_registry.list_models(model_name)
            active_models = self.model_registry.get_active_models()
            
            return {
                "status": "success",
                "models": models,
                "active_models": active_models
            }
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return {"status": "error", "message": str(e)}
    
    def deploy_model(self, model_name: str, version: str) -> Dict[str, str]:
        """Deploy a specific model version"""
        try:
            success = self.model_registry.deploy_model(model_name, version)
            
            if success:
                return {
                    "status": "success",
                    "message": f"Model {model_name} v{version} deployed successfully"
                }
            else:
                return {
                    "status": "failed",
                    "message": f"Failed to deploy model {model_name} v{version}"
                }
        except Exception as e:
            logger.error(f"Error deploying model: {e}")
            return {"status": "error", "message": str(e)}
    
    def rollback_model(self, model_name: str) -> Dict[str, str]:
        """Rollback model to previous version"""
        try:
            success = self.model_registry.rollback_model(model_name)
            
            if success:
                return {
                    "status": "success",
                    "message": f"Model {model_name} rolled back successfully"
                }
            else:
                return {
                    "status": "failed",
                    "message": f"Failed to rollback model {model_name}"
                }
        except Exception as e:
            logger.error(f"Error rolling back model: {e}")
            return {"status": "error", "message": str(e)}


async def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="ExoplanetAI Automated Discovery CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Status command
    subparsers.add_parser("status", help="Get pipeline status")
    
    # Ingestion commands
    ingest_parser = subparsers.add_parser("ingestion", help="Data ingestion commands")
    ingest_subparsers = ingest_parser.add_subparsers(dest="ingest_action")
    ingest_subparsers.add_parser("start", help="Start data ingestion")
    ingest_subparsers.add_parser("stop", help="Stop data ingestion")
    ingest_subparsers.add_parser("status", help="Get ingestion status")
    
    ingest_targets_parser = ingest_subparsers.add_parser("targets", help="Ingest specific targets")
    ingest_targets_parser.add_argument("targets", nargs="+", help="TIC IDs or target names to ingest")
    
    # Discovery commands
    discovery_parser = subparsers.add_parser("discovery", help="Discovery commands")
    discovery_subparsers = discovery_parser.add_subparsers(dest="discovery_action")
    
    discovery_start_parser = discovery_subparsers.add_parser("start", help="Start discovery service")
    discovery_start_parser.add_argument("--confidence", type=float, default=0.85, help="Confidence threshold")
    
    discovery_subparsers.add_parser("stop", help="Stop discovery service")
    
    discovery_cycle_parser = discovery_subparsers.add_parser("cycle", help="Run single discovery cycle")
    discovery_cycle_parser.add_argument("--confidence", type=float, default=0.85, help="Confidence threshold")
    
    # Model commands
    model_parser = subparsers.add_parser("models", help="Model management commands")
    model_subparsers = model_parser.add_subparsers(dest="model_action")
    
    model_list_parser = model_subparsers.add_parser("list", help="List models")
    model_list_parser.add_argument("--name", help="Specific model name")
    
    model_deploy_parser = model_subparsers.add_parser("deploy", help="Deploy model version")
    model_deploy_parser.add_argument("name", help="Model name")
    model_deploy_parser.add_argument("version", help="Model version")
    
    model_rollback_parser = model_subparsers.add_parser("rollback", help="Rollback model")
    model_rollback_parser.add_argument("name", help="Model name")
    
    # Pipeline commands
    pipeline_parser = subparsers.add_parser("pipeline", help="Full pipeline commands")
    pipeline_subparsers = pipeline_parser.add_subparsers(dest="pipeline_action")
    
    pipeline_start_parser = pipeline_subparsers.add_parser("start", help="Start full pipeline")
    pipeline_start_parser.add_argument("--confidence", type=float, default=0.85, help="Confidence threshold")
    
    pipeline_subparsers.add_parser("stop", help="Stop full pipeline")
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize CLI
    cli = AutoDiscoveryCLI()
    result = {}
    
    try:
        # Execute commands
        if args.command == "status":
            result = await cli.status()
        
        elif args.command == "ingestion":
            if args.ingest_action == "start":
                result = await cli.start_ingestion()
            elif args.ingest_action == "stop":
                result = await cli.stop_ingestion()
            elif args.ingest_action == "status":
                status = await cli.status()
                result = status.get("ingestion", {})
            elif args.ingest_action == "targets":
                result = await cli.ingest_targets(args.targets)
        
        elif args.command == "discovery":
            if args.discovery_action == "start":
                result = await cli.start_discovery(args.confidence)
            elif args.discovery_action == "stop":
                result = await cli.stop_discovery()
            elif args.discovery_action == "cycle":
                result = await cli.run_single_cycle(args.confidence)
        
        elif args.command == "models":
            if args.model_action == "list":
                result = cli.list_models(args.name if hasattr(args, 'name') else None)
            elif args.model_action == "deploy":
                result = cli.deploy_model(args.name, args.version)
            elif args.model_action == "rollback":
                result = cli.rollback_model(args.name)
        
        elif args.command == "pipeline":
            if args.pipeline_action == "start":
                # Start both ingestion and discovery
                ingest_result = await cli.start_ingestion()
                discovery_result = await cli.start_discovery(args.confidence)
                result = {
                    "status": "started",
                    "ingestion": ingest_result,
                    "discovery": discovery_result
                }
            elif args.pipeline_action == "stop":
                # Stop both services
                ingest_result = await cli.stop_ingestion()
                discovery_result = await cli.stop_discovery()
                result = {
                    "status": "stopped",
                    "ingestion": ingest_result,
                    "discovery": discovery_result
                }
        
        # Output result
        print(json.dumps(result, indent=2, default=str))
        
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(json.dumps({"status": "error", "message": str(e)}, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
