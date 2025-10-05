"""
Data versioning system for ExoplanetAI using DVC-like approach
Система версионирования данных для ExoplanetAI
"""

import asyncio
import hashlib
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import aiofiles
import git
import pandas as pd

from core.config import get_settings

logger = logging.getLogger(__name__)


class VersionManager:
    """
    Data versioning system for reproducible data management
    Supports Git-based metadata versioning and file-based data versioning
    """

    def __init__(self):
        self.settings = get_settings()
        self.base_path = Path(self.settings.data.data_path)
        self.versions_path = self.base_path / "versions"
        self.metadata_path = self.base_path / "metadata" / "versions"
        self.git_repo: Optional[git.Repo] = None
        
        # Version tracking
        self.current_version = "v1.0.0"
        self.version_file = self.metadata_path / "current_version.json"

    async def initialize(self) -> bool:
        """Initialize versioning system"""
        try:
            # Create directories
            self.versions_path.mkdir(parents=True, exist_ok=True)
            self.metadata_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize Git repository for metadata
            try:
                self.git_repo = git.Repo(self.base_path)
                logger.info("Using existing Git repository")
            except git.InvalidGitRepositoryError:
                # Initialize new Git repo
                self.git_repo = git.Repo.init(self.base_path)
                
                # Create .gitignore for large data files
                gitignore_content = """
# Large data files - use DVC or external storage
*.fits
*.csv
lightcurves/
raw/
processed/

# Cache files
cache/
__pycache__/
*.pyc

# Logs
*.log
logs/

# Temporary files
*.tmp
*.bak
"""
                gitignore_path = self.base_path / ".gitignore"
                with open(gitignore_path, 'w') as f:
                    f.write(gitignore_content)
                
                logger.info("Initialized new Git repository for metadata versioning")
            
            # Load or create current version
            await self._load_current_version()
            
            logger.info(f"Version manager initialized, current version: {self.current_version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize version manager: {e}")
            return False

    async def _load_current_version(self):
        """Load current version from file"""
        if self.version_file.exists():
            try:
                async with aiofiles.open(self.version_file, 'r') as f:
                    content = await f.read()
                    version_data = json.loads(content)
                    self.current_version = version_data.get("version", "v1.0.0")
            except Exception as e:
                logger.warning(f"Could not load version file: {e}")
                self.current_version = "v1.0.0"
        else:
            await self._save_current_version()

    async def _save_current_version(self):
        """Save current version to file"""
        version_data = {
            "version": self.current_version,
            "updated_at": datetime.utcnow().isoformat(),
            "description": f"Data version {self.current_version}"
        }
        
        async with aiofiles.open(self.version_file, 'w') as f:
            await f.write(json.dumps(version_data, indent=2))

    async def create_version(
        self,
        version_name: str,
        description: str,
        files_to_version: List[Path],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a new data version
        
        Args:
            version_name: Version identifier (e.g., "v1.1.0")
            description: Version description
            files_to_version: List of file paths to include in version
            metadata: Additional metadata
            
        Returns:
            Dict with version creation results
        """
        logger.info(f"Creating version {version_name}")
        
        try:
            # Create version directory
            version_path = self.versions_path / version_name
            version_path.mkdir(exist_ok=True)
            
            # Create version manifest
            manifest = {
                "version": version_name,
                "description": description,
                "created_at": datetime.utcnow().isoformat(),
                "files": [],
                "metadata": metadata or {},
                "parent_version": self.current_version
            }
            
            # Process files
            total_size = 0
            for file_path in files_to_version:
                if file_path.exists():
                    # Calculate file hash
                    file_hash = await self._calculate_file_hash(file_path)
                    file_size = file_path.stat().st_size
                    total_size += file_size
                    
                    # Create file entry
                    file_entry = {
                        "path": str(file_path.relative_to(self.base_path)),
                        "hash": file_hash,
                        "size_bytes": file_size,
                        "modified_at": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                    }
                    manifest["files"].append(file_entry)
                    
                    # Create hard link or copy file (depending on filesystem)
                    version_file_path = version_path / file_path.name
                    try:
                        # Try hard link first (saves space)
                        version_file_path.hardlink_to(file_path)
                    except OSError:
                        # Fallback to copy
                        shutil.copy2(file_path, version_file_path)
                    
                    logger.debug(f"Added file to version: {file_path}")
                else:
                    logger.warning(f"File not found: {file_path}")
            
            manifest["total_size_bytes"] = total_size
            manifest["file_count"] = len(manifest["files"])
            
            # Save manifest
            manifest_path = version_path / "manifest.json"
            async with aiofiles.open(manifest_path, 'w') as f:
                await f.write(json.dumps(manifest, indent=2))
            
            # Save metadata version info
            version_metadata_path = self.metadata_path / f"{version_name}.json"
            async with aiofiles.open(version_metadata_path, 'w') as f:
                await f.write(json.dumps(manifest, indent=2))
            
            # Commit to Git
            if self.git_repo:
                try:
                    self.git_repo.index.add([str(version_metadata_path.relative_to(self.base_path))])
                    self.git_repo.index.commit(f"Add version {version_name}: {description}")
                    logger.info(f"Committed version {version_name} to Git")
                except Exception as e:
                    logger.warning(f"Git commit failed: {e}")
            
            # Update current version
            self.current_version = version_name
            await self._save_current_version()
            
            result = {
                "status": "success",
                "version": version_name,
                "description": description,
                "file_count": len(manifest["files"]),
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "version_path": str(version_path),
                "manifest": manifest
            }
            
            logger.info(f"Created version {version_name} with {len(manifest['files'])} files ({result['total_size_mb']} MB)")
            return result
            
        except Exception as e:
            logger.error(f"Failed to create version {version_name}: {e}")
            return {
                "status": "error",
                "message": str(e),
                "version": version_name
            }

    async def restore_version(self, version_name: str, target_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Restore files from a specific version
        
        Args:
            version_name: Version to restore
            target_path: Target directory (defaults to original locations)
            
        Returns:
            Dict with restoration results
        """
        logger.info(f"Restoring version {version_name}")
        
        try:
            version_path = self.versions_path / version_name
            manifest_path = version_path / "manifest.json"
            
            if not manifest_path.exists():
                return {
                    "status": "error",
                    "message": f"Version {version_name} not found"
                }
            
            # Load manifest
            async with aiofiles.open(manifest_path, 'r') as f:
                content = await f.read()
                manifest = json.loads(content)
            
            restored_files = []
            failed_files = []
            
            # Restore files
            for file_entry in manifest["files"]:
                source_path = version_path / Path(file_entry["path"]).name
                
                if target_path:
                    dest_path = target_path / Path(file_entry["path"]).name
                else:
                    dest_path = self.base_path / file_entry["path"]
                
                try:
                    # Create destination directory
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Copy file
                    shutil.copy2(source_path, dest_path)
                    
                    # Verify hash
                    restored_hash = await self._calculate_file_hash(dest_path)
                    if restored_hash == file_entry["hash"]:
                        restored_files.append(str(dest_path))
                    else:
                        failed_files.append(f"{dest_path}: hash mismatch")
                        
                except Exception as e:
                    failed_files.append(f"{dest_path}: {str(e)}")
            
            result = {
                "status": "success" if not failed_files else "partial",
                "version": version_name,
                "restored_files": len(restored_files),
                "failed_files": len(failed_files),
                "failures": failed_files[:10],  # Limit to first 10 failures
                "manifest": manifest
            }
            
            logger.info(f"Restored version {version_name}: {len(restored_files)} files successful, {len(failed_files)} failed")
            return result
            
        except Exception as e:
            logger.error(f"Failed to restore version {version_name}: {e}")
            return {
                "status": "error",
                "message": str(e),
                "version": version_name
            }

    async def list_versions(self) -> List[Dict[str, Any]]:
        """
        List all available versions
        
        Returns:
            List of version information
        """
        versions = []
        
        try:
            if self.metadata_path.exists():
                for version_file in self.metadata_path.glob("v*.json"):
                    if version_file.name != "current_version.json":
                        try:
                            async with aiofiles.open(version_file, 'r') as f:
                                content = await f.read()
                                version_data = json.loads(content)
                                
                                # Add summary info
                                summary = {
                                    "version": version_data["version"],
                                    "description": version_data["description"],
                                    "created_at": version_data["created_at"],
                                    "file_count": version_data.get("file_count", 0),
                                    "total_size_mb": round(version_data.get("total_size_bytes", 0) / (1024 * 1024), 2),
                                    "parent_version": version_data.get("parent_version"),
                                    "is_current": version_data["version"] == self.current_version
                                }
                                versions.append(summary)
                                
                        except Exception as e:
                            logger.warning(f"Could not read version file {version_file}: {e}")
            
            # Sort by creation date
            versions.sort(key=lambda x: x["created_at"], reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to list versions: {e}")
        
        return versions

    async def get_version_info(self, version_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific version
        
        Args:
            version_name: Version to get info for
            
        Returns:
            Version information or None if not found
        """
        try:
            version_metadata_path = self.metadata_path / f"{version_name}.json"
            
            if version_metadata_path.exists():
                async with aiofiles.open(version_metadata_path, 'r') as f:
                    content = await f.read()
                    return json.loads(content)
            
        except Exception as e:
            logger.error(f"Failed to get version info for {version_name}: {e}")
        
        return None

    async def compare_versions(self, version1: str, version2: str) -> Dict[str, Any]:
        """
        Compare two versions
        
        Args:
            version1: First version to compare
            version2: Second version to compare
            
        Returns:
            Comparison results
        """
        try:
            info1 = await self.get_version_info(version1)
            info2 = await self.get_version_info(version2)
            
            if not info1 or not info2:
                return {
                    "status": "error",
                    "message": "One or both versions not found"
                }
            
            # Compare files
            files1 = {f["path"]: f for f in info1["files"]}
            files2 = {f["path"]: f for f in info2["files"]}
            
            added_files = set(files2.keys()) - set(files1.keys())
            removed_files = set(files1.keys()) - set(files2.keys())
            common_files = set(files1.keys()) & set(files2.keys())
            
            modified_files = []
            for file_path in common_files:
                if files1[file_path]["hash"] != files2[file_path]["hash"]:
                    modified_files.append({
                        "path": file_path,
                        "size_change": files2[file_path]["size_bytes"] - files1[file_path]["size_bytes"],
                        "modified_at_v1": files1[file_path]["modified_at"],
                        "modified_at_v2": files2[file_path]["modified_at"]
                    })
            
            comparison = {
                "version1": version1,
                "version2": version2,
                "summary": {
                    "added_files": len(added_files),
                    "removed_files": len(removed_files),
                    "modified_files": len(modified_files),
                    "unchanged_files": len(common_files) - len(modified_files)
                },
                "details": {
                    "added_files": list(added_files),
                    "removed_files": list(removed_files),
                    "modified_files": modified_files
                },
                "size_change_mb": round(
                    (info2.get("total_size_bytes", 0) - info1.get("total_size_bytes", 0)) / (1024 * 1024), 2
                )
            }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Failed to compare versions {version1} and {version2}: {e}")
            return {
                "status": "error",
                "message": str(e)
            }

    async def delete_version(self, version_name: str, keep_metadata: bool = True) -> Dict[str, Any]:
        """
        Delete a version (files and optionally metadata)
        
        Args:
            version_name: Version to delete
            keep_metadata: Whether to keep metadata in Git
            
        Returns:
            Deletion results
        """
        if version_name == self.current_version:
            return {
                "status": "error",
                "message": "Cannot delete current version"
            }
        
        try:
            version_path = self.versions_path / version_name
            version_metadata_path = self.metadata_path / f"{version_name}.json"
            
            deleted_files = 0
            freed_space = 0
            
            # Delete version files
            if version_path.exists():
                for file_path in version_path.rglob('*'):
                    if file_path.is_file():
                        freed_space += file_path.stat().st_size
                        deleted_files += 1
                
                shutil.rmtree(version_path)
                logger.info(f"Deleted version directory: {version_path}")
            
            # Optionally delete metadata
            if not keep_metadata and version_metadata_path.exists():
                version_metadata_path.unlink()
                
                # Remove from Git
                if self.git_repo:
                    try:
                        self.git_repo.index.remove([str(version_metadata_path.relative_to(self.base_path))])
                        self.git_repo.index.commit(f"Remove version {version_name}")
                    except Exception as e:
                        logger.warning(f"Git removal failed: {e}")
            
            result = {
                "status": "success",
                "version": version_name,
                "deleted_files": deleted_files,
                "freed_space_mb": round(freed_space / (1024 * 1024), 2),
                "metadata_kept": keep_metadata
            }
            
            logger.info(f"Deleted version {version_name}: {deleted_files} files, {result['freed_space_mb']} MB freed")
            return result
            
        except Exception as e:
            logger.error(f"Failed to delete version {version_name}: {e}")
            return {
                "status": "error",
                "message": str(e),
                "version": version_name
            }

    async def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file"""
        hash_sha256 = hashlib.sha256()
        
        async with aiofiles.open(file_path, 'rb') as f:
            while chunk := await f.read(8192):
                hash_sha256.update(chunk)
        
        return hash_sha256.hexdigest()

    async def create_data_snapshot(
        self,
        snapshot_name: str,
        description: str,
        include_patterns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Create a snapshot of current data state
        
        Args:
            snapshot_name: Name for the snapshot
            description: Description of the snapshot
            include_patterns: File patterns to include (e.g., ["*.csv", "*.fits"])
            
        Returns:
            Snapshot creation results
        """
        logger.info(f"Creating data snapshot: {snapshot_name}")
        
        try:
            # Find files to include
            files_to_version = []
            
            if include_patterns:
                for pattern in include_patterns:
                    files_to_version.extend(self.base_path.rglob(pattern))
            else:
                # Default patterns
                default_patterns = ["*.csv", "*.fits", "*.json"]
                for pattern in default_patterns:
                    files_to_version.extend(self.base_path.rglob(pattern))
            
            # Filter out cache and temporary files
            exclude_dirs = {"cache", "__pycache__", ".git", "logs"}
            files_to_version = [
                f for f in files_to_version 
                if f.is_file() and not any(exclude_dir in f.parts for exclude_dir in exclude_dirs)
            ]
            
            # Create version
            version_name = f"snapshot_{snapshot_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            result = await self.create_version(
                version_name=version_name,
                description=f"Data snapshot: {description}",
                files_to_version=files_to_version,
                metadata={
                    "snapshot_type": "data_snapshot",
                    "snapshot_name": snapshot_name,
                    "include_patterns": include_patterns or ["*.csv", "*.fits", "*.json"]
                }
            )
            
            if result["status"] == "success":
                logger.info(f"Created data snapshot {snapshot_name} as version {version_name}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to create data snapshot {snapshot_name}: {e}")
            return {
                "status": "error",
                "message": str(e),
                "snapshot_name": snapshot_name
            }

    async def get_version_stats(self) -> Dict[str, Any]:
        """Get statistics about all versions"""
        try:
            versions = await self.list_versions()
            
            if not versions:
                return {
                    "total_versions": 0,
                    "current_version": self.current_version,
                    "total_size_mb": 0
                }
            
            total_size = sum(v["total_size_mb"] for v in versions)
            latest_version = max(versions, key=lambda x: x["created_at"])
            
            stats = {
                "total_versions": len(versions),
                "current_version": self.current_version,
                "latest_version": latest_version["version"],
                "total_size_mb": round(total_size, 2),
                "average_size_mb": round(total_size / len(versions), 2),
                "versions_by_month": {},
                "storage_path": str(self.versions_path)
            }
            
            # Group by month
            for version in versions:
                month_key = version["created_at"][:7]  # YYYY-MM
                if month_key not in stats["versions_by_month"]:
                    stats["versions_by_month"][month_key] = 0
                stats["versions_by_month"][month_key] += 1
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get version stats: {e}")
            return {
                "error": str(e)
            }
