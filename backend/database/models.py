"""
Database models for AstroManas
SQLite database models for storing exoplanet data, search results, and system metrics
"""

import asyncio
import json
import logging
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class ExoplanetRecord:
    """Exoplanet database record"""

    id: Optional[int] = None
    name: str = ""
    host_star: str = ""
    discovery_method: str = ""
    discovery_year: Optional[int] = None
    orbital_period_days: Optional[float] = None
    radius_earth_radii: Optional[float] = None
    mass_earth_masses: Optional[float] = None
    equilibrium_temperature_k: Optional[float] = None
    distance_parsecs: Optional[float] = None
    confidence: float = 0.0
    status: str = "candidate"
    habitable_zone: bool = False
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SearchResult:
    """Search result database record"""

    id: Optional[int] = None
    target_name: str = ""
    catalog: str = ""
    mission: str = ""
    method: str = ""  # bls, gpi, etc.
    exoplanet_detected: bool = False
    detection_confidence: float = 0.0
    processing_time_ms: float = 0.0
    result_data: str = "{}"  # JSON string
    created_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.result_data:
            try:
                data["result_data"] = json.loads(self.result_data)
            except json.JSONDecodeError:
                data["result_data"] = {}
        return data


@dataclass
class SystemMetrics:
    """System performance metrics"""

    id: Optional[int] = None
    timestamp: str = ""
    service_name: str = ""
    metric_name: str = ""
    metric_value: float = 0.0
    metadata: str = "{}"  # JSON string

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.metadata:
            try:
                data["metadata"] = json.loads(self.metadata)
            except json.JSONDecodeError:
                data["metadata"] = {}
        return data


class DatabaseManager:
    """SQLite database manager for AstroManas"""

    def __init__(self, db_path: str = "astromanas.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = None
        self._lock = asyncio.Lock()

    async def initialize(self):
        """Initialize database and create tables"""
        try:
            await self._create_tables()
            await self._create_indexes()
            await self._populate_initial_data()
            logger.info(f"✅ Database initialized: {self.db_path}")
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            raise

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection"""
        if self._connection is None:
            self._connection = sqlite3.connect(
                str(self.db_path), check_same_thread=False, timeout=30.0
            )
            self._connection.row_factory = sqlite3.Row
            # Enable WAL mode for better concurrency
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=NORMAL")
            self._connection.execute("PRAGMA cache_size=10000")
            self._connection.execute("PRAGMA temp_store=MEMORY")
        return self._connection

    async def _create_tables(self):
        """Create database tables"""
        conn = self._get_connection()

        # Exoplanets table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS exoplanets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                host_star TEXT NOT NULL,
                discovery_method TEXT NOT NULL,
                discovery_year INTEGER,
                orbital_period_days REAL,
                radius_earth_radii REAL,
                mass_earth_masses REAL,
                equilibrium_temperature_k REAL,
                distance_parsecs REAL,
                confidence REAL DEFAULT 0.0,
                status TEXT DEFAULT 'candidate',
                habitable_zone BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Search results table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS search_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                target_name TEXT NOT NULL,
                catalog TEXT NOT NULL,
                mission TEXT NOT NULL,
                method TEXT NOT NULL,
                exoplanet_detected BOOLEAN DEFAULT FALSE,
                detection_confidence REAL DEFAULT 0.0,
                processing_time_ms REAL DEFAULT 0.0,
                result_data TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # System metrics table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                service_name TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                metadata TEXT DEFAULT '{}'
            )
        """
        )

        # User sessions table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE NOT NULL,
                user_agent TEXT,
                ip_address TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        conn.commit()

    async def _create_indexes(self):
        """Create database indexes for performance"""
        conn = self._get_connection()

        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_exoplanets_name ON exoplanets(name)",
            "CREATE INDEX IF NOT EXISTS idx_exoplanets_host_star ON exoplanets(host_star)",
            "CREATE INDEX IF NOT EXISTS idx_exoplanets_method ON exoplanets(discovery_method)",
            "CREATE INDEX IF NOT EXISTS idx_exoplanets_confidence ON exoplanets(confidence DESC)",
            "CREATE INDEX IF NOT EXISTS idx_search_results_target ON search_results(target_name)",
            "CREATE INDEX IF NOT EXISTS idx_search_results_method ON search_results(method)",
            "CREATE INDEX IF NOT EXISTS idx_search_results_created ON search_results(created_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_metrics_service ON system_metrics(service_name)",
            "CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON system_metrics(timestamp DESC)",
        ]

        for index_sql in indexes:
            conn.execute(index_sql)

        conn.commit()

    async def _populate_initial_data(self):
        """Populate database with initial exoplanet data"""
        conn = self._get_connection()

        # Check if data already exists
        cursor = conn.execute("SELECT COUNT(*) FROM exoplanets")
        count = cursor.fetchone()[0]

        if count > 0:
            logger.info(f"Database already contains {count} exoplanets")
            return

        # Initial exoplanet data
        initial_exoplanets = [
            ExoplanetRecord(
                name="Kepler-452b",
                host_star="Kepler-452",
                discovery_method="transit",
                discovery_year=2015,
                orbital_period_days=384.8,
                radius_earth_radii=1.6,
                mass_earth_masses=5.0,
                equilibrium_temperature_k=265,
                distance_parsecs=430,
                confidence=0.95,
                status="confirmed",
                habitable_zone=True,
            ),
            ExoplanetRecord(
                name="Proxima Centauri b",
                host_star="Proxima Centauri",
                discovery_method="radial_velocity",
                discovery_year=2016,
                orbital_period_days=11.2,
                radius_earth_radii=1.1,
                mass_earth_masses=1.3,
                equilibrium_temperature_k=234,
                distance_parsecs=1.3,
                confidence=0.99,
                status="confirmed",
                habitable_zone=True,
            ),
            ExoplanetRecord(
                name="TRAPPIST-1e",
                host_star="TRAPPIST-1",
                discovery_method="transit",
                discovery_year=2017,
                orbital_period_days=6.1,
                radius_earth_radii=0.92,
                mass_earth_masses=0.77,
                equilibrium_temperature_k=251,
                distance_parsecs=12.1,
                confidence=0.98,
                status="confirmed",
                habitable_zone=True,
            ),
            ExoplanetRecord(
                name="TOI-715 b",
                host_star="TOI-715",
                discovery_method="transit",
                discovery_year=2024,
                orbital_period_days=19.3,
                radius_earth_radii=1.55,
                mass_earth_masses=3.02,
                equilibrium_temperature_k=280,
                distance_parsecs=42.3,
                confidence=0.92,
                status="confirmed",
                habitable_zone=True,
            ),
            ExoplanetRecord(
                name="K2-18 b",
                host_star="K2-18",
                discovery_method="transit",
                discovery_year=2015,
                orbital_period_days=32.9,
                radius_earth_radii=2.3,
                mass_earth_masses=8.6,
                equilibrium_temperature_k=265,
                distance_parsecs=34.0,
                confidence=0.94,
                status="confirmed",
                habitable_zone=True,
            ),
        ]

        for exoplanet in initial_exoplanets:
            await self.insert_exoplanet(exoplanet)

        logger.info(
            f"✅ Populated database with {len(initial_exoplanets)} initial exoplanets"
        )

    async def insert_exoplanet(self, exoplanet: ExoplanetRecord) -> int:
        """Insert new exoplanet record"""
        async with self._lock:
            conn = self._get_connection()
            cursor = conn.execute(
                """
                INSERT INTO exoplanets (
                    name, host_star, discovery_method, discovery_year,
                    orbital_period_days, radius_earth_radii, mass_earth_masses,
                    equilibrium_temperature_k, distance_parsecs, confidence,
                    status, habitable_zone
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    exoplanet.name,
                    exoplanet.host_star,
                    exoplanet.discovery_method,
                    exoplanet.discovery_year,
                    exoplanet.orbital_period_days,
                    exoplanet.radius_earth_radii,
                    exoplanet.mass_earth_masses,
                    exoplanet.equilibrium_temperature_k,
                    exoplanet.distance_parsecs,
                    exoplanet.confidence,
                    exoplanet.status,
                    exoplanet.habitable_zone,
                ),
            )
            conn.commit()
            return cursor.lastrowid

    async def get_exoplanets(
        self,
        limit: int = 100,
        offset: int = 0,
        method: Optional[str] = None,
        min_confidence: Optional[float] = None,
        habitable_only: bool = False,
    ) -> List[ExoplanetRecord]:
        """Get exoplanets with filtering"""
        async with self._lock:
            conn = self._get_connection()
            query = "SELECT * FROM exoplanets WHERE 1=1"
            params = []

            if method:
                # Validate method parameter to prevent SQL injection
                allowed_methods = [
                    "transit",
                    "radial_velocity",
                    "imaging",
                    "microlensing",
                    "astrometry",
                ]
                if method in allowed_methods:
                    query += " AND discovery_method = ?"
                    params.append(method)
                else:
                    raise ValueError(f"Invalid discovery method: {method}")

            if min_confidence is not None:
                query += " AND confidence >= ?"
                params.append(min_confidence)

            if habitable_only:
                query += " AND habitable_zone = 1"

            # Validate limit and offset to prevent resource exhaustion
            if limit > 10000:  # Maximum 10k results
                limit = 10000
            if offset < 0:
                offset = 0

            query += " ORDER BY confidence DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

            return [
                ExoplanetRecord(
                    id=row["id"],
                    name=row["name"],
                    host_star=row["host_star"],
                    discovery_method=row["discovery_method"],
                    discovery_year=row["discovery_year"],
                    orbital_period_days=row["orbital_period_days"],
                    radius_earth_radii=row["radius_earth_radii"],
                    mass_earth_masses=row["mass_earth_masses"],
                    equilibrium_temperature_k=row["equilibrium_temperature_k"],
                    distance_parsecs=row["distance_parsecs"],
                    confidence=row["confidence"],
                    status=row["status"],
                    habitable_zone=bool(row["habitable_zone"]),
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
                for row in rows
            ]

    async def insert_search_result(self, result: SearchResult) -> int:
        """Insert search result"""
        async with self._lock:
            conn = self._get_connection()
            cursor = conn.execute(
                """
                INSERT INTO search_results (
                    target_name, catalog, mission, method, exoplanet_detected,
                    detection_confidence, processing_time_ms, result_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    result.target_name,
                    result.catalog,
                    result.mission,
                    result.method,
                    result.exoplanet_detected,
                    result.detection_confidence,
                    result.processing_time_ms,
                    result.result_data,
                ),
            )
            conn.commit()
            return cursor.lastrowid

    async def get_search_history(
        self, limit: int = 50, method: Optional[str] = None
    ) -> List[SearchResult]:
        """Get search history"""
        async with self._lock:
            conn = self._get_connection()

            query = "SELECT * FROM search_results WHERE 1=1"
            params = []

            if method:
                # Validate method parameter to prevent SQL injection
                allowed_methods = ["bls", "gpi", "ensemble", "hybrid"]
                if method in allowed_methods:
                    query += " AND method = ?"
                    params.append(method)
                else:
                    raise ValueError(f"Invalid search method: {method}")

            # Validate limit to prevent resource exhaustion
            if limit > 1000:  # Maximum 1k results
                limit = 100

            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

            return [
                SearchResult(
                    id=row["id"],
                    target_name=row["target_name"],
                    catalog=row["catalog"],
                    mission=row["mission"],
                    method=row["method"],
                    exoplanet_detected=bool(row["exoplanet_detected"]),
                    detection_confidence=row["detection_confidence"],
                    processing_time_ms=row["processing_time_ms"],
                    result_data=row["result_data"],
                    created_at=row["created_at"],
                )
                for row in rows
            ]

    async def record_metric(
        self,
        service_name: str,
        metric_name: str,
        metric_value: float,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Record system metric"""
        async with self._lock:
            conn = self._get_connection()
            conn.execute(
                """
                INSERT INTO system_metrics (service_name, metric_name, metric_value, metadata)
                VALUES (?, ?, ?, ?)
            """,
                (service_name, metric_name, metric_value, json.dumps(metadata or {})),
            )
            conn.commit()

    async def get_metrics(
        self, service_name: Optional[str] = None, hours: int = 24
    ) -> List[SystemMetrics]:
        """Get system metrics"""
        async with self._lock:
            conn = self._get_connection()
            query = """
                SELECT * FROM system_metrics
                WHERE timestamp >= datetime('now', '-? hours')
            """
            params = [hours]

            if service_name:
                # Validate service_name to prevent SQL injection
                allowed_services = [
                    "bls_service",
                    "gpi_service",
                    "cpp_accelerated",
                    "python_fallback",
                    "gpi_generator",
                    "search_accelerator",
                ]
                if service_name in allowed_services:
                    query += " AND service_name = ?"
                    params.append(service_name)
                else:
                    # Only allow service names that follow the expected pattern
                    import re

                    if re.match(r"^[a-zA-Z0-9_-]+$", service_name):
                        query += " AND service_name = ?"
                        params.append(service_name)
                    else:
                        raise ValueError(f"Invalid service name: {service_name}")

            query += " ORDER BY timestamp DESC"

            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

            return [
                SystemMetrics(
                    id=row["id"],
                    timestamp=row["timestamp"],
                    service_name=row["service_name"],
                    metric_name=row["metric_name"],
                    metric_value=row["metric_value"],
                    metadata=row["metadata"],
                )
                for row in rows
            ]

    async def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        async with self._lock:
            conn = self._get_connection()

            stats = {}

            # Exoplanet statistics
            cursor = conn.execute("SELECT COUNT(*) FROM exoplanets")
            stats["total_exoplanets"] = cursor.fetchone()[0]

            cursor = conn.execute(
                "SELECT COUNT(*) FROM exoplanets WHERE status = 'confirmed'"
            )
            stats["confirmed_exoplanets"] = cursor.fetchone()[0]

            cursor = conn.execute(
                "SELECT COUNT(*) FROM exoplanets WHERE habitable_zone = 1"
            )
            stats["habitable_zone_planets"] = cursor.fetchone()[0]

            cursor = conn.execute("SELECT AVG(confidence) FROM exoplanets")
            result = cursor.fetchone()[0]
            stats["average_confidence"] = round(result, 3) if result else 0.0

            # Search statistics
            cursor = conn.execute("SELECT COUNT(*) FROM search_results")
            stats["total_searches"] = cursor.fetchone()[0]

            cursor = conn.execute(
                """
                SELECT COUNT(*) FROM search_results
                WHERE created_at >= datetime('now', '-24 hours')
            """
            )
            stats["searches_last_24h"] = cursor.fetchone()[0]

            cursor = conn.execute(
                """
                SELECT method, COUNT(*) as count FROM search_results
                GROUP BY method ORDER BY count DESC
            """
            )
            stats["searches_by_method"] = dict(cursor.fetchall())

            return stats

    async def cleanup_old_data(self, days: int = 30):
        """Clean up old data"""
        async with self._lock:
            conn = self._get_connection()

            # Clean old search results
            cursor = conn.execute(
                """
                DELETE FROM search_results
                WHERE created_at < datetime('now', ? || ' days')
            """,
                (f"-{days}",),
            )

            # Clean old metrics
            cursor = conn.execute(
                """
                DELETE FROM system_metrics
                WHERE timestamp < datetime('now', ? || ' days')
            """,
                (f"-{days}",),
            )

            conn.commit()
            logger.info(f"✅ Cleaned up data older than {days} days")

    async def close(self):
        """Close database connection"""
        if self._connection:
            self._connection.close()
            self._connection = None
            logger.info("Database connection closed")


# Global database instance
db_manager = DatabaseManager()
