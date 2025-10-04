# ExoplanetAI Data Management System Guide

## Overview

ExoplanetAI v2.0 includes a comprehensive data management system that handles automated ingestion, validation, versioning, and preprocessing of astronomical data from NASA, ESA, and other space agencies.

## Architecture

### Core Components

1. **DataManager** - Central coordinator for data ingestion
2. **StorageManager** - Handles data storage and caching
3. **DataValidator** - Validates data integrity and quality
4. **VersionManager** - Git-based data versioning system
5. **LightCurveProcessor** - Advanced preprocessing pipeline

### Data Flow

```
Raw Sources (NASA/ESA/MAST) → Ingestion → Validation → Storage → Versioning → Preprocessing
```

## Data Sources

### Supported Catalogs

- **KOI (Kepler Objects of Interest)** - NASA Exoplanet Archive
- **TOI (TESS Objects of Interest)** - ExoFOP-TESS
- **K2 Candidates** - NASA Exoplanet Archive
- **Light Curves** - MAST via lightkurve

### Data Freshness

- **Planet catalogs**: Cached for 24 hours (KOI/K2) or 12 hours (TOI)
- **Light curves**: Cached for 6 hours
- **Processed data**: Cached until source data changes

## Storage Schema

### Directory Structure

```
data/
├── raw/                    # Raw downloaded data (never modified)
│   ├── nasa/              # NASA Exoplanet Archive data
│   ├── mast/              # MAST archive data
│   ├── exofop/            # ExoFOP-TESS data
│   ├── kepler/            # Kepler mission data
│   └── tess/              # TESS mission data
├── processed/             # Processed data with versioning
│   ├── v1/                # Version 1 processed data
│   ├── v2/                # Version 2 processed data
│   └── lightcurves/       # Processed light curves
├── lightcurves/           # Raw light curve FITS files
│   ├── tess/              # TESS light curves
│   ├── kepler/            # Kepler light curves
│   └── k2/                # K2 light curves
├── metadata/              # Data metadata and manifests
├── versions/              # Version snapshots
├── cache/                 # Redis/file cache
└── checksums/             # Data integrity checksums
```

### File Formats

- **Tables**: CSV format with metadata JSON
- **Light curves**: FITS format following standard conventions
- **Processed data**: FITS with processing history in headers
- **Metadata**: JSON with comprehensive provenance information

## API Endpoints

### Data Ingestion

#### Ingest Single Table
```http
POST /api/v1/data/ingest/table
Content-Type: application/json

{
    "table_type": "koi",
    "force_refresh": false
}
```

#### Ingest Light Curve
```http
POST /api/v1/data/ingest/lightcurve
Content-Type: application/json

{
    "target_name": "TIC 441420236",
    "mission": "TESS",
    "sector_quarter": 1,
    "force_refresh": false
}
```

#### Batch Ingest All Tables
```http
POST /api/v1/data/ingest/batch?force_refresh=false
```

### Data Validation

#### Validate Ingested Data
```http
POST /api/v1/data/validate
Content-Type: application/json

{
    "data_type": "koi",
    "target_name": null
}
```

### Data Preprocessing

#### Preprocess Light Curve
```http
POST /api/v1/data/preprocess/lightcurve
Content-Type: application/json

{
    "target_name": "TIC 441420236",
    "mission": "TESS",
    "processing_params": {
        "remove_outliers": true,
        "sigma_clip_sigma": 5.0,
        "baseline_window_length": 101,
        "wavelet_denoising": false,
        "normalize_method": "median"
    }
}
```

### Data Versioning

#### Create Version
```http
POST /api/v1/data/version/create
Content-Type: application/json

{
    "version_name": "v1.2.0",
    "description": "Updated KOI catalog with new dispositions",
    "file_patterns": ["*.csv", "*.fits"]
}
```

#### List Versions
```http
GET /api/v1/data/version/list
```

#### Get Version Info
```http
GET /api/v1/data/version/v1.2.0
```

### Storage Management

#### Get Storage Statistics
```http
GET /api/v1/data/storage/stats
```

#### Get Ingestion Status
```http
GET /api/v1/data/ingestion/status
```

## Data Validation

### Validation Levels

1. **Schema Validation** - Required columns, data types
2. **Range Validation** - Physical parameter bounds
3. **Coordinate Validation** - RA/Dec ranges and precision
4. **Duplicate Detection** - Cross-matching and deduplication
5. **Quality Assessment** - Data completeness and accuracy

### Quality Metrics

- **Completeness**: Percentage of non-null values
- **Accuracy**: Validation against known ranges
- **Consistency**: Cross-validation between parameters
- **Freshness**: Data age and update frequency

## Light Curve Preprocessing

### Processing Pipeline

1. **Quality Filtering** - Remove bad data points using quality flags
2. **Outlier Detection** - Statistical and ML-based outlier removal
3. **Gap Detection** - Identify and handle data gaps
4. **Baseline Removal** - Savitzky-Golay detrending per segment
5. **Wavelet Denoising** - Optional noise reduction
6. **Normalization** - Flux scaling and centering

### Processing Parameters

```python
default_params = {
    "remove_outliers": True,
    "sigma_clip_sigma": 5.0,
    "sigma_clip_maxiters": 3,
    "baseline_window_length": 101,
    "baseline_polyorder": 2,
    "wavelet_denoising": False,
    "wavelet_type": "db4",
    "wavelet_levels": 6,
    "normalize_method": "median",
    "quality_bitmask": "default",
    "gap_threshold_hours": 0.5,
    "min_points_per_segment": 100
}
```

### Quality Filtering

#### TESS Quality Flags
- **1**: Attitude tweak
- **2**: Safe mode
- **4**: Coarse point
- **8**: Earth point
- **16**: Argabrightening
- **32**: Reaction wheel desaturation
- **64**: Manual exclude
- **128**: Discontinuity
- **256**: Impulsive outlier
- **1024**: Cosmic ray
- **2048**: Straylight

#### Filtering Presets
- **none**: No filtering (bitmask = 0)
- **default**: Moderate filtering (removes major issues)
- **hard**: Aggressive filtering (removes all known issues)
- **hardest**: Very aggressive (removes any non-zero flag)

### Outlier Detection Methods

1. **Sigma Clipping** - Iterative statistical clipping
2. **MAD (Median Absolute Deviation)** - Robust outlier detection
3. **IQR (Interquartile Range)** - Quartile-based detection
4. **Isolation Forest** - ML-based anomaly detection
5. **Modified Z-Score** - MAD-based z-score
6. **Ensemble** - Combination of multiple methods

### Normalization Methods

1. **Median** - Divide by median flux (default)
2. **Mean** - Divide by mean flux
3. **Robust** - Use 5th-95th percentile range
4. **Unity** - Scale to unit variance around mean=1
5. **MinMax** - Scale to [0,1] range
6. **Z-Score** - Center at 0 with unit variance
7. **Quantile** - Use median and MAD for scaling

## Data Versioning

### Version Control System

ExoplanetAI uses a Git-based versioning system for metadata and file-based versioning for large data files.

#### Version Creation Process

1. **File Collection** - Gather files matching specified patterns
2. **Checksum Calculation** - SHA256 hashes for integrity
3. **Manifest Creation** - JSON manifest with file metadata
4. **Git Commit** - Version metadata committed to Git
5. **File Storage** - Data files stored with hard links when possible

#### Version Metadata

```json
{
    "version": "v1.2.0",
    "description": "Updated KOI catalog",
    "created_at": "2024-01-15T10:30:00Z",
    "files": [
        {
            "path": "raw/nasa/koi_20240115_103000.csv",
            "hash": "sha256:abc123...",
            "size_bytes": 1048576,
            "modified_at": "2024-01-15T10:25:00Z"
        }
    ],
    "total_size_bytes": 1048576,
    "file_count": 1,
    "parent_version": "v1.1.0"
}
```

## Monitoring and Maintenance

### Health Checks

- **Data Source Connectivity** - API endpoint availability
- **Storage Health** - Disk space and file integrity
- **Cache Performance** - Hit rates and response times
- **Processing Pipeline** - Success rates and error tracking

### Automated Maintenance

- **Cache Cleanup** - Remove old cached files
- **Log Rotation** - Manage log file sizes
- **Backup Creation** - Regular data backups
- **Integrity Checks** - Verify file checksums

### Performance Optimization

- **Parallel Processing** - Concurrent data ingestion
- **Smart Caching** - Intelligent cache invalidation
- **Compression** - Efficient data storage
- **Indexing** - Fast data retrieval

## Error Handling

### Common Issues

1. **Network Timeouts** - Retry with exponential backoff
2. **Data Format Changes** - Validation and schema evolution
3. **Storage Full** - Automatic cleanup and alerts
4. **Processing Failures** - Detailed error reporting and recovery

### Recovery Procedures

1. **Data Corruption** - Restore from version history
2. **Cache Invalidation** - Force refresh from source
3. **Processing Errors** - Rollback to previous state
4. **System Failures** - Automatic restart and recovery

## Security Considerations

### Access Control

- **Role-Based Access** - User, Researcher, Admin roles
- **API Authentication** - JWT tokens and API keys
- **Rate Limiting** - Prevent abuse and overload
- **Audit Logging** - Track all data operations

### Data Protection

- **Encryption at Rest** - Sensitive data encryption
- **Secure Transport** - HTTPS/TLS for all communications
- **Backup Security** - Encrypted backup storage
- **Access Monitoring** - Real-time security monitoring

## Best Practices

### Data Ingestion

1. **Regular Updates** - Schedule automatic ingestion
2. **Validation First** - Always validate before processing
3. **Incremental Updates** - Only fetch changed data
4. **Error Handling** - Robust error recovery

### Processing

1. **Parameter Documentation** - Document all processing choices
2. **Reproducibility** - Version all processing parameters
3. **Quality Metrics** - Track processing quality
4. **Performance Monitoring** - Monitor processing times

### Storage

1. **Regular Backups** - Automated backup schedules
2. **Integrity Checks** - Regular checksum verification
3. **Cleanup Policies** - Automatic old data removal
4. **Capacity Planning** - Monitor storage growth

## Troubleshooting

### Common Problems

#### Data Ingestion Failures
```bash
# Check network connectivity
curl -I https://exoplanetarchive.ipac.caltech.edu/

# Check API logs
tail -f logs/data_ingestion.log

# Force refresh
curl -X POST "http://localhost:8000/api/v1/data/ingest/table" \
  -H "Content-Type: application/json" \
  -d '{"table_type": "koi", "force_refresh": true}'
```

#### Processing Errors
```bash
# Check processing logs
tail -f logs/preprocessing.log

# Validate input data
curl -X POST "http://localhost:8000/api/v1/data/validate" \
  -H "Content-Type: application/json" \
  -d '{"data_type": "lightcurve", "target_name": "TIC 441420236"}'
```

#### Storage Issues
```bash
# Check storage stats
curl "http://localhost:8000/api/v1/data/storage/stats"

# Clean old cache
curl -X POST "http://localhost:8000/api/v1/data/admin/cleanup?max_age_days=7"
```

## Configuration

### Environment Variables

```bash
# Data paths
DATA_PATH=/path/to/data
CACHE_PATH=/path/to/cache

# API keys
NASA_API_KEY=your_nasa_api_key
ESA_API_KEY=your_esa_api_key

# Redis configuration
REDIS_URL=redis://localhost:6379
CACHE_TTL_HOURS=6

# Processing configuration
MAX_CONCURRENT_DOWNLOADS=5
PROCESSING_TIMEOUT_MINUTES=30
```

### Configuration Files

#### data_config.yaml
```yaml
data_sources:
  nasa_archive:
    base_url: "https://exoplanetarchive.ipac.caltech.edu"
    timeout: 300
    retry_attempts: 3
  
  mast:
    base_url: "https://mast.stsci.edu"
    timeout: 600
    retry_attempts: 5

storage:
  raw_retention_days: 365
  cache_retention_days: 7
  backup_retention_days: 30

processing:
  default_quality_bitmask: "default"
  max_processing_time_minutes: 60
  parallel_processing: true
```

This comprehensive data management system ensures reliable, reproducible, and efficient handling of astronomical data for ExoplanetAI, supporting both research and production use cases.
