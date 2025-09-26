"""Core business logic modules for PDF to Markdown MCP Server."""

# Always available - core watcher functionality
from .watcher import (
    DirectoryWatcher,
    PDFFileHandler,
    WatcherConfig,
    FileValidator,
    SmartFileDetector,
)

__all__ = [
    "DirectoryWatcher",
    "PDFFileHandler",
    "WatcherConfig",
    "FileValidator",
    "SmartFileDetector",
]

# Optional imports that require database dependencies
try:
    from .task_queue import (
        TaskQueue,
        create_task_queue,
    )

    __all__.extend(
        [
            "TaskQueue",
            "create_task_queue",
        ]
    )
except ImportError:
    # TaskQueue requires sqlalchemy - not available in this environment
    TaskQueue = None
    create_task_queue = None

try:
    from .watcher_service import (
        WatcherManager,
        create_watcher_service,
        create_default_watcher_config,
        get_watcher_manager,
    )

    __all__.extend(
        [
            "WatcherManager",
            "create_watcher_service",
            "create_default_watcher_config",
            "get_watcher_manager",
        ]
    )
except ImportError:
    # WatcherService requires TaskQueue - not available in this environment
    WatcherManager = None
    create_watcher_service = None
    create_default_watcher_config = None
    get_watcher_manager = None

# Monitoring and health check components
try:
    from .monitoring import (
        MetricsCollector,
        HealthMonitor,
        ComponentHealth,
        HealthStatus,
        SystemHealth,
        AlertingEngine,
        AlertRule,
        AlertSeverity,
        TracingManager,
        metrics_collector,
        health_monitor,
        alerting_engine,
        create_standard_alerts,
    )

    __all__.extend(
        [
            "MetricsCollector",
            "HealthMonitor",
            "ComponentHealth",
            "HealthStatus",
            "SystemHealth",
            "AlertingEngine",
            "AlertRule",
            "AlertSeverity",
            "TracingManager",
            "metrics_collector",
            "health_monitor",
            "alerting_engine",
            "create_standard_alerts",
        ]
    )
except ImportError:
    # Monitoring components may not be available if dependencies are missing
    MetricsCollector = None
    HealthMonitor = None
    ComponentHealth = None
    HealthStatus = None
    SystemHealth = None
    AlertingEngine = None
    AlertRule = None
    AlertSeverity = None
    TracingManager = None
    metrics_collector = None
    health_monitor = None
    alerting_engine = None
    create_standard_alerts = None
