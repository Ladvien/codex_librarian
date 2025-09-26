"""Core business logic modules for PDF to Markdown MCP Server."""

# Always available - core watcher functionality
from .watcher import (
    DirectoryWatcher,
    FileValidator,
    PDFFileHandler,
    SmartFileDetector,
    WatcherConfig,
)

__all__ = [
    "DirectoryWatcher",
    "FileValidator",
    "PDFFileHandler",
    "SmartFileDetector",
    "WatcherConfig",
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
        create_default_watcher_config,
        create_watcher_service,
        get_watcher_manager,
    )

    __all__.extend(
        [
            "WatcherManager",
            "create_default_watcher_config",
            "create_watcher_service",
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
        AlertingEngine,
        AlertRule,
        AlertSeverity,
        ComponentHealth,
        HealthMonitor,
        HealthStatus,
        MetricsCollector,
        SystemHealth,
        TracingManager,
        alerting_engine,
        create_standard_alerts,
        health_monitor,
        metrics_collector,
    )

    __all__.extend(
        [
            "AlertRule",
            "AlertSeverity",
            "AlertingEngine",
            "ComponentHealth",
            "HealthMonitor",
            "HealthStatus",
            "MetricsCollector",
            "SystemHealth",
            "TracingManager",
            "alerting_engine",
            "create_standard_alerts",
            "health_monitor",
            "metrics_collector",
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
