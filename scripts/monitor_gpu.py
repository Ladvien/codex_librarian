#!/usr/bin/env python3
"""
Real-time GPU monitoring dashboard script.

Provides real-time GPU status monitoring, model loading tracking,
and alerts for GPU issues like high memory usage and temperature.
"""

import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import requests
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn
from rich.table import Table
from rich.text import Text

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from pdf_to_markdown_mcp.core.monitoring import (
        HealthMonitor,
        MetricsCollector,
        HealthStatus
    )
    from pdf_to_markdown_mcp.config import settings
    MONITORING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import monitoring modules: {e}")
    MONITORING_AVAILABLE = False

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class GPUMonitor:
    """Real-time GPU monitoring and alerting."""

    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.console = Console()
        # Don't initialize collectors to avoid registry conflicts
        self.health_monitor = None
        self.metrics_collector = None

        # Alert thresholds
        self.memory_critical_threshold = 95.0  # %
        self.memory_warning_threshold = 85.0   # %
        self.temp_critical_threshold = 85      # °C
        self.temp_warning_threshold = 75       # °C

        # Alert tracking
        self.last_alerts = {}
        self.alert_cooldown = 300  # 5 minutes

        # Model loading tracking
        self.model_load_count = 0
        self.last_model_load = None

    async def get_gpu_status_from_api(self) -> Dict[str, Any]:
        """Get GPU status from health API endpoint."""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                return {
                    "gpu": health_data.get("components", {}).get("gpu", {}),
                    "cuda": health_data.get("components", {}).get("cuda", {}),
                    "system": health_data.get("components", {}).get("system", {}),
                }
        except Exception as e:
            return {"error": str(e)}

        return {}

    async def get_gpu_metrics_from_api(self) -> Dict[str, Any]:
        """Get GPU metrics from JSON metrics endpoint."""
        try:
            response = requests.get(f"{self.api_url}/metrics/json", timeout=10)
            if response.status_code == 200:
                return response.json().get("gpu", {})
        except Exception as e:
            return {"error": str(e)}

        return {}

    async def get_direct_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information directly using pynvml."""
        if not NVML_AVAILABLE:
            return {"error": "NVIDIA ML library not available"}

        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()

            if device_count == 0:
                return {"error": "No GPU devices found"}

            devices = {}
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name_bytes = pynvml.nvmlDeviceGetName(handle)
                name = name_bytes.decode("utf-8") if isinstance(name_bytes, bytes) else str(name_bytes)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

                # Get power info
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                except:
                    power = None

                # Get clock speeds
                try:
                    graphics_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
                    memory_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                except:
                    graphics_clock = None
                    memory_clock = None

                devices[f"gpu_{i}"] = {
                    "name": name,
                    "memory_used_mb": round(mem_info.used / (1024**2), 1),
                    "memory_total_mb": round(mem_info.total / (1024**2), 1),
                    "memory_percent": round((mem_info.used / mem_info.total) * 100, 1),
                    "utilization_percent": util.gpu,
                    "memory_utilization_percent": util.memory,
                    "temperature_c": temp,
                    "power_watts": power,
                    "graphics_clock_mhz": graphics_clock,
                    "memory_clock_mhz": memory_clock,
                }

            return {"devices": devices, "device_count": device_count}

        except Exception as e:
            return {"error": str(e)}

    def create_gpu_table(self, gpu_data: Dict[str, Any]) -> Table:
        """Create a Rich table showing GPU status."""
        table = Table(title="GPU Status", show_header=True, header_style="bold magenta")

        table.add_column("Device", style="cyan", no_wrap=True)
        table.add_column("Name", style="white")
        table.add_column("Memory", style="blue")
        table.add_column("Utilization", style="green")
        table.add_column("Temperature", style="red")
        table.add_column("Power", style="yellow")
        table.add_column("Status", style="white")

        if "devices" in gpu_data:
            for device_id, device in gpu_data["devices"].items():
                # Memory usage
                memory_used = device.get("memory_used_mb", 0)
                memory_total = device.get("memory_total_mb", 0)
                memory_percent = device.get("memory_percent", 0)

                memory_text = f"{memory_used:.0f}/{memory_total:.0f} MB ({memory_percent:.1f}%)"

                # Utilization
                gpu_util = device.get("utilization_percent", 0)
                mem_util = device.get("memory_utilization_percent", 0)
                util_text = f"GPU: {gpu_util}% | Mem: {mem_util}%"

                # Temperature
                temp = device.get("temperature_c", 0)
                temp_text = f"{temp}°C"
                if temp > self.temp_critical_threshold:
                    temp_text = f"[bold red]{temp_text}[/bold red]"
                elif temp > self.temp_warning_threshold:
                    temp_text = f"[bold yellow]{temp_text}[/bold yellow]"

                # Power
                power = device.get("power_watts")
                power_text = f"{power:.1f}W" if power is not None else "N/A"

                # Status
                if memory_percent > self.memory_critical_threshold or temp > self.temp_critical_threshold:
                    status = "[bold red]CRITICAL[/bold red]"
                elif memory_percent > self.memory_warning_threshold or temp > self.temp_warning_threshold:
                    status = "[bold yellow]WARNING[/bold yellow]"
                else:
                    status = "[bold green]HEALTHY[/bold green]"

                table.add_row(
                    device_id.replace("gpu_", "GPU "),
                    device.get("name", "Unknown"),
                    memory_text,
                    util_text,
                    temp_text,
                    power_text,
                    status
                )
        else:
            error_msg = gpu_data.get("error", "No GPU data available")
            table.add_row("N/A", error_msg, "N/A", "N/A", "N/A", "N/A", "[bold red]ERROR[/bold red]")

        return table

    def create_system_table(self, system_data: Dict[str, Any]) -> Table:
        """Create a table showing system resource status."""
        table = Table(title="System Resources", show_header=True, header_style="bold cyan")

        table.add_column("Resource", style="cyan")
        table.add_column("Usage", style="white")
        table.add_column("Available", style="green")
        table.add_column("Status", style="white")

        if PSUTIL_AVAILABLE:
            try:
                # Memory
                memory = psutil.virtual_memory()
                memory_status = "CRITICAL" if memory.percent > 90 else "WARNING" if memory.percent > 75 else "OK"
                table.add_row(
                    "Memory",
                    f"{memory.percent:.1f}%",
                    f"{memory.available / (1024**3):.1f} GB",
                    f"[bold {'red' if memory_status == 'CRITICAL' else 'yellow' if memory_status == 'WARNING' else 'green'}]{memory_status}[/bold {'red' if memory_status == 'CRITICAL' else 'yellow' if memory_status == 'WARNING' else 'green'}]"
                )

                # CPU
                cpu_percent = psutil.cpu_percent(interval=0.1)
                cpu_status = "CRITICAL" if cpu_percent > 95 else "WARNING" if cpu_percent > 80 else "OK"
                table.add_row(
                    "CPU",
                    f"{cpu_percent:.1f}%",
                    f"{psutil.cpu_count()} cores",
                    f"[bold {'red' if cpu_status == 'CRITICAL' else 'yellow' if cpu_status == 'WARNING' else 'green'}]{cpu_status}[/bold {'red' if cpu_status == 'CRITICAL' else 'yellow' if cpu_status == 'WARNING' else 'green'}]"
                )

                # Disk
                try:
                    disk = psutil.disk_usage("/")
                    disk_percent = (disk.used / disk.total) * 100
                    disk_status = "CRITICAL" if disk.free < 1e9 else "WARNING" if disk.free < 5e9 else "OK"
                    table.add_row(
                        "Disk",
                        f"{disk_percent:.1f}%",
                        f"{disk.free / (1024**3):.1f} GB",
                        f"[bold {'red' if disk_status == 'CRITICAL' else 'yellow' if disk_status == 'WARNING' else 'green'}]{disk_status}[/bold {'red' if disk_status == 'CRITICAL' else 'yellow' if disk_status == 'WARNING' else 'green'}]"
                    )
                except:
                    table.add_row("Disk", "N/A", "N/A", "[yellow]UNKNOWN[/yellow]")

            except Exception as e:
                table.add_row("System", f"Error: {e}", "N/A", "[bold red]ERROR[/bold red]")
        else:
            table.add_row("System", "psutil not available", "N/A", "[yellow]UNAVAILABLE[/yellow]")

        return table

    def create_alerts_panel(self) -> Panel:
        """Create a panel showing recent alerts."""
        alert_text = Text()

        current_time = time.time()
        recent_alerts = []

        # Check for current alerts based on latest data
        # This would typically be populated by check_alerts method
        if not hasattr(self, '_recent_alerts'):
            self._recent_alerts = []

        if self._recent_alerts:
            for alert in self._recent_alerts[-5:]:  # Show last 5 alerts
                timestamp = datetime.fromtimestamp(alert['timestamp']).strftime('%H:%M:%S')
                alert_text.append(f"[{alert['severity']}]{timestamp}: {alert['message']}[/{alert['severity']}]\n")
        else:
            alert_text.append("[green]No recent alerts[/green]")

        return Panel(alert_text, title="Recent Alerts", border_style="red")

    def create_model_info_panel(self, gpu_data: Dict[str, Any]) -> Panel:
        """Create a panel showing model loading information."""
        info_text = Text()

        # Model loading frequency
        info_text.append(f"Model Loads: {self.model_load_count}\n", style="cyan")

        if self.last_model_load:
            last_load_time = datetime.fromtimestamp(self.last_model_load).strftime('%H:%M:%S')
            info_text.append(f"Last Load: {last_load_time}\n", style="white")

        # GPU memory for models
        if "devices" in gpu_data:
            for device_id, device in gpu_data["devices"].items():
                memory_used = device.get("memory_used_mb", 0)
                memory_total = device.get("memory_total_mb", 0)
                if memory_used > 100:  # Assume model is loaded if >100MB used
                    info_text.append(f"{device_id}: Model likely loaded ({memory_used:.0f} MB)\n", style="green")
                else:
                    info_text.append(f"{device_id}: No model loaded\n", style="yellow")

        return Panel(info_text, title="Model Status", border_style="blue")

    async def check_alerts(self, gpu_data: Dict[str, Any]):
        """Check for alert conditions and record them."""
        if not hasattr(self, '_recent_alerts'):
            self._recent_alerts = []

        current_time = time.time()

        if "devices" in gpu_data:
            for device_id, device in gpu_data["devices"].items():
                memory_percent = device.get("memory_percent", 0)
                temp = device.get("temperature_c", 0)

                # Memory alerts
                if memory_percent > self.memory_critical_threshold:
                    alert_key = f"{device_id}_memory_critical"
                    if self._should_alert(alert_key, current_time):
                        self._recent_alerts.append({
                            "timestamp": current_time,
                            "severity": "red",
                            "message": f"{device_id}: Memory usage critical ({memory_percent:.1f}%)"
                        })
                        self.last_alerts[alert_key] = current_time

                elif memory_percent > self.memory_warning_threshold:
                    alert_key = f"{device_id}_memory_warning"
                    if self._should_alert(alert_key, current_time):
                        self._recent_alerts.append({
                            "timestamp": current_time,
                            "severity": "yellow",
                            "message": f"{device_id}: Memory usage high ({memory_percent:.1f}%)"
                        })
                        self.last_alerts[alert_key] = current_time

                # Temperature alerts
                if temp > self.temp_critical_threshold:
                    alert_key = f"{device_id}_temp_critical"
                    if self._should_alert(alert_key, current_time):
                        self._recent_alerts.append({
                            "timestamp": current_time,
                            "severity": "red",
                            "message": f"{device_id}: Temperature critical ({temp}°C)"
                        })
                        self.last_alerts[alert_key] = current_time

                elif temp > self.temp_warning_threshold:
                    alert_key = f"{device_id}_temp_warning"
                    if self._should_alert(alert_key, current_time):
                        self._recent_alerts.append({
                            "timestamp": current_time,
                            "severity": "yellow",
                            "message": f"{device_id}: Temperature high ({temp}°C)"
                        })
                        self.last_alerts[alert_key] = current_time

        # Keep only recent alerts (last hour)
        self._recent_alerts = [
            alert for alert in self._recent_alerts
            if current_time - alert["timestamp"] < 3600
        ]

    def _should_alert(self, alert_key: str, current_time: float) -> bool:
        """Check if enough time has passed since last alert of this type."""
        last_alert_time = self.last_alerts.get(alert_key, 0)
        return current_time - last_alert_time > self.alert_cooldown

    def create_layout(self, gpu_data: Dict[str, Any], system_data: Dict[str, Any]) -> Layout:
        """Create the main dashboard layout."""
        layout = Layout()

        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=8)
        )

        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )

        layout["footer"].split_row(
            Layout(name="alerts"),
            Layout(name="model_info")
        )

        # Header
        header_text = Text(f"GPU Monitor Dashboard - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", style="bold white on blue")
        layout["header"].update(Panel(header_text, style="blue"))

        # Main content
        layout["left"].update(self.create_gpu_table(gpu_data))
        layout["right"].update(self.create_system_table(system_data))

        # Footer
        layout["alerts"].update(self.create_alerts_panel())
        layout["model_info"].update(self.create_model_info_panel(gpu_data))

        return layout

    async def run_dashboard(self, refresh_interval: float = 2.0):
        """Run the real-time GPU monitoring dashboard."""
        self.console.print("[bold green]Starting GPU Monitor Dashboard...[/bold green]")
        self.console.print(f"API URL: {self.api_url}")
        self.console.print(f"Refresh interval: {refresh_interval}s")
        self.console.print("Press Ctrl+C to exit\n")

        with Live(console=self.console, refresh_per_second=1/refresh_interval) as live:
            try:
                while True:
                    # Get GPU data
                    gpu_data = await self.get_direct_gpu_info()

                    # Get system data (dummy for now, could be from API)
                    system_data = {}

                    # Check for alerts
                    await self.check_alerts(gpu_data)

                    # Update display
                    layout = self.create_layout(gpu_data, system_data)
                    live.update(layout)

                    await asyncio.sleep(refresh_interval)

            except KeyboardInterrupt:
                self.console.print("\n[bold red]Dashboard stopped by user[/bold red]")

    async def check_gpu_health(self) -> Dict[str, Any]:
        """Perform comprehensive GPU health check."""
        results = {
            "timestamp": datetime.now().isoformat(),
            "gpu_direct": await self.get_direct_gpu_info(),
            "gpu_api": await self.get_gpu_status_from_api(),
            "gpu_metrics": await self.get_gpu_metrics_from_api(),
        }

        # Health summary
        gpu_healthy = True
        issues = []

        if "devices" in results["gpu_direct"]:
            for device_id, device in results["gpu_direct"]["devices"].items():
                memory_percent = device.get("memory_percent", 0)
                temp = device.get("temperature_c", 0)

                if memory_percent > self.memory_critical_threshold:
                    gpu_healthy = False
                    issues.append(f"{device_id}: Critical memory usage ({memory_percent:.1f}%)")

                if temp > self.temp_critical_threshold:
                    gpu_healthy = False
                    issues.append(f"{device_id}: Critical temperature ({temp}°C)")

        results["health_summary"] = {
            "healthy": gpu_healthy,
            "issues": issues,
            "device_count": results["gpu_direct"].get("device_count", 0)
        }

        return results


async def main():
    """Main function to run GPU monitoring."""
    import argparse

    parser = argparse.ArgumentParser(description="GPU Monitoring Dashboard")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API URL")
    parser.add_argument("--refresh", type=float, default=2.0, help="Refresh interval in seconds")
    parser.add_argument("--check-only", action="store_true", help="Run single health check and exit")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")

    args = parser.parse_args()

    monitor = GPUMonitor(api_url=args.api_url)

    if args.check_only:
        # Single health check
        results = await monitor.check_gpu_health()

        if args.json:
            print(json.dumps(results, indent=2))
        else:
            console = Console()
            console.print(f"[bold]GPU Health Check - {results['timestamp']}[/bold]")

            health = results["health_summary"]
            status_color = "green" if health["healthy"] else "red"
            console.print(f"Status: [bold {status_color}]{'HEALTHY' if health['healthy'] else 'UNHEALTHY'}[/bold {status_color}]")
            console.print(f"Devices: {health['device_count']}")

            if health["issues"]:
                console.print("\n[bold red]Issues:[/bold red]")
                for issue in health["issues"]:
                    console.print(f"  • {issue}")

            if "devices" in results["gpu_direct"]:
                console.print("\n[bold]Device Details:[/bold]")
                for device_id, device in results["gpu_direct"]["devices"].items():
                    console.print(f"  {device_id}: {device['name']}")
                    console.print(f"    Memory: {device['memory_used_mb']:.0f}/{device['memory_total_mb']:.0f} MB ({device['memory_percent']:.1f}%)")
                    console.print(f"    Temperature: {device['temperature_c']}°C")
                    console.print(f"    Utilization: {device['utilization_percent']}%")
    else:
        # Run dashboard
        await monitor.run_dashboard(refresh_interval=args.refresh)


if __name__ == "__main__":
    asyncio.run(main())