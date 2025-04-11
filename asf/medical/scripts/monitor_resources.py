"""
Module description.

This module provides functionality for...
"""
import time
import json
import argparse
import logging
import platform
import psutil
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def get_system_info() -> dict:
    """
    Get system information.
    
    Returns:
        Dictionary with system information
    """
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "memory_total": psutil.virtual_memory().total,
        "disk_total": psutil.disk_usage('/').total
    }

def get_resource_usage() -> dict:
    """
    Get current resource usage.
    
    Returns:
        Dictionary with resource usage information
    """
    return {
        "timestamp": datetime.now().isoformat(),
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_percent": psutil.virtual_memory().percent,
        "memory_used": psutil.virtual_memory().used,
        "disk_percent": psutil.disk_usage('/').percent,
        "disk_used": psutil.disk_usage('/').used,
        "network_sent": psutil.net_io_counters().bytes_sent,
        "network_recv": psutil.net_io_counters().bytes_recv
    }

def get_process_info(process_name: str = None) -> list:
    """
    Get information about processes.
    
    Args:
        process_name: Optional process name filter
        
    Returns:
        List of dictionaries with process information
    """
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'username', 'memory_percent', 'cpu_percent']):
        try:
            proc_info = proc.info
            
            if process_name and process_name.lower() not in proc_info['name'].lower():
                continue
            
            processes.append({
                "pid": proc_info['pid'],
                "name": proc_info['name'],
                "username": proc_info['username'],
                "memory_percent": proc_info['memory_percent'],
                "cpu_percent": proc_info['cpu_percent']
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    return sorted(processes, key=lambda x: x['memory_percent'], reverse=True)

def monitor_resources(interval: int, count: int, output_file: str = None, process_name: str = None) -> None:
    """
    Monitor system resources.
    
    Args:
        interval: Monitoring interval in seconds
        count: Number of monitoring iterations (0 for infinite)
        output_file: Optional output file for JSON data
        process_name: Optional process name filter
    """
    system_info = get_system_info()
    logger.info(f"System information: {json.dumps(system_info, indent=2)}")
    
    data = {
        "system_info": system_info,
        "resource_usage": []
    }
    
    iteration = 0
    try:
        while count == 0 or iteration < count:
            resource_usage = get_resource_usage()
            logger.info(f"Resource usage: CPU: {resource_usage['cpu_percent']}%, Memory: {resource_usage['memory_percent']}%, Disk: {resource_usage['disk_percent']}%")
            
            if process_name:
                processes = get_process_info(process_name)
                if processes:
                    logger.info(f"Top processes matching '{process_name}':")
                    for proc in processes[:5]:  # Show top 5 processes
                        logger.info(f"  PID: {proc['pid']}, Name: {proc['name']}, Memory: {proc['memory_percent']:.2f}%, CPU: {proc['cpu_percent']:.2f}%")
                else:
                    logger.info(f"No processes matching '{process_name}' found")
            
            data["resource_usage"].append(resource_usage)
            
            if output_file:
                with open(output_file, 'w') as f:
                    json.dump(data, f, indent=2)
            
            iteration += 1
            
            if count == 0 or iteration < count:
                time.sleep(interval)
    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user")
    
    logger.info(f"Monitoring completed after {iteration} iterations")

def main():
    """Main function.

    Args:
        # TODO: Add parameter descriptions

    Returns:
        # TODO: Add return description
    """
    parser = argparse.ArgumentParser(description="Monitor system resources")
    parser.add_argument("--interval", type=int, default=5, help="Monitoring interval in seconds")
    parser.add_argument("--count", type=int, default=0, help="Number of monitoring iterations (0 for infinite)")
    parser.add_argument("--output", help="Output file for JSON data")
    parser.add_argument("--process", help="Process name filter")
    args = parser.parse_args()
    
    monitor_resources(args.interval, args.count, args.output, args.process)

if __name__ == "__main__":
    main()
