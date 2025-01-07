import time
import psutil
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates, nvmlDeviceGetMemoryInfo, nvmlShutdown


def get_size(bytes: int) -> str:
    """Convert bytes to human readable format"""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes < 1024:
            return f"{bytes:.2f}{unit}"
        bytes /= 1024


def print_system_usage() -> None:
    try:
        nvmlInit()
        device_count = 1  # Assuming single GPU; adjust if needed

        print("Monitoring System Usage (Ctrl+C to stop):")
        print("-" * 80)

        while True:
            # CPU Usage
            cpu_percent = psutil.cpu_percent(interval=None)

            # CPU Memory
            cpu_memory = psutil.virtual_memory()
            cpu_memory_used = cpu_memory.used
            cpu_memory_total = cpu_memory.total
            cpu_memory_percent = cpu_memory.percent

            # Print CPU stats
            # print("\033[H\033[J")  # Clear screen
            print("\n" + "=" * 80)  # Separator line
            print(f"CPU Usage: {cpu_percent}%")
            print(f"CPU Memory: {get_size(cpu_memory_used)}/{get_size(cpu_memory_total)} ({cpu_memory_percent}%)")
            print("-" * 80)

            # GPU stats
            for i in range(device_count):
                handle = nvmlDeviceGetHandleByIndex(i)
                utilization = nvmlDeviceGetUtilizationRates(handle)
                memory_info = nvmlDeviceGetMemoryInfo(handle)

                gpu_memory_used = memory_info.used
                gpu_memory_total = memory_info.total
                gpu_memory_percent = (gpu_memory_used / gpu_memory_total) * 100

                print(f"GPU {i} Usage: {utilization.gpu}%")
                print(f"GPU {i} Memory: {get_size(gpu_memory_used)}/{get_size(gpu_memory_total)} ({gpu_memory_percent:.1f}%)")

            print("-" * 80)
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nExiting monitoring.")
    finally:
        nvmlShutdown()


if __name__ == "__main__":
    print_system_usage()
