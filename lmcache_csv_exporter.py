#!/usr/bin/env python3
"""
LMCache CSV Exporter

This script exports LMCache statistics to CSV files for monitoring and analysis.
It can be run as a standalone process or integrated into existing applications.
"""

import time
import argparse
import signal
import sys
from typing import Optional
from lmcache.observability import LMCStatsMonitor


class LMCacheCSVExporter:
    def __init__(self, output_file: str, interval: float = 1.0, instance_id: str = None):
        self.output_file = output_file
        self.interval = interval
        self.instance_id = instance_id
        self.running = True
        
        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        print(f"\nReceived signal {signum}, shutting down gracefully...")
        self.running = False
    
    def run(self):
        """Main export loop"""
        print(f"Starting LMCache CSV exporter...")
        print(f"Output file: {self.output_file}")
        print(f"Export interval: {self.interval} seconds")
        print(f"Instance ID: {self.instance_id or 'auto-detect'}")
        print("Press Ctrl+C to stop")
        
        while self.running:
            try:
                # Get the global stats monitor
                stats_monitor = LMCStatsMonitor.GetOrCreate()
                
                # Export current stats
                stats_monitor.export_to_csv(
                    filename=self.output_file,
                    request_id=f"export_{int(time.time())}",
                    instance_id=self.instance_id
                )
                
                print(f"Exported stats to {self.output_file} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Wait for next export
                time.sleep(self.interval)
                
            except Exception as e:
                print(f"Error during export: {e}")
                time.sleep(self.interval)
        
        print("LMCache CSV exporter stopped.")


def main():
    parser = argparse.ArgumentParser(description="Export LMCache statistics to CSV")
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="/home/w00917303/vllm_lmcache.csv",
        help="Output CSV file path (default: /home/w00917303/vllm_lmcache.csv)"
    )
    parser.add_argument(
        "--interval", "-i",
        type=float,
        default=1.0,
        help="Export interval in seconds (default: 1.0)"
    )
    parser.add_argument(
        "--instance-id",
        type=str,
        default=None,
        help="Instance ID for identification (default: auto-detect)"
    )
    
    args = parser.parse_args()
    
    # Create and run exporter
    exporter = LMCacheCSVExporter(
        output_file=args.output,
        interval=args.interval,
        instance_id=args.instance_id
    )
    
    exporter.run()


if __name__ == "__main__":
    main()
