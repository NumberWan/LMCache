#!/usr/bin/env python3
"""
LMCache Request-based CSV Exporter

This script exports LMCache statistics per request to CSV files.
It hooks into LMCache operations to capture per-request statistics.
"""

import time
import argparse
import signal
import sys
import threading
from typing import Optional, Dict
from lmcache.observability import LMCStatsMonitor


class LMCacheRequestExporter:
    def __init__(self, output_file: str, instance_id: str = None):
        self.output_file = output_file
        self.instance_id = instance_id
        self.running = True
        self.stats_monitor = LMCStatsMonitor.GetOrCreate()
        self.request_counter = 0
        
        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        print(f"\nReceived signal {signum}, shutting down gracefully...")
        self.running = False
    
    def export_request_stats(self, request_id: str = None):
        """Export statistics for a single request"""
        if request_id is None:
            self.request_counter += 1
            request_id = f"request_{self.request_counter}"
        
        # Export per-request stats (this will clear the stats)
        self.stats_monitor.export_request_stats_to_csv(
            filename=self.output_file,
            request_id=request_id,
            instance_id=self.instance_id
        )
        
        print(f"Exported stats for request {request_id} to {self.output_file}")
    
    def run_periodic_export(self, interval: float = 1.0):
        """Run periodic export (for testing)"""
        print(f"Starting LMCache request exporter...")
        print(f"Output file: {self.output_file}")
        print(f"Export interval: {interval} seconds")
        print(f"Instance ID: {self.instance_id or 'auto-detect'}")
        print("Press Ctrl+C to stop")
        
        while self.running:
            try:
                # Simulate request processing
                self.export_request_stats()
                time.sleep(interval)
                
            except Exception as e:
                print(f"Error during export: {e}")
                time.sleep(interval)
        
        print("LMCache request exporter stopped.")


def main():
    parser = argparse.ArgumentParser(description="Export LMCache statistics per request to CSV")
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="/home/w00917303/vllm_lmcache_requests.csv",
        help="Output CSV file path (default: /home/w00917303/vllm_lmcache_requests.csv)"
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
    exporter = LMCacheRequestExporter(
        output_file=args.output,
        instance_id=args.instance_id
    )
    
    exporter.run_periodic_export(interval=args.interval)


if __name__ == "__main__":
    main()
