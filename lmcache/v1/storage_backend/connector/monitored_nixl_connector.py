"""
Monitored NIXL Connector

This is a wrapper around the NIXL connector that adds KV cache transfer monitoring.
It integrates with the production-stack monitoring system to track transfer times.
"""

import time
import logging
from typing import Optional, Any, Dict
from lmcache.v1.storage_backend.connector.nixl_connector import NixlConnector
from lmcache.v1.storage_backend.connector.nixl_connector_v2 import NixlConnectorV2

logger = logging.getLogger(__name__)

# Try to import the monitoring system
try:
    import sys
    import os
    # Add production-stack to path if available
    production_stack_path = os.path.join(os.path.dirname(__file__), "../../../../../../production-stack/src")
    if os.path.exists(production_stack_path):
        sys.path.insert(0, production_stack_path)
        from vllm_router.services.kv_transfer_monitor import get_kv_transfer_monitor
        MONITORING_AVAILABLE = True
    else:
        MONITORING_AVAILABLE = False
        logger.warning("Production-stack monitoring not available")
except ImportError:
    MONITORING_AVAILABLE = False
    logger.warning("Could not import monitoring system")


class MonitoredNixlConnector(NixlConnector):
    """
    NIXL Connector with KV cache transfer monitoring.
    
    This wrapper adds monitoring capabilities to track transfer times
    and expose them as Prometheus metrics.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._monitor = get_kv_transfer_monitor() if MONITORING_AVAILABLE else None
        self._instance_id = kwargs.get('instance_id', 'unknown')
        
    def _generate_transfer_id(self) -> str:
        """Generate a unique transfer ID"""
        import uuid
        return f"nixl_{uuid.uuid4().hex[:8]}"
    
    def write(self, uid: str, data: bytes, **kwargs) -> str:
        """
        Write data with monitoring.
        
        This method wraps the original write method to add transfer monitoring.
        """
        if not self._monitor:
            return super().write(uid, data, **kwargs)
        
        transfer_id = self._generate_transfer_id()
        transfer_type = "send"
        
        # Start monitoring
        self._monitor.start_transfer(
            transfer_id=transfer_id,
            transfer_type=transfer_type,
            size_bytes=len(data),
            source_instance=self._instance_id,
            target_instance=getattr(self, 'peer_name', 'unknown')
        )
        
        try:
            # Perform the actual write operation
            result = super().write(uid, data, **kwargs)
            
            # End monitoring
            self._monitor.end_transfer(transfer_id)
            
            return result
            
        except Exception as e:
            # End monitoring even if transfer fails
            self._monitor.end_transfer(transfer_id)
            raise e
    
    def read(self, uid: str, **kwargs) -> bytes:
        """
        Read data with monitoring.
        
        This method wraps the original read method to add transfer monitoring.
        """
        if not self._monitor:
            return super().read(uid, **kwargs)
        
        transfer_id = self._generate_transfer_id()
        transfer_type = "receive"
        
        # Start monitoring
        self._monitor.start_transfer(
            transfer_id=transfer_id,
            transfer_type=transfer_type,
            size_bytes=0,  # We don't know the size beforehand
            source_instance=getattr(self, 'peer_name', 'unknown'),
            target_instance=self._instance_id
        )
        
        try:
            # Perform the actual read operation
            result = super().read(uid, **kwargs)
            
            # Update size and end monitoring
            if hasattr(self._monitor, '_active_transfers'):
                with self._monitor._lock:
                    if transfer_id in self._monitor._active_transfers:
                        self._monitor._active_transfers[transfer_id].size_bytes = len(result)
            
            self._monitor.end_transfer(transfer_id)
            
            return result
            
        except Exception as e:
            # End monitoring even if transfer fails
            self._monitor.end_transfer(transfer_id)
            raise e


class MonitoredNixlConnectorV2(NixlConnectorV2):
    """
    NIXL Connector V2 with KV cache transfer monitoring.
    
    This wrapper adds monitoring capabilities to track transfer times
    and expose them as Prometheus metrics.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._monitor = get_kv_transfer_monitor() if MONITORING_AVAILABLE else None
        self._instance_id = kwargs.get('instance_id', 'unknown')
        
    def _generate_transfer_id(self) -> str:
        """Generate a unique transfer ID"""
        import uuid
        return f"nixl_v2_{uuid.uuid4().hex[:8]}"
    
    def write(self, uid: str, data: bytes, **kwargs) -> str:
        """
        Write data with monitoring.
        
        This method wraps the original write method to add transfer monitoring.
        """
        if not self._monitor:
            return super().write(uid, data, **kwargs)
        
        transfer_id = self._generate_transfer_id()
        transfer_type = "send"
        
        # Start monitoring
        self._monitor.start_transfer(
            transfer_id=transfer_id,
            transfer_type=transfer_type,
            size_bytes=len(data),
            source_instance=self._instance_id,
            target_instance=getattr(self, 'peer_name', 'unknown')
        )
        
        try:
            # Perform the actual write operation
            result = super().write(uid, data, **kwargs)
            
            # End monitoring
            self._monitor.end_transfer(transfer_id)
            
            return result
            
        except Exception as e:
            # End monitoring even if transfer fails
            self._monitor.end_transfer(transfer_id)
            raise e
    
    def read(self, uid: str, **kwargs) -> bytes:
        """
        Read data with monitoring.
        
        This method wraps the original read method to add transfer monitoring.
        """
        if not self._monitor:
            return super().read(uid, **kwargs)
        
        transfer_id = self._generate_transfer_id()
        transfer_type = "receive"
        
        # Start monitoring
        self._monitor.start_transfer(
            transfer_id=transfer_id,
            transfer_type=transfer_type,
            size_bytes=0,  # We don't know the size beforehand
            source_instance=getattr(self, 'peer_name', 'unknown'),
            target_instance=self._instance_id
        )
        
        try:
            # Perform the actual read operation
            result = super().read(uid, **kwargs)
            
            # Update size and end monitoring
            if hasattr(self._monitor, '_active_transfers'):
                with self._monitor._lock:
                    if transfer_id in self._monitor._active_transfers:
                        self._monitor._active_transfers[transfer_id].size_bytes = len(result)
            
            self._monitor.end_transfer(transfer_id)
            
            return result
            
        except Exception as e:
            # End monitoring even if transfer fails
            self._monitor.end_transfer(transfer_id)
            raise e

