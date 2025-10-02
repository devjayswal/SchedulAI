"""
GPU Utilities for SchedulAI

This module provides GPU detection, configuration, and fallback mechanisms
for the SchedulAI training system.
"""

import torch
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class GPUManager:
    """Manages GPU detection and configuration for training."""
    
    def __init__(self):
        self.device = self._detect_device()
        self.gpu_info = self._get_gpu_info()
        self._log_gpu_status()
    
    def _detect_device(self) -> torch.device:
        """Detect the best available device (GPU or CPU)."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def _get_gpu_info(self) -> Dict[str, Any]:
        """Get detailed GPU information."""
        gpu_info = {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "current_device": None,
            "device_name": None,
            "cuda_version": None,
            "pytorch_version": torch.__version__
        }
        
        if torch.cuda.is_available():
            gpu_info.update({
                "current_device": torch.cuda.current_device(),
                "device_name": torch.cuda.get_device_name(0),
                "cuda_version": torch.version.cuda,
                "memory_allocated": torch.cuda.memory_allocated(0),
                "memory_reserved": torch.cuda.memory_reserved(0),
                "max_memory": torch.cuda.max_memory_allocated(0)
            })
        
        return gpu_info
    
    def _log_gpu_status(self):
        """Log GPU status information."""
        if self.gpu_info["available"]:
            logger.info(f"GPU Available: {self.gpu_info['device_name']}")
            logger.info(f"CUDA Version: {self.gpu_info['cuda_version']}")
            logger.info(f"PyTorch Version: {self.gpu_info['pytorch_version']}")
            logger.info(f"Device Count: {self.gpu_info['device_count']}")
        else:
            logger.warning("GPU not available. Using CPU for training.")
            logger.info(f"PyTorch Version: {self.gpu_info['pytorch_version']}")
    
    def get_device_config(self) -> str:
        """Get device configuration string for PPO models."""
        return "auto"  # Let stable-baselines3 handle device selection
    
    def get_optimized_batch_size(self, base_batch_size: int = 64) -> int:
        """Get optimized batch size based on GPU availability."""
        if self.gpu_info["available"]:
            # Increase batch size for GPU training
            return min(base_batch_size * 2, 256)
        else:
            # Keep smaller batch size for CPU training
            return base_batch_size
    
    def get_optimized_n_envs(self, base_n_envs: int = 4) -> int:
        """Get optimized number of parallel environments based on GPU availability."""
        if self.gpu_info["available"]:
            # Increase parallel environments for GPU training
            return min(base_n_envs * 2, 16)
        else:
            # Keep smaller number for CPU training
            return base_n_envs
    
    def clear_gpu_memory(self):
        """Clear GPU memory cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU memory cache cleared")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage in MB."""
        if not self.gpu_info["available"]:
            return {"allocated": 0, "reserved": 0, "max": 0}
        
        return {
            "allocated": torch.cuda.memory_allocated(0) / 1024**2,  # MB
            "reserved": torch.cuda.memory_reserved(0) / 1024**2,    # MB
            "max": torch.cuda.max_memory_allocated(0) / 1024**2     # MB
        }
    
    def optimize_for_gpu(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize configuration parameters for GPU training."""
        optimized_config = config.copy()
        
        if self.gpu_info["available"]:
            # GPU optimizations - only include valid PPO parameters
            optimized_config.update({
                "batch_size": self.get_optimized_batch_size(config.get("batch_size", 64)),
                "device": "auto"
            })
            
            # Increase training parameters for GPU
            if "total_timesteps" in config:
                optimized_config["total_timesteps"] = int(config["total_timesteps"] * 1.5)
            
            logger.info("Configuration optimized for GPU training")
        else:
            # CPU optimizations
            optimized_config.update({
                "device": "cpu"
            })
            
            logger.info("Configuration optimized for CPU training")
        
        return optimized_config

# Global GPU manager instance
gpu_manager = GPUManager()

def get_gpu_manager() -> GPUManager:
    """Get the global GPU manager instance."""
    return gpu_manager

def is_gpu_available() -> bool:
    """Check if GPU is available."""
    return gpu_manager.gpu_info["available"]

def get_device() -> torch.device:
    """Get the current device."""
    return gpu_manager.device

def get_device_config() -> str:
    """Get device configuration string."""
    return gpu_manager.get_device_config()

def optimize_config_for_gpu(config: Dict[str, Any]) -> Dict[str, Any]:
    """Optimize configuration for GPU training."""
    return gpu_manager.optimize_for_gpu(config)
