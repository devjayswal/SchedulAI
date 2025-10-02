"""
Model Management Utility for Timetable Scheduling
This module handles model persistence, loading, and knowledge retention across training sessions.
"""

import os
import torch
import json
import shutil
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import logging


class ModelManager:
    """Manages model persistence and knowledge retention."""
    
    def __init__(self, base_dir: str = "models"):
        self.base_dir = base_dir
        self.continuous_dir = os.path.join(base_dir, "continuous")
        self.enhanced_dir = os.path.join(base_dir, "enhanced")
        self.legacy_dir = os.path.join(base_dir, "legacy")
        self.checkpoints_dir = os.path.join(base_dir, "checkpoints")
        
        # Create directories
        for dir_path in [self.continuous_dir, self.enhanced_dir, self.legacy_dir, self.checkpoints_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger("ModelManager")
        self.logger.setLevel(logging.INFO)
    
    def get_model_info(self, model_path: str) -> Dict[str, Any]:
        """Get information about a saved model."""
        if not os.path.exists(model_path):
            return {"exists": False}
        
        try:
            # Get file stats
            stat = os.stat(model_path)
            file_size = stat.st_size
            modified_time = datetime.fromtimestamp(stat.st_mtime)
            
            # Try to load model metadata
            metadata = {}
            try:
                checkpoint = torch.load(model_path, map_location='cpu')
                if isinstance(checkpoint, dict):
                    metadata = {
                        "has_policy": "policy_state_dict" in checkpoint,
                        "has_optimizer": "optimizer_state_dict" in checkpoint,
                        "checkpoint_keys": list(checkpoint.keys())
                    }
            except:
                metadata = {"load_error": "Could not load checkpoint"}
            
            return {
                "exists": True,
                "file_size": file_size,
                "modified_time": modified_time.isoformat(),
                "metadata": metadata
            }
        except Exception as e:
            return {"exists": True, "error": str(e)}
    
    def save_model_with_metadata(self, model, model_path: str, metadata: Dict[str, Any] = None):
        """Save model with additional metadata."""
        try:
            # Prepare checkpoint data
            checkpoint = {
                'policy_state_dict': model.policy.state_dict(),
                'optimizer_state_dict': model.policy.optimizer.state_dict() if hasattr(model.policy, 'optimizer') else None,
                'metadata': metadata or {},
                'timestamp': datetime.now().isoformat(),
                'model_type': type(model).__name__
            }
            
            # Save model
            torch.save(checkpoint, model_path)
            
            # Save metadata separately
            metadata_path = model_path.replace('.pth', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(checkpoint['metadata'], f, indent=2)
            
            self.logger.info(f"Model saved to {model_path} with metadata")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            return False
    
    def load_model_with_metadata(self, model, model_path: str) -> Tuple[bool, Dict[str, Any]]:
        """Load model and return metadata."""
        try:
            if not os.path.exists(model_path):
                return False, {"error": "Model file not found"}
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Load model weights
            if 'policy_state_dict' in checkpoint:
                model.policy.load_state_dict(checkpoint['policy_state_dict'])
            
            if 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict'] is not None:
                if hasattr(model.policy, 'optimizer'):
                    model.policy.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Return metadata
            metadata = checkpoint.get('metadata', {})
            metadata['loaded_timestamp'] = checkpoint.get('timestamp', 'unknown')
            metadata['model_type'] = checkpoint.get('model_type', 'unknown')
            
            self.logger.info(f"Model loaded from {model_path}")
            return True, metadata
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False, {"error": str(e)}
    
    def create_checkpoint(self, model, job_id: str, checkpoint_type: str = "continuous") -> str:
        """Create a checkpoint of the current model."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join(
            self.checkpoints_dir, 
            f"{checkpoint_type}_checkpoint_{job_id}_{timestamp}.pth"
        )
        
        metadata = {
            "job_id": job_id,
            "checkpoint_type": checkpoint_type,
            "created_at": datetime.now().isoformat(),
            "episode_count": getattr(model, 'episode_count', 0),
            "training_steps": getattr(model, 'training_steps', 0)
        }
        
        if self.save_model_with_metadata(model, checkpoint_path, metadata):
            self.logger.info(f"Checkpoint created: {checkpoint_path}")
            return checkpoint_path
        else:
            return None
    
    def get_latest_model(self, model_type: str = "continuous") -> Optional[str]:
        """Get the path to the latest model of the specified type."""
        if model_type == "continuous":
            search_dir = self.continuous_dir
            pattern = "continuous_model"
        elif model_type == "enhanced":
            search_dir = self.enhanced_dir
            pattern = "enhanced"
        elif model_type == "legacy":
            search_dir = self.legacy_dir
            pattern = "legacy"
        else:
            return None
        
        if not os.path.exists(search_dir):
            return None
        
        # Find all model files
        model_files = []
        for file in os.listdir(search_dir):
            if file.endswith('.pth') and pattern in file:
                file_path = os.path.join(search_dir, file)
                stat = os.stat(file_path)
                model_files.append((file_path, stat.st_mtime))
        
        if not model_files:
            return None
        
        # Return the most recent one
        latest_file = max(model_files, key=lambda x: x[1])
        return latest_file[0]
    
    def cleanup_old_models(self, keep_count: int = 5, model_type: str = "continuous"):
        """Clean up old models, keeping only the most recent ones."""
        if model_type == "continuous":
            search_dir = self.continuous_dir
            pattern = "continuous_model"
        elif model_type == "enhanced":
            search_dir = self.enhanced_dir
            pattern = "enhanced"
        elif model_type == "legacy":
            search_dir = self.legacy_dir
            pattern = "legacy"
        else:
            return
        
        if not os.path.exists(search_dir):
            return
        
        # Find all model files
        model_files = []
        for file in os.listdir(search_dir):
            if file.endswith('.pth') and pattern in file:
                file_path = os.path.join(search_dir, file)
                stat = os.stat(file_path)
                model_files.append((file_path, stat.st_mtime))
        
        if len(model_files) <= keep_count:
            return
        
        # Sort by modification time (oldest first)
        model_files.sort(key=lambda x: x[1])
        
        # Remove oldest files
        files_to_remove = model_files[:-keep_count]
        for file_path, _ in files_to_remove:
            try:
                os.remove(file_path)
                # Also remove metadata file if it exists
                metadata_path = file_path.replace('.pth', '_metadata.json')
                if os.path.exists(metadata_path):
                    os.remove(metadata_path)
                self.logger.info(f"Removed old model: {file_path}")
            except Exception as e:
                self.logger.error(f"Failed to remove {file_path}: {e}")
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """Get statistics about all saved models."""
        stats = {
            "continuous": {"count": 0, "total_size": 0, "latest": None},
            "enhanced": {"count": 0, "total_size": 0, "latest": None},
            "legacy": {"count": 0, "total_size": 0, "latest": None},
            "checkpoints": {"count": 0, "total_size": 0}
        }
        
        for model_type in ["continuous", "enhanced", "legacy"]:
            if model_type == "continuous":
                search_dir = self.continuous_dir
                pattern = "continuous_model"
            elif model_type == "enhanced":
                search_dir = self.enhanced_dir
                pattern = "enhanced"
            else:  # legacy
                search_dir = self.legacy_dir
                pattern = "legacy"
            
            if os.path.exists(search_dir):
                for file in os.listdir(search_dir):
                    if file.endswith('.pth') and pattern in file:
                        file_path = os.path.join(search_dir, file)
                        stat = os.stat(file_path)
                        stats[model_type]["count"] += 1
                        stats[model_type]["total_size"] += stat.st_size
                        
                        if stats[model_type]["latest"] is None or stat.st_mtime > stats[model_type]["latest"]["mtime"]:
                            stats[model_type]["latest"] = {
                                "file": file,
                                "mtime": stat.st_mtime,
                                "size": stat.st_size
                            }
        
        # Checkpoints
        if os.path.exists(self.checkpoints_dir):
            for file in os.listdir(self.checkpoints_dir):
                if file.endswith('.pth'):
                    file_path = os.path.join(self.checkpoints_dir, file)
                    stat = os.stat(file_path)
                    stats["checkpoints"]["count"] += 1
                    stats["checkpoints"]["total_size"] += stat.st_size
        
        return stats
    
    def backup_models(self, backup_dir: str = "models_backup") -> bool:
        """Create a backup of all models."""
        try:
            if os.path.exists(backup_dir):
                shutil.rmtree(backup_dir)
            
            shutil.copytree(self.base_dir, backup_dir)
            self.logger.info(f"Models backed up to {backup_dir}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to backup models: {e}")
            return False


# Global model manager instance
model_manager = ModelManager()


def get_model_manager() -> ModelManager:
    """Get the global model manager instance."""
    return model_manager
