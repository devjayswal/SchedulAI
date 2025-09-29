"""
Configuration Processor for Dynamic Parameters

This module processes dynamic configuration parameters from frontend requests
and applies them to the training environment with reasonable defaults.
"""

from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

class ConfigProcessor:
    """Processes and validates dynamic configuration parameters."""
    
    def __init__(self):
        self.defaults = {
            "schedule_config": {
                "weekdays": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
                "time_slots": [
                    "09:00-10:00", "10:00-11:00", "11:00-12:00", 
                    "12:00-13:00", "14:00-15:00", "15:00-16:00"
                ],
                "lunch_break": "12:00-13:00",
                "max_daily_slots": 6,
                "max_weekly_slots": 30
            },
            "infrastructure_config": {
                "default_classroom_count": 5,
                "default_lab_count": 2,
                "default_theory_room_count": 3,
                "classroom_capacity": {"theory": 50, "lab": 30}
            },
            "training_config": {
                "learning_rate": 3e-4,
                "batch_size": 64,
                "n_steps": 1024,
                "n_epochs": 4,
                "total_timesteps": 500000,
                "use_enhanced_rewards": True,
                "use_curriculum_learning": True
            }
        }
    
    def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the incoming request and merge with defaults.
        
        Args:
            request_data: Raw request data from frontend
            
        Returns:
            Processed configuration with defaults applied
        """
        processed_config = {}
        
        # Process schedule configuration
        processed_config["schedule_config"] = self._process_schedule_config(
            request_data.get("schedule_config", {})
        )
        
        # Process infrastructure configuration
        processed_config["infrastructure_config"] = self._process_infrastructure_config(
            request_data.get("infrastructure_config", {}),
            request_data.get("classrooms", [])
        )
        
        # Process training configuration
        processed_config["training_config"] = self._process_training_config(
            request_data.get("training_config", {})
        )
        
        # Add core data
        processed_config.update({
            "branches": request_data.get("branches", []),
            "faculty": request_data.get("faculty", []),
            "classrooms": request_data.get("classrooms", [])
        })
        
        # Handle legacy fields for backward compatibility
        if "weekdays" in request_data:
            processed_config["schedule_config"]["weekdays"] = request_data["weekdays"]
        if "time_slots" in request_data:
            processed_config["schedule_config"]["time_slots"] = request_data["time_slots"]
        
        return processed_config
    
    def _process_schedule_config(self, schedule_config: Dict[str, Any]) -> Dict[str, Any]:
        """Process schedule configuration with validation."""
        config = self.defaults["schedule_config"].copy()
        config.update(schedule_config)
        
        # Validate time slots
        if len(config["time_slots"]) < 3:
            logger.warning("Too few time slots, using defaults")
            config["time_slots"] = self.defaults["schedule_config"]["time_slots"]
        
        # Validate weekdays
        if len(config["weekdays"]) < 3:
            logger.warning("Too few weekdays, using defaults")
            config["weekdays"] = self.defaults["schedule_config"]["weekdays"]
        
        return config
    
    def _process_infrastructure_config(self, infra_config: Dict[str, Any], classrooms: List[Dict]) -> Dict[str, Any]:
        """Process infrastructure configuration with classroom analysis."""
        config = self.defaults["infrastructure_config"].copy()
        config.update(infra_config)
        
        # Analyze actual classrooms if provided
        if classrooms:
            theory_count = sum(1 for c in classrooms if c.get("type") == "theory")
            lab_count = sum(1 for c in classrooms if c.get("type") == "lab")
            
            config["actual_classroom_count"] = len(classrooms)
            config["actual_theory_room_count"] = theory_count
            config["actual_lab_count"] = lab_count
            
            # Update defaults based on actual data
            if theory_count > 0:
                config["default_theory_room_count"] = theory_count
            if lab_count > 0:
                config["default_lab_count"] = lab_count
            if len(classrooms) > 0:
                config["default_classroom_count"] = len(classrooms)
        
        return config
    
    def _process_training_config(self, training_config: Dict[str, Any]) -> Dict[str, Any]:
        """Process training configuration with validation."""
        config = self.defaults["training_config"].copy()
        config.update(training_config)
        
        # Validate learning rate
        if config["learning_rate"] <= 0 or config["learning_rate"] > 1:
            logger.warning(f"Invalid learning rate {config['learning_rate']}, using default")
            config["learning_rate"] = self.defaults["training_config"]["learning_rate"]
        
        # Validate batch size
        if config["batch_size"] < 1 or config["batch_size"] > 1024:
            logger.warning(f"Invalid batch size {config['batch_size']}, using default")
            config["batch_size"] = self.defaults["training_config"]["batch_size"]
        
        return config
    
    def get_environment_config(self, processed_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract environment configuration for PPO training.
        
        Args:
            processed_config: Processed configuration from process_request
            
        Returns:
            Environment configuration for PPO
        """
        schedule_config = processed_config["schedule_config"]
        infra_config = processed_config["infrastructure_config"]
        
        # Calculate total courses
        total_courses = sum(len(branch.get("courses", [])) for branch in processed_config.get("branches", []))
        
        # Calculate time slots (excluding lunch break)
        time_slots = [ts for ts in schedule_config["time_slots"] if ts != schedule_config["lunch_break"]]
        
        return {
            "num_courses": total_courses,
            "num_slots": len(time_slots),
            "num_classrooms": infra_config.get("actual_classroom_count", infra_config["default_classroom_count"]),
            "n_envs": 4,
            "time_slots": time_slots,
            "weekdays": schedule_config["weekdays"],
            "lunch_break": schedule_config["lunch_break"]
        }
    
    def get_training_config(self, processed_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract training configuration for PPO.
        
        Args:
            processed_config: Processed configuration from process_request
            
        Returns:
            Training configuration for PPO
        """
        training_config = processed_config["training_config"]
        
        return {
            "learning_rate": training_config["learning_rate"],
            "batch_size": training_config["batch_size"],
            "n_steps": training_config["n_steps"],
            "n_epochs": training_config["n_epochs"],
            "total_timesteps": training_config["total_timesteps"],
            "use_enhanced_rewards": training_config["use_enhanced_rewards"],
            "use_curriculum_learning": training_config["use_curriculum_learning"]
        }

# Global instance
config_processor = ConfigProcessor()
