# Dynamic Configuration System

## Overview

The dynamic configuration system allows users to customize timetable generation parameters through the frontend interface. All parameters have reasonable defaults, so users only need to change what they want to modify.

## Key Features

✅ **Dynamic Parameters**: All configurable parameters can be set from frontend  
✅ **Reasonable Defaults**: Sensible defaults for all settings  
✅ **Backward Compatibility**: Works with existing request format  
✅ **Validation**: Invalid values are corrected to defaults  
✅ **Real-time Processing**: Configuration is processed on the backend  

## Configuration Sections

### 1. Schedule Configuration (`schedule_config`)

Controls the basic scheduling constraints:

```json
{
  "schedule_config": {
    "weekdays": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
    "time_slots": [
      "09:00-10:00", "10:00-11:00", "11:00-12:00", 
      "12:00-13:00", "14:00-15:00", "15:00-16:00"
    ],
    "lunch_break": "12:00-13:00",
    "max_daily_slots": 6,
    "max_weekly_slots": 30
  }
}
```

**Parameters:**
- `weekdays`: Days of the week (default: Mon-Fri)
- `time_slots`: Available time slots (default: 6 slots)
- `lunch_break`: Lunch break time (default: 12:00-13:00)
- `max_daily_slots`: Maximum slots per day (default: 6)
- `max_weekly_slots`: Maximum slots per week (default: 30)

### 2. Infrastructure Configuration (`infrastructure_config`)

Controls classroom and resource settings:

```json
{
  "infrastructure_config": {
    "default_classroom_count": 5,
    "default_lab_count": 2,
    "default_theory_room_count": 3,
    "classroom_capacity": {
      "theory": 50,
      "lab": 30
    }
  }
}
```

**Parameters:**
- `default_classroom_count`: Total classrooms (default: 5)
- `default_lab_count`: Number of labs (default: 2)
- `default_theory_room_count`: Number of theory rooms (default: 3)
- `classroom_capacity`: Capacity for each room type

### 3. Training Configuration (`training_config`)

Controls PPO training parameters:

```json
{
  "training_config": {
    "learning_rate": 3e-4,
    "batch_size": 64,
    "n_steps": 1024,
    "n_epochs": 4,
    "total_timesteps": 500000,
    "use_enhanced_rewards": true,
    "use_curriculum_learning": true
  }
}
```

**Parameters:**
- `learning_rate`: PPO learning rate (default: 3e-4)
- `batch_size`: Training batch size (default: 64)
- `n_steps`: Steps per update (default: 1024)
- `n_epochs`: Training epochs (default: 4)
- `total_timesteps`: Total training steps (default: 500000)
- `use_enhanced_rewards`: Enable enhanced reward system (default: true)
- `use_curriculum_learning`: Enable curriculum learning (default: true)

## Frontend Integration

### Configuration Form

The frontend includes an advanced configuration form with three tabs:

1. **Schedule Tab**: Time slots, weekdays, lunch break
2. **Infrastructure Tab**: Classroom counts, capacities
3. **Training Tab**: PPO parameters, learning options

### Usage

```jsx
import ConfigForm from '@/components/create/ConfigForm';

// In your component
const [dynamicConfig, setDynamicConfig] = useState({});

<ConfigForm 
  onConfigChange={setDynamicConfig}
  initialConfig={dynamicConfig}
/>
```

### Toggle Advanced Config

Users can toggle the advanced configuration panel:

```jsx
<Button onClick={() => setShowAdvancedConfig(!showAdvancedConfig)}>
  ⚙️ Advanced Config
</Button>
```

## Backend Processing

### Configuration Processor

The `ConfigProcessor` class handles all dynamic configuration:

```python
from utils.config_processor import config_processor

# Process request
processed_config = config_processor.process_request(request_data)

# Extract configurations
env_config = config_processor.get_environment_config(processed_config)
training_config = config_processor.get_training_config(processed_config)
```

### Request Flow

1. **Frontend** sends request with optional configuration
2. **Backend** processes configuration with defaults
3. **Validation** corrects invalid values
4. **Training** uses processed configuration

## Default Values

All parameters have sensible defaults:

| Parameter | Default Value | Range |
|-----------|---------------|-------|
| Learning Rate | 3e-4 | 0.0001 - 0.01 |
| Batch Size | 64 | 1 - 512 |
| N Steps | 1024 | 64 - 4096 |
| N Epochs | 4 | 1 - 20 |
| Classroom Count | 5 | 1 - 20 |
| Lab Count | 2 | 0 - 10 |
| Theory Room Count | 3 | 1 - 15 |

## Validation Rules

The system validates and corrects invalid values:

- **Learning Rate**: Must be between 0.0001 and 0.01
- **Batch Size**: Must be between 1 and 512
- **N Steps**: Must be between 64 and 4096
- **N Epochs**: Must be between 1 and 20
- **Time Slots**: Must have at least 3 slots
- **Weekdays**: Must have at least 3 days

## Backward Compatibility

The system maintains full backward compatibility:

```json
// Legacy format still works
{
  "weekdays": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
  "time_slots": ["09:00-10:00", "10:00-11:00", "11:00-12:00"],
  "branches": [...],
  "faculty": [...],
  "classrooms": [...]
}
```

## Example Requests

### Minimal Request (Uses All Defaults)

```json
{
  "branches": [
    {
      "branch_name": "CSE",
      "semester": 4,
      "courses": [
        {
          "subject_code": "CS201",
          "subject_name": "Data Structures",
          "subject_type": "theory",
          "credits": 3,
          "faculty_id": "F01"
        }
      ]
    }
  ],
  "faculty": [
    {"id": "F01", "name": "Dr. Smith", "email": "smith@institute.edu"}
  ],
  "classrooms": [
    {"id": "CR101", "type": "theory"}
  ]
}
```

### Full Configuration Request

```json
{
  "schedule_config": {
    "max_daily_slots": 8,
    "max_weekly_slots": 40
  },
  "infrastructure_config": {
    "default_classroom_count": 8,
    "default_lab_count": 3
  },
  "training_config": {
    "learning_rate": 5e-4,
    "batch_size": 128,
    "use_enhanced_rewards": true
  },
  "branches": [...],
  "faculty": [...],
  "classrooms": [...]
}
```

## Testing

Run the test suite to validate the configuration system:

```bash
cd Scheduler
python test_dynamic_config.py
```

The test suite validates:
- Configuration processing
- Default value application
- Validation and error correction
- Backward compatibility
- Environment and training config extraction

## Benefits

1. **Flexibility**: Users can customize any parameter
2. **Simplicity**: Defaults work out of the box
3. **Validation**: Invalid values are automatically corrected
4. **Compatibility**: Works with existing systems
5. **Performance**: Optimized parameters for better training

## Migration Guide

### For Existing Users

No changes required! The system is fully backward compatible.

### For New Implementations

1. Use the new configuration format for better control
2. Set only the parameters you want to customize
3. Let defaults handle the rest

### For Frontend Developers

1. Add the `ConfigForm` component to your create page
2. Handle the `onConfigChange` callback
3. Include the configuration in your API requests

The dynamic configuration system provides the perfect balance between flexibility and simplicity, allowing users to customize their timetable generation while maintaining sensible defaults for optimal performance.
