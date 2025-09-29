import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Input } from '../ui/input';
import { Button } from '../ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs';

const ConfigForm = ({ onConfigChange, initialConfig = {} }) => {
  const [config, setConfig] = useState({
    schedule_config: {
      weekdays: initialConfig.schedule_config?.weekdays || ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
      time_slots: initialConfig.schedule_config?.time_slots || [
        "09:00-10:00", "10:00-11:00", "11:00-12:00", 
        "12:00-13:00", "14:00-15:00", "15:00-16:00"
      ],
      lunch_break: initialConfig.schedule_config?.lunch_break || "12:00-13:00",
      max_daily_slots: initialConfig.schedule_config?.max_daily_slots || 6,
      max_weekly_slots: initialConfig.schedule_config?.max_weekly_slots || 30
    },
    infrastructure_config: {
      default_classroom_count: initialConfig.infrastructure_config?.default_classroom_count || 5,
      default_lab_count: initialConfig.infrastructure_config?.default_lab_count || 2,
      default_theory_room_count: initialConfig.infrastructure_config?.default_theory_room_count || 3,
      classroom_capacity: {
        theory: initialConfig.infrastructure_config?.classroom_capacity?.theory || 50,
        lab: initialConfig.infrastructure_config?.classroom_capacity?.lab || 30
      }
    },
    training_config: {
      learning_rate: initialConfig.training_config?.learning_rate || 3e-4,
      batch_size: initialConfig.training_config?.batch_size || 64,
      n_steps: initialConfig.training_config?.n_steps || 1024,
      n_epochs: initialConfig.training_config?.n_epochs || 4,
      total_timesteps: initialConfig.training_config?.total_timesteps || 500000,
      use_enhanced_rewards: initialConfig.training_config?.use_enhanced_rewards || true,
      use_curriculum_learning: initialConfig.training_config?.use_curriculum_learning || true
    }
  });

  const handleConfigChange = (section, field, value) => {
    const newConfig = {
      ...config,
      [section]: {
        ...config[section],
        [field]: value
      }
    };
    setConfig(newConfig);
    onConfigChange(newConfig);
  };

  const handleNestedConfigChange = (section, parentField, childField, value) => {
    const newConfig = {
      ...config,
      [section]: {
        ...config[section],
        [parentField]: {
          ...config[section][parentField],
          [childField]: value
        }
      }
    };
    setConfig(newConfig);
    onConfigChange(newConfig);
  };

  const resetToDefaults = () => {
    const defaultConfig = {
      schedule_config: {
        weekdays: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
        time_slots: [
          "09:00-10:00", "10:00-11:00", "11:00-12:00", 
          "12:00-13:00", "14:00-15:00", "15:00-16:00"
        ],
        lunch_break: "12:00-13:00",
        max_daily_slots: 6,
        max_weekly_slots: 30
      },
      infrastructure_config: {
        default_classroom_count: 5,
        default_lab_count: 2,
        default_theory_room_count: 3,
        classroom_capacity: {
          theory: 50,
          lab: 30
        }
      },
      training_config: {
        learning_rate: 3e-4,
        batch_size: 64,
        n_steps: 1024,
        n_epochs: 4,
        total_timesteps: 500000,
        use_enhanced_rewards: true,
        use_curriculum_learning: true
      }
    };
    setConfig(defaultConfig);
    onConfigChange(defaultConfig);
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex justify-between items-center">
          Advanced Configuration
          <Button variant="outline" onClick={resetToDefaults}>
            Reset to Defaults
          </Button>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="schedule" className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="schedule">Schedule</TabsTrigger>
            <TabsTrigger value="infrastructure">Infrastructure</TabsTrigger>
            <TabsTrigger value="training">Training</TabsTrigger>
          </TabsList>
          
          <TabsContent value="schedule" className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="text-sm font-medium">Max Daily Slots</label>
                <Input
                  type="number"
                  value={config.schedule_config.max_daily_slots}
                  onChange={(e) => handleConfigChange('schedule_config', 'max_daily_slots', parseInt(e.target.value))}
                  min="1"
                  max="12"
                />
              </div>
              <div>
                <label className="text-sm font-medium">Max Weekly Slots</label>
                <Input
                  type="number"
                  value={config.schedule_config.max_weekly_slots}
                  onChange={(e) => handleConfigChange('schedule_config', 'max_weekly_slots', parseInt(e.target.value))}
                  min="1"
                  max="60"
                />
              </div>
            </div>
            <div>
              <label className="text-sm font-medium">Lunch Break</label>
              <Input
                value={config.schedule_config.lunch_break}
                onChange={(e) => handleConfigChange('schedule_config', 'lunch_break', e.target.value)}
                placeholder="12:00-13:00"
              />
            </div>
          </TabsContent>
          
          <TabsContent value="infrastructure" className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="text-sm font-medium">Default Classroom Count</label>
                <Input
                  type="number"
                  value={config.infrastructure_config.default_classroom_count}
                  onChange={(e) => handleConfigChange('infrastructure_config', 'default_classroom_count', parseInt(e.target.value))}
                  min="1"
                  max="20"
                />
              </div>
              <div>
                <label className="text-sm font-medium">Default Lab Count</label>
                <Input
                  type="number"
                  value={config.infrastructure_config.default_lab_count}
                  onChange={(e) => handleConfigChange('infrastructure_config', 'default_lab_count', parseInt(e.target.value))}
                  min="0"
                  max="10"
                />
              </div>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="text-sm font-medium">Theory Room Capacity</label>
                <Input
                  type="number"
                  value={config.infrastructure_config.classroom_capacity.theory}
                  onChange={(e) => handleNestedConfigChange('infrastructure_config', 'classroom_capacity', 'theory', parseInt(e.target.value))}
                  min="10"
                  max="200"
                />
              </div>
              <div>
                <label className="text-sm font-medium">Lab Capacity</label>
                <Input
                  type="number"
                  value={config.infrastructure_config.classroom_capacity.lab}
                  onChange={(e) => handleNestedConfigChange('infrastructure_config', 'classroom_capacity', 'lab', parseInt(e.target.value))}
                  min="10"
                  max="100"
                />
              </div>
            </div>
          </TabsContent>
          
          <TabsContent value="training" className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="text-sm font-medium">Learning Rate</label>
                <Input
                  type="number"
                  step="0.0001"
                  value={config.training_config.learning_rate}
                  onChange={(e) => handleConfigChange('training_config', 'learning_rate', parseFloat(e.target.value))}
                  min="0.0001"
                  max="0.01"
                />
              </div>
              <div>
                <label className="text-sm font-medium">Batch Size</label>
                <Input
                  type="number"
                  value={config.training_config.batch_size}
                  onChange={(e) => handleConfigChange('training_config', 'batch_size', parseInt(e.target.value))}
                  min="1"
                  max="512"
                />
              </div>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="text-sm font-medium">N Steps</label>
                <Input
                  type="number"
                  value={config.training_config.n_steps}
                  onChange={(e) => handleConfigChange('training_config', 'n_steps', parseInt(e.target.value))}
                  min="64"
                  max="4096"
                />
              </div>
              <div>
                <label className="text-sm font-medium">N Epochs</label>
                <Input
                  type="number"
                  value={config.training_config.n_epochs}
                  onChange={(e) => handleConfigChange('training_config', 'n_epochs', parseInt(e.target.value))}
                  min="1"
                  max="20"
                />
              </div>
            </div>
            <div>
              <label className="text-sm font-medium">Total Timesteps</label>
              <Input
                type="number"
                value={config.training_config.total_timesteps}
                onChange={(e) => handleConfigChange('training_config', 'total_timesteps', parseInt(e.target.value))}
                min="10000"
                max="2000000"
              />
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  id="enhanced_rewards"
                  checked={config.training_config.use_enhanced_rewards}
                  onChange={(e) => handleConfigChange('training_config', 'use_enhanced_rewards', e.target.checked)}
                />
                <label htmlFor="enhanced_rewards" className="text-sm font-medium">
                  Use Enhanced Rewards
                </label>
              </div>
              <div className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  id="curriculum_learning"
                  checked={config.training_config.use_curriculum_learning}
                  onChange={(e) => handleConfigChange('training_config', 'use_curriculum_learning', e.target.checked)}
                />
                <label htmlFor="curriculum_learning" className="text-sm font-medium">
                  Use Curriculum Learning
                </label>
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
};

export default ConfigForm;
