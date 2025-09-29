import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { Progress } from '../ui/progress';
import { 
  Clock, 
  CheckCircle, 
  XCircle, 
  Loader2, 
  Eye, 
  Trash2,
  RefreshCw,
  Calendar,
  Timer
} from 'lucide-react';

const JobStatusCard = ({ job, onViewDetails, onCleanup, onRefresh }) => {
  const [isRefreshing, setIsRefreshing] = useState(false);

  const getStatusIcon = (status) => {
    switch (status) {
      case 'pending':
        return <Clock className="h-4 w-4 text-yellow-500" />;
      case 'running':
        return <Loader2 className="h-4 w-4 text-blue-500 animate-spin" />;
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'failed':
        return <XCircle className="h-4 w-4 text-red-500" />;
      default:
        return <Clock className="h-4 w-4 text-gray-500" />;
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'pending':
        return 'bg-yellow-100 text-yellow-800';
      case 'running':
        return 'bg-blue-100 text-blue-800';
      case 'completed':
        return 'bg-green-100 text-green-800';
      case 'failed':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const formatElapsedTime = (seconds) => {
    if (!seconds) return 'N/A';
    
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    
    if (hours > 0) {
      return `${hours}h ${minutes}m ${secs}s`;
    } else if (minutes > 0) {
      return `${minutes}m ${secs}s`;
    } else {
      return `${secs}s`;
    }
  };

  const formatDate = (dateString) => {
    if (!dateString) return 'N/A';
    return new Date(dateString).toLocaleString();
  };

  const handleRefresh = async () => {
    setIsRefreshing(true);
    await onRefresh(job.job_id);
    setIsRefreshing(false);
  };

  return (
    <Card className="w-full">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg flex items-center gap-2">
            {getStatusIcon(job.status)}
            Job {job.job_id.substring(0, 8)}...
          </CardTitle>
          <div className="flex items-center gap-2">
            <Badge className={getStatusColor(job.status)}>
              {job.status.toUpperCase()}
            </Badge>
            <Button
              variant="outline"
              size="sm"
              onClick={handleRefresh}
              disabled={isRefreshing}
            >
              <RefreshCw className={`h-4 w-4 ${isRefreshing ? 'animate-spin' : ''}`} />
            </Button>
          </div>
        </div>
      </CardHeader>
      
      <CardContent className="space-y-4">
        {/* Progress Bar */}
        {job.progress && (
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="font-medium">{job.progress.current_phase || 'Processing'}</span>
              <span>{Math.round(job.progress.percentage || 0)}%</span>
            </div>
            <Progress 
              value={job.progress.percentage || 0} 
              className="h-2"
            />
            {job.progress.current_step && job.progress.total_steps && (
              <div className="text-xs text-gray-500">
                Step {job.progress.current_step} of {job.progress.total_steps}
              </div>
            )}
          </div>
        )}

        {/* Job Details */}
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div className="flex items-center gap-2">
            <Calendar className="h-4 w-4 text-gray-400" />
            <span className="text-gray-600">Created:</span>
            <span>{formatDate(job.metadata?.created_at)}</span>
          </div>
          
          <div className="flex items-center gap-2">
            <Timer className="h-4 w-4 text-gray-400" />
            <span className="text-gray-600">Elapsed:</span>
            <span>{formatElapsedTime(job.elapsed_time)}</span>
          </div>
        </div>

        {/* Recent Logs */}
        {job.recent_logs && job.recent_logs.length > 0 && (
          <div className="space-y-2">
            <h4 className="text-sm font-medium text-gray-700">Recent Activity</h4>
            <div className="bg-gray-50 rounded-md p-3 max-h-32 overflow-y-auto">
              {job.recent_logs.slice(-3).map((log, index) => (
                <div key={index} className="text-xs text-gray-600 mb-1">
                  {log}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Estimated Completion */}
        {job.progress?.estimated_completion && job.status === 'running' && (
          <div className="text-sm text-blue-600 bg-blue-50 p-2 rounded-md">
            <strong>Estimated completion:</strong> {formatDate(job.progress.estimated_completion)}
          </div>
        )}

        {/* Action Buttons */}
        <div className="flex gap-2 pt-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => onViewDetails(job.job_id)}
            className="flex-1"
          >
            <Eye className="h-4 w-4 mr-2" />
            View Details
          </Button>
          
          {(job.status === 'completed' || job.status === 'failed') && (
            <Button
              variant="outline"
              size="sm"
              onClick={() => onCleanup(job.job_id)}
              className="text-red-600 hover:text-red-700"
            >
              <Trash2 className="h-4 w-4" />
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  );
};

export default JobStatusCard;
