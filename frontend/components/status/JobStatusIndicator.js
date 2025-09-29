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
  ExternalLink,
  RefreshCw
} from 'lucide-react';

const JobStatusIndicator = ({ jobId, onViewDetails }) => {
  const [jobStatus, setJobStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [autoRefresh, setAutoRefresh] = useState(true);

  const fetchJobStatus = async () => {
    if (!jobId) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
      const response = await fetch(`${apiUrl}/status/job/${jobId}`);
      
      if (!response.ok) {
        throw new Error('Failed to fetch job status');
      }
      
      const data = await response.json();
      setJobStatus(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (jobId) {
      fetchJobStatus();
    }
  }, [jobId]);

  useEffect(() => {
    if (autoRefresh && jobId && jobStatus?.status === 'running') {
      const interval = setInterval(fetchJobStatus, 3000); // Refresh every 3 seconds for running jobs
      return () => clearInterval(interval);
    }
  }, [autoRefresh, jobId, jobStatus?.status]);

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

  if (!jobId) {
    return null;
  }

  if (loading && !jobStatus) {
    return (
      <Card className="w-full">
        <CardContent className="p-4">
          <div className="flex items-center justify-center">
            <Loader2 className="h-6 w-6 animate-spin mr-2" />
            <span>Loading job status...</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card className="w-full border-red-200">
        <CardContent className="p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center text-red-600">
              <XCircle className="h-5 w-5 mr-2" />
              <span>Error loading job status</span>
            </div>
            <Button variant="outline" size="sm" onClick={fetchJobStatus}>
              <RefreshCw className="h-4 w-4" />
            </Button>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!jobStatus) {
    return null;
  }

  return (
    <Card className="w-full">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg flex items-center gap-2">
            {getStatusIcon(jobStatus.status)}
            Job Status
          </CardTitle>
          <div className="flex items-center gap-2">
            <Badge className={getStatusColor(jobStatus.status)}>
              {jobStatus.status.toUpperCase()}
            </Badge>
            <Button
              variant="outline"
              size="sm"
              onClick={fetchJobStatus}
              disabled={loading}
            >
              <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
            </Button>
          </div>
        </div>
      </CardHeader>
      
      <CardContent className="space-y-4">
        {/* Progress Bar */}
        {jobStatus.progress && (
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="font-medium">{jobStatus.progress.current_phase || 'Processing'}</span>
              <span>{Math.round(jobStatus.progress.percentage || 0)}%</span>
            </div>
            <Progress 
              value={jobStatus.progress.percentage || 0} 
              className="h-2"
            />
            {jobStatus.progress.current_step && jobStatus.progress.total_steps && (
              <div className="text-xs text-gray-500">
                Step {jobStatus.progress.current_step} of {jobStatus.progress.total_steps}
              </div>
            )}
          </div>
        )}

        {/* Job Details */}
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="text-gray-600">Job ID:</span>
            <p className="font-mono text-xs">{jobId.substring(0, 12)}...</p>
          </div>
          <div>
            <span className="text-gray-600">Elapsed Time:</span>
            <p>{formatElapsedTime(jobStatus.elapsed_time)}</p>
          </div>
        </div>

        {/* Recent Activity */}
        {jobStatus.recent_logs && jobStatus.recent_logs.length > 0 && (
          <div className="space-y-2">
            <h4 className="text-sm font-medium text-gray-700">Recent Activity</h4>
            <div className="bg-gray-50 rounded-md p-3 max-h-24 overflow-y-auto">
              {jobStatus.recent_logs.slice(-2).map((log, index) => (
                <div key={index} className="text-xs text-gray-600 mb-1">
                  {log}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Estimated Completion */}
        {jobStatus.progress?.estimated_completion && jobStatus.status === 'running' && (
          <div className="text-sm text-blue-600 bg-blue-50 p-2 rounded-md">
            <strong>Estimated completion:</strong> {new Date(jobStatus.progress.estimated_completion).toLocaleString()}
          </div>
        )}

        {/* Action Buttons */}
        <div className="flex gap-2 pt-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => onViewDetails(jobId)}
            className="flex-1"
          >
            <Eye className="h-4 w-4 mr-2" />
            View Details
          </Button>
          
          <Button
            variant="outline"
            size="sm"
            onClick={() => window.open(`/status`, '_blank')}
          >
            <ExternalLink className="h-4 w-4 mr-2" />
            Full Dashboard
          </Button>
        </div>

        {/* Auto-refresh toggle */}
        {jobStatus.status === 'running' && (
          <div className="flex items-center justify-between text-xs text-gray-500 pt-2 border-t">
            <span>Auto-refresh: {autoRefresh ? 'ON' : 'OFF'}</span>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setAutoRefresh(!autoRefresh)}
              className="h-6 px-2 text-xs"
            >
              {autoRefresh ? 'Disable' : 'Enable'}
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default JobStatusIndicator;
