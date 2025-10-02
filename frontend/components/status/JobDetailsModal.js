import React, { useState, useEffect } from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '../ui/dialog';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { Progress } from '../ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs';
import { Alert, AlertDescription } from '../ui/alert';
import { 
  Clock, 
  CheckCircle, 
  XCircle, 
  Loader2, 
  Calendar,
  Timer,
  Activity,
  FileText,
  RefreshCw,
  AlertCircle
} from 'lucide-react';
import { statusApi, apiUtils } from '@/lib/api';

const JobDetailsModal = ({ jobId, isOpen, onClose }) => {
  const [jobDetails, setJobDetails] = useState(null);
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchJobDetails = async () => {
    if (!jobId) return;
    
    setLoading(true);
    setError(null);
    
    try {
      // Fetch job details
      const details = await statusApi.getJobStatus(jobId);
      setJobDetails(details);
      
      // Fetch logs
      const logsData = await statusApi.getJobLogs(jobId, 100);
      setLogs(logsData.logs || []);
    } catch (err) {
      setError(apiUtils.handleError(err, 'Failed to fetch job details'));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (isOpen && jobId) {
      fetchJobDetails();
    }
  }, [isOpen, jobId]);

  const getStatusIcon = (status) => {
    switch (status) {
      case 'pending':
        return <Clock className="h-5 w-5 text-yellow-500" />;
      case 'running':
        return <Loader2 className="h-5 w-5 text-blue-500 animate-spin" />;
      case 'completed':
        return <CheckCircle className="h-5 w-5 text-green-500" />;
      case 'failed':
        return <XCircle className="h-5 w-5 text-red-500" />;
      default:
        return <Clock className="h-5 w-5 text-gray-500" />;
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

  if (!isOpen) return null;

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-4xl max-h-[80vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            {jobDetails && getStatusIcon(jobDetails.status)}
            Job Details: {jobId?.substring(0, 12)}...
          </DialogTitle>
        </DialogHeader>

        {loading && (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="h-8 w-8 animate-spin" />
            <span className="ml-2">Loading job details...</span>
          </div>
        )}

        {error && (
          <div className="bg-red-50 border border-red-200 rounded-md p-4">
            <p className="text-red-800">Error: {error}</p>
            <Button 
              variant="outline" 
              size="sm" 
              onClick={fetchJobDetails}
              className="mt-2"
            >
              <RefreshCw className="h-4 w-4 mr-2" />
              Retry
            </Button>
          </div>
        )}

        {jobDetails && !loading && (
          <Tabs defaultValue="overview" className="w-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="overview">Overview</TabsTrigger>
              <TabsTrigger value="progress">Progress</TabsTrigger>
              <TabsTrigger value="logs">Logs</TabsTrigger>
            </TabsList>

            <TabsContent value="overview" className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <h3 className="font-medium">Status</h3>
                  <Badge className={getStatusColor(jobDetails.status)}>
                    {jobDetails.status.toUpperCase()}
                  </Badge>
                </div>
                
                <div className="space-y-2">
                  <h3 className="font-medium">Job Type</h3>
                  <p className="text-sm text-gray-600">
                    {jobDetails.metadata?.type || 'N/A'}
                  </p>
                </div>
              </div>

              {jobDetails.status === 'completed' && (
                <div className="space-y-2">
                  <h3 className="font-medium">Actions</h3>
                  <Button 
                    onClick={() => {
                      window.open(`/view?job_id=${jobId}`, '_blank');
                    }}
                    className="w-full"
                  >
                    <Calendar className="h-4 w-4 mr-2" />
                    View Generated Timetable
                  </Button>
                </div>
              )}

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <h3 className="font-medium">Created At</h3>
                  <p className="text-sm text-gray-600">
                    {formatDate(jobDetails.metadata?.created_at)}
                  </p>
                </div>
                
                <div className="space-y-2">
                  <h3 className="font-medium">Elapsed Time</h3>
                  <p className="text-sm text-gray-600">
                    {formatElapsedTime(jobDetails.elapsed_time)}
                  </p>
                </div>
              </div>

              {jobDetails.metadata?.data && (
                <div className="space-y-2">
                  <h3 className="font-medium">Configuration</h3>
                  <div className="bg-gray-50 rounded-md p-3">
                    <pre className="text-xs text-gray-600 overflow-x-auto">
                      {JSON.stringify(jobDetails.metadata.data, null, 2)}
                    </pre>
                  </div>
                </div>
              )}
            </TabsContent>

            <TabsContent value="progress" className="space-y-4">
              {jobDetails.progress ? (
                <div className="space-y-4">
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="font-medium">Current Phase</span>
                      <span className="text-sm text-gray-600">
                        {jobDetails.progress.current_phase || 'N/A'}
                      </span>
                    </div>
                    <Progress 
                      value={jobDetails.progress.percentage || 0} 
                      className="h-3"
                    />
                    <div className="text-center text-sm text-gray-600">
                      {Math.round(jobDetails.progress.percentage || 0)}%
                    </div>
                  </div>

                  {jobDetails.progress.current_step && jobDetails.progress.total_steps && (
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <h4 className="font-medium">Current Step</h4>
                        <p className="text-2xl font-bold text-blue-600">
                          {jobDetails.progress.current_step}
                        </p>
                      </div>
                      <div>
                        <h4 className="font-medium">Total Steps</h4>
                        <p className="text-2xl font-bold text-gray-600">
                          {jobDetails.progress.total_steps}
                        </p>
                      </div>
                    </div>
                  )}

                  {jobDetails.progress.estimated_completion && (
                    <div className="bg-blue-50 border border-blue-200 rounded-md p-3">
                      <h4 className="font-medium text-blue-800">Estimated Completion</h4>
                      <p className="text-blue-600">
                        {formatDate(jobDetails.progress.estimated_completion)}
                      </p>
                    </div>
                  )}
                </div>
              ) : (
                <p className="text-gray-500 text-center py-8">No progress information available</p>
              )}
            </TabsContent>

            <TabsContent value="logs" className="space-y-4">
              <div className="flex justify-between items-center">
                <h3 className="font-medium">Recent Logs</h3>
                <Button 
                  variant="outline" 
                  size="sm" 
                  onClick={fetchJobDetails}
                >
                  <RefreshCw className="h-4 w-4 mr-2" />
                  Refresh
                </Button>
              </div>
              
              <div className="bg-gray-900 text-green-400 rounded-md p-4 max-h-96 overflow-y-auto font-mono text-sm">
                {logs.length > 0 ? (
                  logs.map((log, index) => (
                    <div key={index} className="mb-1">
                      {log}
                    </div>
                  ))
                ) : (
                  <p className="text-gray-500">No logs available</p>
                )}
              </div>
            </TabsContent>
          </Tabs>
        )}
      </DialogContent>
    </Dialog>
  );
};

export default JobDetailsModal;
