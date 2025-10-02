import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs';
import { Badge } from '../ui/badge';
import { Alert, AlertDescription } from '../ui/alert';
import { 
  RefreshCw, 
  Activity, 
  CheckCircle, 
  XCircle, 
  Clock, 
  Loader2,
  Trash2,
  Eye,
  AlertCircle
} from 'lucide-react';
import JobStatusCard from './JobStatusCard';
import JobDetailsModal from './JobDetailsModal';
import { statusApi, apiUtils } from '@/lib/api';

const StatusDashboard = () => {
  const [jobs, setJobs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedJobId, setSelectedJobId] = useState(null);
  const [showDetailsModal, setShowDetailsModal] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [refreshInterval, setRefreshInterval] = useState(null);

  const fetchJobs = async () => {
    try {
      const data = await statusApi.getAllJobs();
      setJobs(data.jobs || []);
      setError(null);
    } catch (err) {
      setError(apiUtils.handleError(err, 'Failed to fetch jobs'));
    } finally {
      setLoading(false);
    }
  };

  const handleRefresh = async (jobId = null) => {
    if (jobId) {
      // Refresh specific job
      try {
        const jobData = await statusApi.getJobStatus(jobId);
        setJobs(prevJobs => 
          prevJobs.map(job => 
            job.job_id === jobId ? jobData : job
          )
        );
      } catch (err) {
        console.error('Failed to refresh job:', err);
        setError(apiUtils.handleError(err, 'Failed to refresh job'));
      }
    } else {
      // Refresh all jobs
      await fetchJobs();
    }
  };

  const handleCleanup = async (jobId) => {
    try {
      await statusApi.cleanupJob(jobId);
      setJobs(prevJobs => prevJobs.filter(job => job.job_id !== jobId));
    } catch (err) {
      console.error('Failed to cleanup job:', err);
      setError(apiUtils.handleError(err, 'Failed to cleanup job'));
    }
  };

  const handleViewDetails = (jobId) => {
    setSelectedJobId(jobId);
    setShowDetailsModal(true);
  };

  const getStatusCounts = () => {
    const counts = {
      pending: 0,
      running: 0,
      completed: 0,
      failed: 0
    };
    
    jobs.forEach(job => {
      if (counts.hasOwnProperty(job.status)) {
        counts[job.status]++;
      }
    });
    
    return counts;
  };

  const getActiveJobs = () => {
    return jobs.filter(job => job.status === 'pending' || job.status === 'running');
  };

  const getCompletedJobs = () => {
    return jobs.filter(job => job.status === 'completed' || job.status === 'failed');
  };

  useEffect(() => {
    fetchJobs();
  }, []);

  useEffect(() => {
    if (autoRefresh) {
      const interval = setInterval(fetchJobs, 5000); // Refresh every 5 seconds
      setRefreshInterval(interval);
    } else {
      if (refreshInterval) {
        clearInterval(refreshInterval);
        setRefreshInterval(null);
      }
    }

    return () => {
      if (refreshInterval) {
        clearInterval(refreshInterval);
      }
    };
  }, [autoRefresh]);

  const statusCounts = getStatusCounts();
  const activeJobs = getActiveJobs();
  const completedJobs = getCompletedJobs();

  if (loading) {
    return (
      <div className="flex items-center justify-center py-8">
        <Loader2 className="h-8 w-8 animate-spin" />
        <span className="ml-2">Loading job status...</span>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">📊 Job Status Dashboard</h1>
          <p className="text-gray-600">Monitor your timetable generation jobs</p>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            onClick={() => setAutoRefresh(!autoRefresh)}
            className={autoRefresh ? 'bg-green-50 text-green-700' : ''}
          >
            <Activity className="h-4 w-4 mr-2" />
            Auto Refresh {autoRefresh ? 'ON' : 'OFF'}
          </Button>
          <Button variant="outline" onClick={fetchJobs}>
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
        </div>
      </div>

      {/* Error Message */}
      {error && (
        <Alert variant="destructive" className="mb-6">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            {error}
            <Button variant="outline" size="sm" onClick={fetchJobs} className="mt-2 ml-2">
              Retry
            </Button>
          </AlertDescription>
        </Alert>
      )}

      {/* Status Summary */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Pending</p>
                <p className="text-2xl font-bold text-yellow-600">{statusCounts.pending}</p>
              </div>
              <Clock className="h-8 w-8 text-yellow-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Running</p>
                <p className="text-2xl font-bold text-blue-600">{statusCounts.running}</p>
              </div>
              <Loader2 className="h-8 w-8 text-blue-500 animate-spin" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Completed</p>
                <p className="text-2xl font-bold text-green-600">{statusCounts.completed}</p>
              </div>
              <CheckCircle className="h-8 w-8 text-green-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Failed</p>
                <p className="text-2xl font-bold text-red-600">{statusCounts.failed}</p>
              </div>
              <XCircle className="h-8 w-8 text-red-500" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Jobs List */}
      <Tabs defaultValue="active" className="w-full">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="active">
            Active Jobs ({activeJobs.length})
          </TabsTrigger>
          <TabsTrigger value="completed">
            Completed Jobs ({completedJobs.length})
          </TabsTrigger>
        </TabsList>

        <TabsContent value="active" className="space-y-4">
          {activeJobs.length > 0 ? (
            <div className="grid gap-4">
              {activeJobs.map((job) => (
                <JobStatusCard
                  key={job.job_id}
                  job={job}
                  onViewDetails={handleViewDetails}
                  onCleanup={handleCleanup}
                  onRefresh={handleRefresh}
                />
              ))}
            </div>
          ) : (
            <Card>
              <CardContent className="p-8 text-center">
                <Activity className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">No Active Jobs</h3>
                <p className="text-gray-600">No jobs are currently running or pending.</p>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="completed" className="space-y-4">
          {completedJobs.length > 0 ? (
            <div className="grid gap-4">
              {completedJobs.map((job) => (
                <JobStatusCard
                  key={job.job_id}
                  job={job}
                  onViewDetails={handleViewDetails}
                  onCleanup={handleCleanup}
                  onRefresh={handleRefresh}
                />
              ))}
            </div>
          ) : (
            <Card>
              <CardContent className="p-8 text-center">
                <CheckCircle className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">No Completed Jobs</h3>
                <p className="text-gray-600">No jobs have been completed yet.</p>
              </CardContent>
            </Card>
          )}
        </TabsContent>
      </Tabs>

      {/* Job Details Modal */}
      <JobDetailsModal
        jobId={selectedJobId}
        isOpen={showDetailsModal}
        onClose={() => {
          setShowDetailsModal(false);
          setSelectedJobId(null);
        }}
      />
    </div>
  );
};

export default StatusDashboard;
