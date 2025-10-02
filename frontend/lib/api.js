// API utility functions for frontend-backend communication
import { API_CONFIG, ENDPOINTS } from '@/config/api';

const API_BASE_URL = API_CONFIG.BASE_URL;

class ApiError extends Error {
  constructor(message, status, response) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.response = response;
  }
}

// Generic API request function
async function apiRequest(endpoint, options = {}) {
  const url = `${API_BASE_URL}${endpoint}`;
  const config = {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
    ...options,
  };

  try {
    const response = await fetch(url, config);
    
    if (!response.ok) {
      let errorMessage = `HTTP error! status: ${response.status}`;
      try {
        const errorData = await response.json();
        errorMessage = errorData.detail || errorData.message || errorMessage;
      } catch (e) {
        // If response is not JSON, use status text
        errorMessage = response.statusText || errorMessage;
      }
      throw new ApiError(errorMessage, response.status, response);
    }

    // Handle empty responses
    const contentType = response.headers.get('content-type');
    if (contentType && contentType.includes('application/json')) {
      return await response.json();
    } else {
      return await response.text();
    }
  } catch (error) {
    if (error instanceof ApiError) {
      throw error;
    }
    throw new ApiError(`Network error: ${error.message}`, 0, null);
  }
}

// Timetable API functions
export const timetableApi = {
  // Create a new timetable
  async create(data) {
    return apiRequest(ENDPOINTS.TIMETABLE.CREATE, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  },

  // Create timetable with specific training method
  async createContinuous(data) {
    return apiRequest(ENDPOINTS.TIMETABLE.CONTINUOUS, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  },

  async createEnhanced(data) {
    return apiRequest(ENDPOINTS.TIMETABLE.ENHANCED, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  },

  async createLegacy(data) {
    return apiRequest(ENDPOINTS.TIMETABLE.LEGACY, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  },

  // Get all timetables
  async getAll() {
    return apiRequest(ENDPOINTS.TIMETABLE.GET_ALL);
  },

  // Get specific timetable
  async getById(id) {
    return apiRequest(ENDPOINTS.TIMETABLE.GET_BY_ID(id));
  },

  // Get complete timetable data
  async getData(id) {
    return apiRequest(ENDPOINTS.TIMETABLE.GET_DATA(id));
  },

  // Get generated timetable by job ID
  async getGeneratedByJobId(jobId) {
    return apiRequest(ENDPOINTS.TIMETABLE.GENERATED_BY_JOB(jobId));
  },

  // Update timetable
  async update(id, data) {
    return apiRequest(ENDPOINTS.TIMETABLE.UPDATE(id), {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  },

  // Delete timetable
  async delete(id) {
    return apiRequest(ENDPOINTS.TIMETABLE.DELETE(id), {
      method: 'DELETE',
    });
  },

  // Regenerate timetable
  async regenerate(id) {
    return apiRequest(ENDPOINTS.TIMETABLE.REGENERATE(id), {
      method: 'POST',
    });
  },

  // Export timetable
  async export(id, type = 'master') {
    return apiRequest(ENDPOINTS.TIMETABLE.EXPORT(id, type));
  },
};

// Status API functions
export const statusApi = {
  // Get all jobs
  async getAllJobs() {
    return apiRequest(ENDPOINTS.STATUS.JOBS);
  },

  // Get specific job status
  async getJobStatus(jobId) {
    return apiRequest(ENDPOINTS.STATUS.JOB(jobId));
  },

  // Get job progress
  async getJobProgress(jobId) {
    return apiRequest(ENDPOINTS.STATUS.JOB_PROGRESS(jobId));
  },

  // Get job logs
  async getJobLogs(jobId, limit = 50) {
    return apiRequest(`${ENDPOINTS.STATUS.JOB_LOGS(jobId)}?limit=${limit}`);
  },

  // Get active jobs
  async getActiveJobs() {
    return apiRequest(ENDPOINTS.STATUS.ACTIVE_JOBS);
  },

  // Get completed jobs
  async getCompletedJobs() {
    return apiRequest(ENDPOINTS.STATUS.COMPLETED_JOBS);
  },

  // Cleanup job
  async cleanupJob(jobId) {
    return apiRequest(ENDPOINTS.STATUS.JOB(jobId), {
      method: 'DELETE',
    });
  },

  // Health check
  async healthCheck() {
    return apiRequest(ENDPOINTS.STATUS.HEALTH);
  },
};

// Utility functions
export const apiUtils = {
  // Check if API is available
  async isApiAvailable() {
    try {
      await statusApi.healthCheck();
      return true;
    } catch (error) {
      return false;
    }
  },

  // Get API base URL
  getBaseUrl() {
    return API_BASE_URL;
  },

  // Format error message for display
  formatError(error) {
    if (error instanceof ApiError) {
      return error.message;
    }
    return error.message || 'An unexpected error occurred';
  },

  // Handle API errors with user-friendly messages
  handleError(error, fallbackMessage = 'Something went wrong') {
    console.error('API Error:', error);
    
    if (error instanceof ApiError) {
      switch (error.status) {
        case 404:
          return 'The requested resource was not found';
        case 400:
          return 'Invalid request. Please check your input';
        case 500:
          return 'Server error. Please try again later';
        case 0:
          return 'Network error. Please check your connection';
        default:
          return error.message || fallbackMessage;
      }
    }
    
    return fallbackMessage;
  },
};

export default {
  timetableApi,
  statusApi,
  apiUtils,
  ApiError,
};
