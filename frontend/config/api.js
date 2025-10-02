// API Configuration
export const API_CONFIG = {
  BASE_URL: process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000",
  TIMEOUT: 30000, // 30 seconds
  RETRY_ATTEMPTS: 3,
  RETRY_DELAY: 1000, // 1 second
};

// Environment detection
export const isDevelopment = process.env.NODE_ENV === 'development';
export const isProduction = process.env.NODE_ENV === 'production';

// API endpoints
export const ENDPOINTS = {
  TIMETABLE: {
    CREATE: '/timetable/',
    GET_ALL: '/timetable/',
    GET_BY_ID: (id) => `/timetable/${id}`,
    GET_DATA: (id) => `/timetable/${id}/data`,
    UPDATE: (id) => `/timetable/${id}`,
    DELETE: (id) => `/timetable/${id}`,
    REGENERATE: (id) => `/timetable/${id}/regenerate`,
    EXPORT: (id, type) => `/timetable/${id}/export/${type}`,
    GENERATED_BY_JOB: (jobId) => `/timetable/generated/${jobId}`,
    CONTINUOUS: '/timetable/continuous',
    ENHANCED: '/timetable/enhanced',
    LEGACY: '/timetable/legacy',
  },
  STATUS: {
    JOBS: '/status/jobs',
    JOB: (jobId) => `/status/job/${jobId}`,
    JOB_PROGRESS: (jobId) => `/status/job/${jobId}/progress`,
    JOB_LOGS: (jobId) => `/status/job/${jobId}/logs`,
    JOB_STREAM: (jobId) => `/status/job/${jobId}/stream`,
    ACTIVE_JOBS: '/status/jobs/active',
    COMPLETED_JOBS: '/status/jobs/completed',
    HEALTH: '/status/health',
  },
};

export default API_CONFIG;
