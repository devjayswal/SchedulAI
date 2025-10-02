# Frontend-Backend Integration

This document describes the integration between the frontend and backend of the SchedulAI application.

## Overview

The frontend has been updated to use real API calls instead of sample data. All pages now communicate with the backend through a centralized API utility.

## API Integration

### API Utility (`lib/api.js`)

The API utility provides:
- Centralized API configuration
- Error handling and formatting
- Type-safe API calls
- Retry logic and timeout handling

### Configuration (`config/api.js`)

- Centralized endpoint definitions
- Environment-based configuration
- API timeout and retry settings

## Updated Pages

### 1. HomePage (`components/HomePage.js`)
- **Before**: Used hardcoded sample data for timetable history
- **After**: Fetches real timetable data from `/timetable/` endpoint
- **Features**:
  - Real-time timetable list
  - Delete functionality
  - API availability check
  - Error handling with retry

### 2. Create Page (`app/create/page.js`)
- **Before**: Used mock `fetchTimetable` function
- **After**: Uses real API calls for fetching and updating timetables
- **Features**:
  - Load existing timetable data for editing
  - Submit to real API endpoints
  - Proper error handling

### 3. View Page (`app/view/page.js`)
- **Before**: Used sample data as fallback
- **After**: Fetches real timetable data and generated timetables
- **Features**:
  - Real timetable data display
  - Generated timetable viewing by job ID
  - Regenerate functionality

### 4. Status Dashboard (`components/status/StatusDashboard.js`)
- **Before**: Already used API calls but with hardcoded URLs
- **After**: Uses centralized API utility
- **Features**:
  - Consistent error handling
  - Better user feedback

### 5. Job Details Modal (`components/status/JobDetailsModal.js`)
- **Before**: Used hardcoded API URLs
- **After**: Uses centralized API utility
- **Features**:
  - Consistent error handling
  - Better user experience

## API Endpoints Used

### Timetable Endpoints
- `POST /timetable/` - Create new timetable
- `GET /timetable/` - Get all timetables
- `GET /timetable/{id}` - Get specific timetable
- `GET /timetable/{id}/data` - Get complete timetable data
- `PUT /timetable/{id}` - Update timetable
- `DELETE /timetable/{id}` - Delete timetable
- `POST /timetable/{id}/regenerate` - Regenerate timetable
- `GET /timetable/generated/{job_id}` - Get generated timetable by job ID

### Status Endpoints
- `GET /status/jobs` - Get all jobs
- `GET /status/job/{job_id}` - Get specific job status
- `GET /status/job/{job_id}/logs` - Get job logs
- `DELETE /status/job/{job_id}` - Cleanup job
- `GET /status/health` - Health check

## Error Handling

The integration includes comprehensive error handling:
- Network errors
- API errors (4xx, 5xx)
- Timeout handling
- User-friendly error messages
- Retry mechanisms

## Environment Configuration

Set the following environment variable:
```bash
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
```

## Testing the Integration

1. Start the backend server:
   ```bash
   cd Scheduler
   python main.py
   ```

2. Start the frontend development server:
   ```bash
   cd frontend
   npm run dev
   ```

3. Test the integration:
   - Create a new timetable
   - View timetable history
   - Check job status
   - View generated timetables

## Benefits

1. **Real Data**: All pages now use real backend data
2. **Consistency**: Centralized API handling
3. **Error Handling**: Better user experience with proper error messages
4. **Maintainability**: Easy to update API endpoints
5. **Type Safety**: Better development experience
6. **Performance**: Optimized API calls with retry logic
