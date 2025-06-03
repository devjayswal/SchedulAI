# Time Table Scheduler

## Overview
The Time Table Scheduler is a web-based application designed to automate the creation and management of timetables for educational institutions. It supports scheduling for multiple branches, semesters, faculties, and classrooms while adhering to various constraints.

## Features
- **Timetable Generation**: Automatically generate timetables for students, faculties, and classrooms.
- **Editable Timetables**: Modify generated timetables directly in the application.
- **Export/Import**: Export timetables in CSV or Excel format and import data for scheduling.
- **Constraint Handling**: Supports hard and soft constraints for optimal scheduling.
- **User-Friendly Interface**: Intuitive UI with color-coded elements for better usability.

## Project Structure
cuda 12.9 




## Installation

### Prerequisites
- Node.js and npm
- Python 3.9+
- MongoDB
- Virtual Environment (optional)

### Frontend Setup
### Frontend Setup
1. Navigate to the `frontend` directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```
## API Endpoints

### Timetable Management:
- POST `/timetable/`: Create a new timetable.
- GET `/timetable/{id}`: Retrieve a timetable by ID.
- PUT `/timetable/{id}`: Update a timetable.
- DELETE `/timetable/{id}`: Delete a timetable.

### Faculty Management:
- POST `/faculty/`: Add a new faculty.
- GET `/faculty/`: List all faculties.

### Classroom Management:
- POST `/classroom/`: Add a new classroom.
- GET `/classroom/`: List all classrooms.

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Start the backend server:
   ```bash
   python main.py
   ```

## Usage
- Access the frontend at http://localhost:3000
- The backend API is available at http://localhost:8000
Timetable Management:
POST /timetable/: Create a new timetable.
GET /timetable/{id}: Retrieve a timetable by ID.
PUT /timetable/{id}: Update a timetable.
DELETE /timetable/{id}: Delete a timetable.
Faculty Management:
POST /faculty/: Add a new faculty.
GET /faculty/: List all faculties.
Classroom Management:
POST /classroom/: Add a new classroom.
GET /classroom/: List all classrooms.
Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

License
This project is licensed under the MIT License.

