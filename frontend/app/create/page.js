"use client";
import { useState, useEffect } from "react";
import { useSearchParams } from "next/navigation";
import BranchForm from "@/components/create/BranchForm";
import CourseForm from "@/components/create/CourseForm";
import FacultyForm from "@/components/create/FacultyForm";
import ClassroomForm from "@/components/create/ClassroomForm";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert";
import { CheckCircle, AlertCircle, Loader2, Upload, FileText } from "lucide-react";

// Mock function to simulate fetching from backend
const fetchTimetable = async (id) => {
  console.log(`Fetching timetable for ID: ${id}`);
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve({
        branches: [
          {
            branch_name: "CSE",
            semester: 4,
            courses: []
          }
        ],
        courses: [
          {
            branch_name: "CSE",
            semester: 4,
            courses: [
              {
                subject_code: "CS201",
                subject_name: "Data Structures",
                subject_type: "theory",
                credits: 3,
                faculty_id: "F01"
              }
            ]
          }
        ],
        faculty: [
          {
            id: "F01",
            name: "Dr. Smith",
            email: "smith@institute.edu"
          }
        ],
        classrooms: [
          {
            id: "CR101",
            type: "theory"
          }
        ],
      });
    }, 1000);
  });
};

export default function CreatePage() {
  const searchParams = useSearchParams();
  const timetableId = searchParams.get("id"); // Get ID if passed

  // State for storing form data
  const [timetableData, setTimetableData] = useState({});
  const [branches, setBranches] = useState([]);
  const [courses, setCourses] = useState([]);
  const [faculty, setFaculty] = useState([]);
  const [classrooms, setClassrooms] = useState([]);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState("");
  const [submitSuccess, setSubmitSuccess] = useState(false);
  const [jobId, setJobId] = useState(null);
  const [importMode, setImportMode] = useState(false);
  const [importFile, setImportFile] = useState(null);
  const [importError, setImportError] = useState("");

  // Fetch timetable if an ID is passed
  useEffect(() => {
    if (timetableId) {
      fetchTimetable(timetableId).then((data) => {
        setBranches(data.branches);
        setCourses(data.courses);
        setFaculty(data.faculty);
        setClassrooms(data.classrooms);
        setTimetableData(data.timetable);
        setLoading(false);
      });
    } else {
      setLoading(false); // No ID, just allow empty form
    }
  }, [timetableId]);

  // Handle file import
  const handleFileImport = (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setImportFile(file);
    setImportError("");

    // Check file type
    if (!file.name.endsWith('.json') && !file.name.endsWith('.csv')) {
      setImportError("Please select a JSON or CSV file");
      return;
    }

    // Read and parse the file
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const content = e.target.result;
        let data;

        if (file.name.endsWith('.json')) {
          data = JSON.parse(content);
        } else {
          // For CSV, we'll need to parse it (simplified version)
          setImportError("CSV import not yet implemented. Please use JSON format.");
          return;
        }

        // Validate and populate the form data
        if (data.branches) setBranches(data.branches);
        if (data.courses) setCourses(data.courses);
        if (data.faculty) setFaculty(data.faculty);
        if (data.classrooms) setClassrooms(data.classrooms);

        setImportMode(false);
        alert("Data imported successfully!");
      } catch (error) {
        setImportError("Error parsing file: " + error.message);
      }
    };

    reader.readAsText(file);
  };

  // Handle submit
  const handleSubmit = async () => {
    // Validate that we have required data
    if (branches.length === 0) {
      setSubmitError("Please add at least one branch before creating a timetable.");
      return;
    }
    if (courses.length === 0) {
      setSubmitError("Please add at least one course before creating a timetable.");
      return;
    }
    if (faculty.length === 0) {
      setSubmitError("Please add at least one faculty member before creating a timetable.");
      return;
    }
    if (classrooms.length === 0) {
      setSubmitError("Please add at least one classroom before creating a timetable.");
      return;
    }

    setSubmitting(true);
    setSubmitError("");
    setSubmitSuccess(false);

    const finalData = {
      weekdays: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
      time_slots: [
        "09:00-10:00",
        "10:00-11:00", 
        "11:00-12:00",
        "12:00-13:00",
        "14:00-15:00",
        "15:00-16:00"
      ],
      branches: courses, // courses already contains branches with their courses
      faculty: faculty,
      classrooms: classrooms
    };

    try {
      // Get API URL from environment variable or use default
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
      const endpoint = timetableId ? `${apiUrl}/timetable/${timetableId}` : `${apiUrl}/timetable/`;
      const method = timetableId ? "PUT" : "POST";

      console.log("Submitting Data:", JSON.stringify(finalData, null, 2));
      console.log("API Endpoint:", endpoint);

      const response = await fetch(endpoint, {
        method: method,
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(finalData),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || errorData.message || `HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      console.log("Success:", result);
      
      // Handle async job response
      if (result.job_id) {
        setJobId(result.job_id);
        setSubmitSuccess(true);
        setSubmitError("");
        
        // Reset form after successful submission (optional)
        if (!timetableId) {
          setBranches([]);
          setCourses([]);
          setFaculty([]);
          setClassrooms([]);
        }
        
        console.log(`Timetable creation job started with ID: ${result.job_id}`);
      } else {
        setSubmitSuccess(true);
        setSubmitError("");
      }

    } catch (error) {
      console.error("Error submitting timetable:", error);
      setSubmitError(error.message || "Failed to create timetable. Please try again.");
      setSubmitSuccess(false);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="p-6 max-w-4xl mx-auto">
      {/* Page Title */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">📅 Create Timetable</h1>
          <p className="text-gray-600">
            {timetableId ? "Modify an existing timetable" : "Create a new timetable from scratch"}
          </p>
        </div>
        <Button
          onClick={() => setImportMode(!importMode)}
          variant="outline"
          className="flex items-center space-x-2"
        >
          <Upload className="h-4 w-4" />
          <span>Import Data</span>
        </Button>
      </div>

      {/* Import Section */}
      {importMode && (
        <Card className="mb-6">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <FileText className="h-5 w-5" />
              <span>Import Timetable Data</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <p className="text-sm text-gray-600">
                Import your timetable data from a JSON file. The file should contain branches, courses, faculty, and classrooms data.
              </p>
              <input
                type="file"
                accept=".json,.csv"
                onChange={handleFileImport}
                className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
              />
              {importError && (
                <Alert>
                  <AlertCircle className="h-4 w-4" />
                  <AlertDescription>{importError}</AlertDescription>
                </Alert>
              )}
              {importFile && (
                <div className="text-sm text-green-600">
                  Selected file: {importFile.name}
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Success Message */}
      {submitSuccess && (
        <Alert className="mb-6 border-green-200 bg-green-50">
          <CheckCircle className="h-4 w-4 text-green-600" />
          <AlertTitle className="text-green-800">Success!</AlertTitle>
          <AlertDescription className="text-green-700">
            {timetableId ? "Timetable updated successfully!" : "Timetable creation job started! Check the logs for progress."}
            {jobId && (
              <div className="mt-2">
                <p className="text-sm font-medium">Job ID: {jobId}</p>
                <p className="text-sm">You can monitor the progress at: <code className="bg-gray-100 px-1 rounded">/logs/{jobId}</code></p>
              </div>
            )}
          </AlertDescription>
        </Alert>
      )}

      {/* Error Message */}
      {submitError && (
        <Alert variant="destructive" className="mb-6">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{submitError}</AlertDescription>
        </Alert>
      )}

      {/* Loader */}
      {loading ? (
        <div className="space-y-4">
          <Skeleton className="h-12 w-full" />
          <Skeleton className="h-12 w-full" />
          <Skeleton className="h-12 w-full" />
          <Skeleton className="h-12 w-full" />
        </div>
      ) : (
        <>
          {/* Forms inside a Card */}
          <Card className="mb-6">
            <CardHeader>
              <CardTitle>Branch Details</CardTitle>
            </CardHeader>
            <CardContent>
              <BranchForm setBranches = {setBranches} branches = {branches}  />
            </CardContent>
          </Card>

          <Card className="mb-6">
            <CardHeader>
              <CardTitle>Course Details</CardTitle>
            </CardHeader>
            <CardContent>
              <CourseForm branches={branches} courses={courses} setCourses={setCourses} />
            </CardContent>
          </Card>

          <Card className="mb-6">
            <CardHeader>
              <CardTitle>Faculty Details</CardTitle>
            </CardHeader>
            <CardContent>
              <FacultyForm courses={courses} faculties={faculty} setFaculties={setFaculty} />
            </CardContent>
          </Card>

          <Card className="mb-6">
            <CardHeader>
              <CardTitle>Classroom Details</CardTitle>
            </CardHeader>
            <CardContent>
              <ClassroomForm classrooms={classrooms} setClassrooms={setClassrooms} />
            </CardContent>
          </Card>

          {/* Submit Button */}
          <Button 
            className="w-full mt-6 bg-blue-600 hover:bg-blue-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            onClick={handleSubmit}
            disabled={submitting}
          >
            {submitting ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                {timetableId ? "Updating..." : "Creating..."}
              </>
            ) : (
              timetableId ? "Modify Timetable" : "Create Timetable"
            )}
          </Button>
          
        </>
      )}
    </div>
  );
}
