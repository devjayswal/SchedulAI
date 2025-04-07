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

// Mock function to simulate fetching from backend
const fetchTimetable = async (id) => {
  console.log(`Fetching timetable for ID: ${id}`);
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve({
        branches: [{ name: "CSE" }],
        courses: [{ name: "AI", branch: "CSE" }],
        faculties: [{ name: "Dr. Smith", courses: ["AI"] }],
        classrooms: [{ type: "Theory" }],
      });
    }, 1000);
  });
};

export default function CreatePage() {
  const searchParams = useSearchParams();
  const timetableId = searchParams.get("id"); // Get ID if passed

  // State for storing form data
  const [branches, setBranches] = useState([]);
  const [courses, setCourses] = useState([]);
  const [faculties, setFaculties] = useState([]);
  const [classrooms, setClassrooms] = useState([]);
  const [loading, setLoading] = useState(true);

  // Fetch timetable if an ID is passed
  useEffect(() => {
    if (timetableId) {
      fetchTimetable(timetableId).then((data) => {
        setBranches(data.branches);
        setCourses(data.courses);
        setFaculties(data.faculties);
        setClassrooms(data.classrooms);
        setLoading(false);
      });
    } else {
      setLoading(false); // No ID, just allow empty form
    }
  }, [timetableId]);

  // Handle submit
  const handleSubmit = async () => {
    const timetableData = { branches, courses, faculties, classrooms };
    console.log("Submitting Data:", JSON.stringify(timetableData));
    // Send this JSON to backend via POST request
  };

  return (
    <div className="p-6 max-w-4xl mx-auto">
      {/* Page Title */}
      <h1 className="text-3xl font-bold text-gray-900">📅 Create Timetable</h1>
      <p className="text-gray-600 mb-6">
        {timetableId ? "Modify an existing timetable" : "Create a new timetable from scratch"}
      </p>

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
              <BranchForm onBranchAdd={(newBranch) => setBranches([...branches, { name: newBranch }])}  />
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
              <FacultyForm courses={courses} faculties={faculties} setFaculties={setFaculties} />
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
            className="w-full mt-6 bg-blue-600 hover:bg-blue-700 transition-all"
            onClick={handleSubmit}
          >
            💾 Save Timetable
          </Button>
        </>
      )}
    </div>
  );
}
