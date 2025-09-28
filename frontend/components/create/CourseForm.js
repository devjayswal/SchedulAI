"use client";
import { useState } from "react";
import { Dialog, DialogTrigger, DialogContent,DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectTrigger, SelectContent, SelectItem } from "@/components/ui/select";
import { X, ChevronDown, ChevronUp } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";

export default function CourseForm({ branches, courses, setCourses }) {
  const [courseName, setCourseName] = useState("");
  const [courseCode, setCourseCode] = useState("");
  const [selectedBranch, setSelectedBranch] = useState("");
  const [subjectType, setSubjectType] = useState("");
  const [credits, setCredits] = useState("");
  const [facultyId, setFacultyId] = useState("");
  const [open, setOpen] = useState(false);
  const [showAll, setShowAll] = useState(false);

  const handleAddCourse = () => {
    if (!courseName.trim() || !courseCode.trim() || !selectedBranch || !subjectType || !credits || !facultyId) return;
    
    const newCourse = {
      subject_code: courseCode.trim(),
      subject_name: courseName.trim(),
      subject_type: subjectType,
      credits: parseInt(credits),
      faculty_id: facultyId.trim()
    };

    // Find the branch and add course to it
    setCourses((prevCourses) => {
      const updatedCourses = [...prevCourses];
      const branchIndex = updatedCourses.findIndex(branch => branch.branch_name === selectedBranch);
      
      if (branchIndex !== -1) {
        updatedCourses[branchIndex].courses.push(newCourse);
      } else {
        // If branch doesn't exist in courses, create it
        updatedCourses.push({
          branch_name: selectedBranch,
          semester: branches.find(b => b.branch_name === selectedBranch)?.semester || 1,
          courses: [newCourse]
        });
      }
      return updatedCourses;
    });

    setCourseName("");
    setCourseCode("");
    setSelectedBranch("");
    setSubjectType("");
    setCredits("");
    setFacultyId("");
    setOpen(false);
  };

  const handleRemoveCourse = (branchName, courseIndex) => {
    setCourses((prevCourses) => {
      const updatedCourses = [...prevCourses];
      const branchIndex = updatedCourses.findIndex(branch => branch.branch_name === branchName);
      
      if (branchIndex !== -1) {
        updatedCourses[branchIndex].courses.splice(courseIndex, 1);
        // Remove branch if no courses left
        if (updatedCourses[branchIndex].courses.length === 0) {
          updatedCourses.splice(branchIndex, 1);
        }
      }
      return updatedCourses;
    });
  };

  const allCourses = courses.flatMap(branch => 
    branch.courses.map(course => ({ ...course, branchName: branch.branch_name }))
  );
  const displayedCourses = showAll ? allCourses : allCourses.slice(0, 3);

  return (
    <div className="space-y-4">
      {/* Display existing courses */}
      {allCourses.length > 0 && (
        <div className="space-y-2">
          <h4 className="text-sm font-medium text-gray-700">Added Courses ({allCourses.length})</h4>
          <div className="space-y-2 max-h-60 overflow-y-auto">
            {displayedCourses.map((course, index) => (
              <Card key={index} className="p-3">
                <CardContent className="p-0 flex items-center justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <span className="font-medium">{course.subject_name}</span>
                      <span className="text-sm text-gray-500">({course.subject_code})</span>
                    </div>
                    <div className="text-sm text-gray-500">
                      {course.branchName} • {course.subject_type} • {course.credits} credits • {course.faculty_id}
                    </div>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => handleRemoveCourse(course.branchName, index)}
                    className="h-6 w-6 p-0 text-red-500 hover:text-red-700"
                  >
                    <X className="h-4 w-4" />
                  </Button>
                </CardContent>
              </Card>
            ))}
          </div>
          {allCourses.length > 3 && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowAll(!showAll)}
              className="w-full text-blue-600 hover:text-blue-800"
            >
              {showAll ? (
                <>
                  <ChevronUp className="h-4 w-4 mr-1" />
                  Show Less
                </>
              ) : (
                <>
                  <ChevronDown className="h-4 w-4 mr-1" />
                  Show All ({allCourses.length - 3} more)
                </>
              )}
            </Button>
          )}
        </div>
      )}

      {/* Add Course Dialog */}
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogTrigger asChild>
          <Button>Add Course</Button>
        </DialogTrigger>
        <DialogContent className="max-w-md">
          <DialogTitle>Add New Course</DialogTitle>
          <DialogDescription>
            Enter the course details and select the branch.
          </DialogDescription>

          <Input
            type="text"
            placeholder="Course Name"
            value={courseName}
            onChange={(e) => setCourseName(e.target.value)}
          />
          <Input
            type="text"
            placeholder="Course Code"
            value={courseCode}
            onChange={(e) => setCourseCode(e.target.value)}
          />
          <Select onValueChange={setSelectedBranch}>
            <SelectTrigger>Choose Branch</SelectTrigger>
            <SelectContent>
              {Array.isArray(branches) && branches.length > 0 ? (
                branches.map((branch, index) => (
                  <SelectItem key={index} value={branch.branch_name}>
                    {branch.branch_name}
                  </SelectItem>
                ))
              ) : (
                <SelectItem disabled>No branches available</SelectItem>
              )}
            </SelectContent>
          </Select>
          <Select onValueChange={setSubjectType}>
            <SelectTrigger>Subject Type</SelectTrigger>
            <SelectContent>
              <SelectItem value="theory">Theory</SelectItem>
              <SelectItem value="lab">Lab</SelectItem>
            </SelectContent>
          </Select>
          <Input
            type="number"
            placeholder="Credits"
            value={credits}
            onChange={(e) => setCredits(e.target.value)}
            min={1}
            max={6}
          />
          <Input
            type="text"
            placeholder="Faculty ID (e.g., F01)"
            value={facultyId}
            onChange={(e) => setFacultyId(e.target.value)}
          />
          <Button onClick={handleAddCourse} className="mt-2">Save</Button>
        </DialogContent>
      </Dialog>
    </div>
  );
}
