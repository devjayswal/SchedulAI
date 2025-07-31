"use client";
import { useState } from "react";
import { Dialog, DialogTrigger, DialogContent,DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectTrigger, SelectContent, SelectItem } from "@/components/ui/select";

export default function CourseForm({ branches, courses, setCourses }) {
  const [courseName, setCourseName] = useState("");
  const [courseCode, setCourseCode] = useState("");
  const [selectedBranch, setSelectedBranch] = useState("");
  const [open, setOpen] = useState(false);

  const handleAddCourse = () => {
    if (!courseName.trim() || !courseCode.trim() || !selectedBranch) return;
    setCourses((prevCourses) => [
      ...prevCourses,
      { name: courseName.trim(), code: courseCode.trim(), branch: selectedBranch },
    ]);
    setCourseName("");
    setCourseCode("");
    setSelectedBranch("");
    setOpen(false);

  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button>Add Course</Button>
      </DialogTrigger>
      <DialogContent>
        <DialogTitle>Add New Course</DialogTitle>
        <DialogDescription>
          Enter the course name, code, and select the branch.
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
                <SelectItem key={index} value={branch.name}>
                  {branch.name}
                </SelectItem>
              ))
            ) : (
              <SelectItem disabled>No branches available</SelectItem>
            )}

          </SelectContent>
        </Select>
        <Button onClick={handleAddCourse} className="mt-2">Save</Button>
      </DialogContent>
    </Dialog>
  );
}
