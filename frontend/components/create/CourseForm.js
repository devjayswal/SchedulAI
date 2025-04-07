"use client";
import { useState } from "react";
import { Dialog, DialogTrigger, DialogContent } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectTrigger, SelectContent, SelectItem } from "@/components/ui/select";

export default function CourseForm({ branches, onCourseAdd }) {
  const [courseName, setCourseName] = useState("");
  const [courseCode, setCourseCode] = useState("");
  const [selectedBranch, setSelectedBranch] = useState("");

  const handleAddCourse = () => {
    if (!courseName.trim() || !courseCode.trim() || !selectedBranch) return;
    onCourseAdd({ name: courseName, code: courseCode, branch: selectedBranch });
    setCourseName("");
    setCourseCode("");
  };

  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button>Add Course</Button>
      </DialogTrigger>
      <DialogContent>
        <h3 className="text-lg font-semibold">Add Course</h3>
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
            {branches.map((branch, index) => (
              <SelectItem key={index} value={branch}>{branch}</SelectItem>
            ))}
          </SelectContent>
        </Select>
        <Button onClick={handleAddCourse} className="mt-2">Save</Button>
      </DialogContent>
    </Dialog>
  );
}
