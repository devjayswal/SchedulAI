"use client";
import { useState } from "react";
import { Dialog, DialogTrigger, DialogContent,DialogTitle } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { MultiSelect } from "@/components/ui/multiselect"; // ✅ Correct import

export default function FacultyForm({ courses, onFacultyAdd }) {
  const [facultyName, setFacultyName] = useState("");
  const [shortName, setShortName] = useState("");
  const [selectedCourses, setSelectedCourses] = useState([]);

  const handleAddFaculty = () => {
    if (!facultyName.trim() || !shortName.trim() || selectedCourses.length === 0) return;
    onFacultyAdd({ name: facultyName, shortName, courses: selectedCourses });
    setFacultyName("");
    setShortName("");
    setSelectedCourses([]);
  };

  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button>Add Faculty</Button>
      </DialogTrigger>
      <DialogContent>
      <DialogTitle>Add Faculty</DialogTitle>
      <Input
          type="text"
          placeholder="Full Name"
          value={facultyName}
          onChange={(e) => setFacultyName(e.target.value)}
        />
        <Input
          type="text"
          placeholder="Short Name"
          value={shortName}
          onChange={(e) => setShortName(e.target.value)}
        />
        {/* ✅ Use MultiSelect Here */}
        <MultiSelect
          options={courses.map((course) => ({ label: course.name, value: course.code }))}
          selected={selectedCourses}
          onChange={setSelectedCourses}
        />
        <Button onClick={handleAddFaculty} className="mt-2">Save</Button>
      </DialogContent>
    </Dialog>
  );
}
