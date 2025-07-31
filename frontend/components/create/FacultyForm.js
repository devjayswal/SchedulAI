"use client";
import { useState } from "react";
import { Dialog, DialogTrigger, DialogContent,DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { MultiSelect } from "@/components/ui/multiselect"; // ✅ Correct import

export default function FacultyForm({ courses,faculties ,setFaculties }) {
  const [facultyName, setFacultyName] = useState("");
  const [shortName, setShortName] = useState("");
  const [selectedCourses, setSelectedCourses] = useState([]);
  const [open, setOpen] = useState(false);

  const handleAddFaculty = () => {
    if (!facultyName.trim() || !shortName.trim() || selectedCourses.length === 0) return;
    setFaculties((prev)=>[...prev, {
      name: facultyName,
      shortName: shortName,
      courses: selectedCourses,
    }]);
    setFacultyName("");
    setShortName("");
    setSelectedCourses([]);
    setOpen(false);
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button>Add Faculty</Button>
      </DialogTrigger>
      <DialogContent>
      <DialogTitle>Add Faculty</DialogTitle>
      <DialogDescription>
          Enter the faculty's full name, short name, and select courses.
        </DialogDescription>
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
