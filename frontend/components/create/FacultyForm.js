"use client";
import { useState } from "react";
import { Dialog, DialogTrigger, DialogContent,DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { MultiSelect } from "@/components/ui/multiselect";
import { X, ChevronDown, ChevronUp } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";

export default function FacultyForm({ courses, faculties, setFaculties }) {
  const [facultyId, setFacultyId] = useState("");
  const [facultyName, setFacultyName] = useState("");
  const [email, setEmail] = useState("");
  const [open, setOpen] = useState(false);
  const [showAll, setShowAll] = useState(false);

  const handleAddFaculty = () => {
    if (!facultyId.trim() || !facultyName.trim() || !email.trim()) return;
    
    const newFaculty = {
      id: facultyId.trim(),
      name: facultyName.trim(),
      email: email.trim()
    };
    
    setFaculties((prev) => [...prev, newFaculty]);
    setFacultyId("");
    setFacultyName("");
    setEmail("");
    setOpen(false);
  };

  const handleRemoveFaculty = (index) => {
    setFaculties(prev => prev.filter((_, i) => i !== index));
  };

  const displayedFaculties = showAll ? faculties : faculties.slice(0, 3);

  return (
    <div className="space-y-4">
      {/* Display existing faculties */}
      {faculties.length > 0 && (
        <div className="space-y-2">
          <h4 className="text-sm font-medium text-gray-700">Added Faculty ({faculties.length})</h4>
          <div className="space-y-2 max-h-60 overflow-y-auto">
            {displayedFaculties.map((faculty, index) => (
              <Card key={index} className="p-3">
                <CardContent className="p-0 flex items-center justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <span className="font-medium">{faculty.name}</span>
                      <span className="text-sm text-gray-500">({faculty.id})</span>
                    </div>
                    <div className="text-sm text-gray-500">{faculty.email}</div>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => handleRemoveFaculty(index)}
                    className="h-6 w-6 p-0 text-red-500 hover:text-red-700"
                  >
                    <X className="h-4 w-4" />
                  </Button>
                </CardContent>
              </Card>
            ))}
          </div>
          {faculties.length > 3 && (
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
                  Show All ({faculties.length - 3} more)
                </>
              )}
            </Button>
          )}
        </div>
      )}

      {/* Add Faculty Dialog */}
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogTrigger asChild>
          <Button>Add Faculty</Button>
        </DialogTrigger>
        <DialogContent>
          <DialogTitle>Add Faculty</DialogTitle>
          <DialogDescription>
            Enter the faculty's ID, name, and email.
          </DialogDescription>
          <Input
            type="text"
            placeholder="Faculty ID (e.g., F01)"
            value={facultyId}
            onChange={(e) => setFacultyId(e.target.value)}
          />
          <Input
            type="text"
            placeholder="Full Name"
            value={facultyName}
            onChange={(e) => setFacultyName(e.target.value)}
          />
          <Input
            type="email"
            placeholder="Email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
          />
          <Button onClick={handleAddFaculty} className="mt-2">Save</Button>
        </DialogContent>
      </Dialog>
    </div>
  );
}
