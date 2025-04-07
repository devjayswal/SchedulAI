"use client";
import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogTrigger, DialogTitle } from "@/components/ui/dialog";

export default function HomePage() {
  const router = useRouter();
  const [history, setHistory] = useState([]);
  const [selectedTimetable, setSelectedTimetable] = useState(null);
  const [isDeleteDialogOpen, setIsDeleteDialogOpen] = useState(false);

  useEffect(() => {
    // Sample timetable history data
    setHistory([
      { id: 1, name: "Timetable - 2024/04/07" },
      { id: 2, name: "Timetable - 2024/04/06" },
      { id: 3, name: "Timetable - 2024/04/05" },
    ]);
  }, []);

  const handleDelete = () => {
    setHistory((prev) => prev.filter((item) => item.id !== selectedTimetable));
    setIsDeleteDialogOpen(false);
  };

  const handlePrint = (id) => {
    alert(`Printing timetable ${id}... (Feature Coming Soon!)`);
  };

  return (
    <div className="p-6 max-w-4xl mx-auto">
      {/* Description */}
      <p className="text-gray-600 mb-4">
        Welcome to Scheduler! Easily create and manage timetables for faculties, courses, and classrooms.
      </p>

      {/* Create New Button */}
      <Button onClick={() => router.push("/create")}>Create New</Button>

      {/* Timetable History */}
      <div className="mt-6">
        <h2 className="text-lg font-semibold">Past Timetables</h2>
        <ul className="mt-2 space-y-2">
          {history.length === 0 ? (
            <p className="text-gray-500">No timetables found.</p>
          ) : (
            history.map((item) => (
              <li
                key={item.id}
                className="p-3 bg-gray-100 rounded-md flex justify-between items-center hover:bg-gray-200"
              >
                <span>{item.name}</span>
                <div className="space-x-2">
                  <Button onClick={() => router.push(`/create?id=${item.id}`)} size="sm">
                    Modify
                  </Button>
                  <Button onClick={() => handlePrint(item.id)} size="sm" variant="outline">
                    Print
                  </Button>
                  <Button
                    onClick={() => {
                      setSelectedTimetable(item.id);
                      setIsDeleteDialogOpen(true);
                    }}
                    size="sm"
                    variant="destructive"
                  >
                    Delete
                  </Button>
                </div>
              </li>
            ))
          )}
        </ul>
      </div>

      {/* Delete Confirmation Dialog */}
      <Dialog open={isDeleteDialogOpen} onOpenChange={setIsDeleteDialogOpen}>
        <DialogContent>
          <DialogTitle>Delete Timetable</DialogTitle>
          <p>Are you sure you want to delete this timetable? This action cannot be undone.</p>
          <div className="flex justify-end space-x-2 mt-4">
            <Button onClick={() => setIsDeleteDialogOpen(false)} variant="outline">
              Cancel
            </Button>
            <Button onClick={handleDelete} variant="destructive">
              Delete
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
