"use client";
import { useState } from "react";
import { Dialog, DialogTrigger, DialogContent,DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectTrigger, SelectContent, SelectItem } from "@/components/ui/select";
import { X, ChevronDown, ChevronUp } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";

export default function ClassroomForm({ setClassrooms, classrooms }) {
  const [roomId, setRoomId] = useState("");
  const [roomType, setRoomType] = useState("");
  const [open, setOpen] = useState(false);
  const [showAll, setShowAll] = useState(false);

  const handleAddRoom = () => {
    if (!roomId.trim() || !roomType) return;
    
    const newClassroom = {
      id: roomId.trim(),
      type: roomType
    };
    
    setClassrooms((prev) => [...prev, newClassroom]);
    setRoomId("");
    setRoomType("");
    setOpen(false);
  };

  const handleRemoveClassroom = (index) => {
    setClassrooms(prev => prev.filter((_, i) => i !== index));
  };

  const displayedClassrooms = showAll ? classrooms : classrooms.slice(0, 3);

  return (
    <div className="space-y-4">
      {/* Display existing classrooms */}
      {classrooms.length > 0 && (
        <div className="space-y-2">
          <h4 className="text-sm font-medium text-gray-700">Added Classrooms ({classrooms.length})</h4>
          <div className="space-y-2 max-h-60 overflow-y-auto">
            {displayedClassrooms.map((classroom, index) => (
              <Card key={index} className="p-3">
                <CardContent className="p-0 flex items-center justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <span className="font-medium">{classroom.id}</span>
                      <span className="text-sm text-gray-500">({classroom.type})</span>
                    </div>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => handleRemoveClassroom(index)}
                    className="h-6 w-6 p-0 text-red-500 hover:text-red-700"
                  >
                    <X className="h-4 w-4" />
                  </Button>
                </CardContent>
              </Card>
            ))}
          </div>
          {classrooms.length > 3 && (
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
                  Show All ({classrooms.length - 3} more)
                </>
              )}
            </Button>
          )}
        </div>
      )}

      {/* Add Classroom Dialog */}
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogTrigger asChild>
          <Button>Add Classroom</Button>
        </DialogTrigger>
        <DialogContent>
          <DialogTitle>Add Classroom</DialogTitle>
          <DialogDescription>
            Enter the classroom ID and select its type.
          </DialogDescription>
          <Input
            type="text"
            placeholder="Room ID (e.g., CR101, LAB201)"
            value={roomId}
            onChange={(e) => setRoomId(e.target.value)}
          />
          <Select onValueChange={setRoomType}>
            <SelectTrigger>Choose Type</SelectTrigger>
            <SelectContent>
              <SelectItem value="theory">Theory Room</SelectItem>
              <SelectItem value="lab">Computer Lab</SelectItem>
            </SelectContent>
          </Select>
          <Button onClick={handleAddRoom} className="mt-2">Save</Button>
        </DialogContent>
      </Dialog>
    </div>
  );
}
