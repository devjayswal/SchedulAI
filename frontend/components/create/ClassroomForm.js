"use client";
import { useState } from "react";
import { Dialog, DialogTrigger, DialogContent,DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectTrigger, SelectContent, SelectItem } from "@/components/ui/select";

export default function ClassroomForm({ setClassrooms, classrooms }) {
  const [roomName, setRoomName] = useState("");
  const [roomType, setRoomType] = useState("");
  const [open, setOpen] = useState(false);

  const handleAddRoom = () => {
    if (!roomName.trim() || !roomType) return;
   setClassrooms((prev) => [
      ...prev,
      { name: roomName.trim(), type: roomType },
    ]);
    setRoomName("");
    setOpen(false);
    setRoomType("");
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button>Add Classroom</Button>
      </DialogTrigger>
      <DialogContent>
        <DialogTitle>Add Classroom</DialogTitle>
        <DialogDescription>
          Enter the classroom name and select its type.
        </DialogDescription>
        <Input
          type="text"
          placeholder="Room Name"
          value={roomName}
          onChange={(e) => setRoomName(e.target.value)}
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
  );
}
