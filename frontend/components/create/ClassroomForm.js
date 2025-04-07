"use client";
import { useState } from "react";
import { Dialog, DialogTrigger, DialogContent } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectTrigger, SelectContent, SelectItem } from "@/components/ui/select";

export default function ClassroomForm({ onClassroomAdd }) {
  const [roomName, setRoomName] = useState("");
  const [roomType, setRoomType] = useState("");

  const handleAddRoom = () => {
    if (!roomName.trim() || !roomType) return;
    onClassroomAdd({ name: roomName, type: roomType });
    setRoomName("");
  };

  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button>Add Classroom</Button>
      </DialogTrigger>
      <DialogContent>
        <h3 className="text-lg font-semibold">Add Classroom</h3>
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
