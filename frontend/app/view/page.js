"use client";
import { useState, useEffect } from "react";
import { useSearchParams } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogTrigger } from "@/components/ui/dialog";

export default function View() {
    const searchParams = useSearchParams();
    const id = searchParams.get("id"); // Get ID from URL if modifying

    const [timetable, setTimetable] = useState({
        faculty: ["Dr. John Doe", "Prof. Alice Smith"],
        courses: ["Math 101", "Physics 202"],
        branches: ["CSE", "ECE"],
        classrooms: ["Room 101", "Lab 202"],
    });

    //create a excell like ui for timetable
    return (
    <>
        
    
    
    
    </>
    )
}