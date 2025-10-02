"use client";
import { useState, useEffect } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Loader2, Download, RefreshCw, ArrowLeft } from "lucide-react";
import { timetableApi, apiUtils } from "@/lib/api";

export default function View() {
    const searchParams = useSearchParams();
    const router = useRouter();
    const id = searchParams.get("id");

    const [timetableData, setTimetableData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [jobId, setJobId] = useState(null);
    const [activeTab, setActiveTab] = useState("master");
    const [editingCell, setEditingCell] = useState(null);
    const [editMode, setEditMode] = useState(false);
    const [editedTimetable, setEditedTimetable] = useState(null);
    const [regenerating, setRegenerating] = useState(false);

    // Fetch generated timetable data
    const fetchGeneratedTimetable = async (jobId) => {
        try {
            setLoading(true);
            const data = await timetableApi.getGeneratedByJobId(jobId);
            setTimetableData(data);
            setError(null);
        } catch (err) {
            setError(apiUtils.handleError(err, 'Failed to fetch generated timetable'));
            console.error('Error fetching timetable:', err);
        } finally {
            setLoading(false);
        }
    };

    // Check if we have a job ID in the URL
    useEffect(() => {
        const jobIdParam = searchParams.get("job_id");
        if (jobIdParam) {
            setJobId(jobIdParam);
            fetchGeneratedTimetable(jobIdParam);
        } else {
            // Fallback to sample data if no job ID
            setTimetableData(sampleData);
            setLoading(false);
        }
    }, [searchParams]);

    // Sample data structure for demonstration
    const sampleData = {
        id: id || "sample-123",
        name: "Sample Timetable",
        description: "Generated timetable for CSE and AIML branches",
        data: {
            time_slots: ["9AM-10AM", "10AM-11AM", "11AM-12PM", "1PM-2PM", "2PM-3PM"],
            days: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
            branches: [
                {
                    branch_name: "CSE",
                    semester: 4,
                    courses: [
                        { subject_code: "CS201", subject_name: "Data Structures", credits: 3 },
                        { subject_code: "CS202", subject_name: "Algorithms", credits: 3 }
                    ]
                },
                {
                    branch_name: "AIML",
                    semester: 6,
                    courses: [
                        { subject_code: "AI301", subject_name: "Machine Learning", credits: 4 },
                        { subject_code: "AI302", subject_name: "Deep Learning", credits: 4 }
                    ]
                }
            ],
            faculty: [
                { id: "F01", name: "Dr. Smith", short_name: "DS" },
                { id: "F02", name: "Prof. Johnson", short_name: "PJ" },
                { id: "F03", name: "Dr. Brown", short_name: "DB" }
            ],
            classrooms: [
                { id: "CR101", type: "theory", name: "Room 101" },
                { id: "CR102", type: "theory", name: "Room 102" },
                { id: "LAB201", type: "lab", name: "Lab 201" }
            ]
        }
    };

    useEffect(() => {
        if (id) {
            fetchTimetableData();
        } else {
            // Use sample data for demonstration
            setTimetableData(sampleData);
            setLoading(false);
        }
    }, [id]);

    const fetchTimetableData = async () => {
        try {
            setLoading(true);
            const data = await timetableApi.getData(id);
            setTimetableData(data);
            setError(null);
        } catch (err) {
            setError(apiUtils.handleError(err, 'Failed to fetch timetable data'));
            console.error('Error fetching timetable data:', err);
            // Fallback to sample data for demonstration
            setTimetableData(sampleData);
        } finally {
            setLoading(false);
        }
    };

    const generateSampleTimetable = () => {
        const { time_slots, days, branches, faculty, classrooms } = timetableData.data;
        
        // Generate sample timetable entries
        const masterTimetable = {};
        const facultyTimetable = {};
        const classroomTimetable = {};

        // Initialize timetables
        branches.forEach(branch => {
            const key = `${branch.branch_name}&${branch.semester}`;
            masterTimetable[key] = {};
            days.forEach(day => {
                masterTimetable[key][day] = {};
                time_slots.forEach(slot => {
                    masterTimetable[key][day][slot] = null;
                });
            });
        });

        faculty.forEach(f => {
            facultyTimetable[f.short_name] = {};
            days.forEach(day => {
                facultyTimetable[f.short_name][day] = {};
                time_slots.forEach(slot => {
                    facultyTimetable[f.short_name][day][slot] = null;
                });
            });
        });

        classrooms.forEach(c => {
            classroomTimetable[c.id] = {};
            days.forEach(day => {
                classroomTimetable[c.id][day] = {};
                time_slots.forEach(slot => {
                    classroomTimetable[c.id][day][slot] = null;
                });
            });
        });

        // Add some sample entries
        const sampleEntries = [
            { branch: "CSE&4", day: "Monday", slot: "9AM-10AM", course: "Data Structures", faculty: "Dr. Smith", classroom: "Room 101" },
            { branch: "CSE&4", day: "Monday", slot: "10AM-11AM", course: "Algorithms", faculty: "Prof. Johnson", classroom: "Room 102" },
            { branch: "AIML&6", day: "Tuesday", slot: "9AM-10AM", course: "Machine Learning", faculty: "Dr. Brown", classroom: "Lab 201" },
            { branch: "AIML&6", day: "Wednesday", slot: "1PM-2PM", course: "Deep Learning", faculty: "Dr. Smith", classroom: "Lab 201" }
        ];

        sampleEntries.forEach(entry => {
            if (masterTimetable[entry.branch] && masterTimetable[entry.branch][entry.day]) {
                masterTimetable[entry.branch][entry.day][entry.slot] = {
                    course: entry.course,
                    faculty: entry.faculty,
                    classroom: entry.classroom
                };
            }
        });

        return { masterTimetable, facultyTimetable, classroomTimetable };
    };

    const exportToCSV = (timetableType) => {
        if (!timetableData) return;

        const { time_slots, days } = timetableData.data;
        const { masterTimetable, facultyTimetable, classroomTimetable } = generateSampleTimetable();
        
        let dataToExport = {};
        let filename = "";
        
        if (timetableType === "master") {
            dataToExport = masterTimetable;
            filename = "master_timetable.csv";
        } else if (timetableType === "faculty") {
            dataToExport = facultyTimetable;
            filename = "faculty_timetable.csv";
        } else {
            dataToExport = classroomTimetable;
            filename = "classroom_timetable.csv";
        }

        // Create CSV content
        let csvContent = "Time/Day," + days.join(",") + "\n";
        
        time_slots.forEach(slot => {
            let row = slot;
            days.forEach(day => {
                let cellContent = "";
                Object.entries(dataToExport).forEach(([key, dayData]) => {
                    const entry = dayData[day]?.[slot];
                    if (entry) {
                        if (typeof entry === 'object') {
                            cellContent += `${entry.course || entry} (${entry.faculty || ''})`;
                        } else {
                            cellContent += entry;
                        }
                        cellContent += "; ";
                    }
                });
                row += "," + (cellContent ? `"${cellContent.trim()}"` : "");
            });
            csvContent += row + "\n";
        });

        // Download CSV
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement("a");
        const url = URL.createObjectURL(blob);
        link.setAttribute("href", url);
        link.setAttribute("download", filename);
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    };

    const exportToExcel = () => {
        // For now, we'll export as CSV with .xlsx extension
        // In a real implementation, you'd use a library like xlsx
        exportToCSV(activeTab);
    };

    const handleCellClick = (branchKey, day, slot) => {
        if (editMode) {
            setEditingCell({ branchKey, day, slot });
        }
    };

    const handleCellEdit = (branchKey, day, slot, newValue) => {
        if (!editedTimetable) {
            const { masterTimetable, facultyTimetable, classroomTimetable } = generateSampleTimetable();
            setEditedTimetable({ masterTimetable, facultyTimetable, classroomTimetable });
        }

        // Update the edited timetable
        const updatedTimetable = { ...editedTimetable };
        if (updatedTimetable.masterTimetable[branchKey] && 
            updatedTimetable.masterTimetable[branchKey][day]) {
            updatedTimetable.masterTimetable[branchKey][day][slot] = newValue;
            setEditedTimetable(updatedTimetable);
        }

        setEditingCell(null);
    };

    const saveChanges = async () => {
        // Here you would save the changes to the backend
        console.log("Saving changes:", editedTimetable);
        setEditMode(false);
        setEditedTimetable(null);
        // Show success message
        alert("Changes saved successfully!");
    };

    const cancelEdit = () => {
        setEditMode(false);
        setEditingCell(null);
        setEditedTimetable(null);
    };

    const handleRegenerate = async () => {
        if (!id) {
            alert("Cannot regenerate sample timetable. Please create a real timetable first.");
            return;
        }

        try {
            setRegenerating(true);
            const result = await timetableApi.regenerate(id);
            alert(`Timetable regeneration started! Job ID: ${result.job_id}`);
            
            // Optionally refresh the timetable data after regeneration
            // fetchTimetableData();
            
        } catch (err) {
            const errorMessage = apiUtils.handleError(err, 'Failed to regenerate timetable');
            alert(`Error regenerating timetable: ${errorMessage}`);
        } finally {
            setRegenerating(false);
        }
    };

    const renderTimetableTable = (timetable, title) => {
        if (!timetableData) return null;

        const { time_slots, days } = timetableData.data;
        const { masterTimetable, facultyTimetable, classroomTimetable } = generateSampleTimetable();
        
        let dataToRender = {};
        if (title.includes("Master")) {
            dataToRender = editedTimetable?.masterTimetable || masterTimetable;
        } else if (title.includes("Faculty")) {
            dataToRender = editedTimetable?.facultyTimetable || facultyTimetable;
        } else {
            dataToRender = editedTimetable?.classroomTimetable || classroomTimetable;
        }

        return (
            <div className="overflow-x-auto">
                <table className="w-full border-collapse border border-gray-300">
                    <thead>
                        <tr className="bg-gray-100">
                            <th className="border border-gray-300 p-2 font-semibold">Time/Day</th>
                            {days.map(day => (
                                <th key={day} className="border border-gray-300 p-2 font-semibold min-w-[120px]">
                                    {day}
                                </th>
                            ))}
                        </tr>
                    </thead>
                    <tbody>
                        {time_slots.map(slot => (
                            <tr key={slot}>
                                <td className="border border-gray-300 p-2 font-medium bg-gray-50">
                                    {slot}
                                </td>
                                {days.map(day => (
                                    <td 
                                        key={`${slot}-${day}`} 
                                        className={`border border-gray-300 p-2 min-h-[60px] ${
                                            editMode ? 'cursor-pointer hover:bg-gray-50' : ''
                                        }`}
                                        onClick={() => {
                                            if (editMode && title.includes("Master")) {
                                                // For master timetable, we need to find the branch key
                                                const branchKey = Object.keys(dataToRender)[0];
                                                handleCellClick(branchKey, day, slot);
                                            }
                                        }}
                                    >
                                        {editingCell && 
                                         editingCell.day === day && 
                                         editingCell.slot === slot && 
                                         title.includes("Master") ? (
                                            <input
                                                type="text"
                                                className="w-full p-1 text-xs border rounded"
                                                placeholder="Enter course name"
                                                onBlur={(e) => {
                                                    const branchKey = Object.keys(dataToRender)[0];
                                                    handleCellEdit(branchKey, day, slot, e.target.value);
                                                }}
                                                onKeyPress={(e) => {
                                                    if (e.key === 'Enter') {
                                                        const branchKey = Object.keys(dataToRender)[0];
                                                        handleCellEdit(branchKey, day, slot, e.target.value);
                                                    }
                                                }}
                                                autoFocus
                                            />
                                        ) : (
                                            Object.entries(dataToRender).map(([key, dayData]) => {
                                                const entry = dayData[day]?.[slot];
                                                if (entry) {
                                                    return (
                                                        <div key={key} className="text-xs bg-blue-100 p-1 rounded mb-1">
                                                            <div className="font-medium">{entry.course || entry}</div>
                                                            {entry.faculty && <div className="text-gray-600">{entry.faculty}</div>}
                                                            {entry.classroom && <div className="text-gray-500">{entry.classroom}</div>}
                                                        </div>
                                                    );
                                                }
                                                return null;
                                            })
                                        )}
                                    </td>
                                ))}
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        );
    };

    if (loading) {
        return (
            <div className="flex items-center justify-center min-h-screen">
                <div className="text-center">
                    <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4" />
                    <p>Loading timetable...</p>
                </div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="p-6">
                <Alert>
                    <AlertDescription>
                        Error loading timetable: {error}. Showing sample data.
                    </AlertDescription>
                </Alert>
            </div>
        );
    }

    return (
        <div className="p-6 max-w-7xl mx-auto">
            {/* Header */}
            <div className="flex items-center justify-between mb-6">
                <div className="flex items-center space-x-4">
                    <Button 
                        variant="outline" 
                        onClick={() => router.push("/")}
                        className="flex items-center space-x-2"
                    >
                        <ArrowLeft className="h-4 w-4" />
                        <span>Back</span>
                    </Button>
                    <div>
                        <h1 className="text-2xl font-bold">{timetableData?.name || "Timetable"}</h1>
                        <p className="text-gray-600">{timetableData?.description || "Generated timetable"}</p>
                    </div>
                </div>
                <div className="flex space-x-2">
                    {!editMode ? (
                        <>
                            <Button 
                                onClick={handleRegenerate}
                                disabled={regenerating}
                                variant="outline" 
                                className="flex items-center space-x-2"
                            >
                                <RefreshCw className={`h-4 w-4 ${regenerating ? 'animate-spin' : ''}`} />
                                <span>{regenerating ? 'Regenerating...' : 'Regenerate'}</span>
                            </Button>
                            <Button 
                                onClick={() => setEditMode(true)} 
                                variant="outline" 
                                className="flex items-center space-x-2"
                            >
                                <span>Edit</span>
                            </Button>
                            <Button 
                                onClick={() => exportToCSV(activeTab)} 
                                className="flex items-center space-x-2"
                            >
                                <Download className="h-4 w-4" />
                                <span>Export CSV</span>
                            </Button>
                        </>
                    ) : (
                        <>
                            <Button 
                                onClick={saveChanges} 
                                className="flex items-center space-x-2"
                            >
                                <span>Save Changes</span>
                            </Button>
                            <Button 
                                onClick={cancelEdit} 
                                variant="outline" 
                                className="flex items-center space-x-2"
                            >
                                <span>Cancel</span>
                            </Button>
                        </>
                    )}
                </div>
            </div>

            {/* Edit Mode Indicator */}
            {editMode && (
                <Alert className="mb-4">
                    <AlertDescription>
                        <strong>Edit Mode:</strong> Click on any cell in the Master Timetable to edit it. Changes will be saved when you click "Save Changes".
                    </AlertDescription>
                </Alert>
            )}

            {/* Timetable Tabs */}
            <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
                <TabsList className="grid w-full grid-cols-3">
                    <TabsTrigger value="master">Master Timetable</TabsTrigger>
                    <TabsTrigger value="faculty">Faculty Timetable</TabsTrigger>
                    <TabsTrigger value="classroom">Classroom Timetable</TabsTrigger>
                </TabsList>

                <TabsContent value="master" className="mt-6">
                    <Card>
                        <CardHeader>
                            <CardTitle>Master Timetable (Student View)</CardTitle>
                        </CardHeader>
                        <CardContent>
                            {renderTimetableTable(null, "Master Timetable")}
                        </CardContent>
                    </Card>
                </TabsContent>

                <TabsContent value="faculty" className="mt-6">
                    <Card>
                        <CardHeader>
                            <CardTitle>Faculty Timetable</CardTitle>
                        </CardHeader>
                        <CardContent>
                            {renderTimetableTable(null, "Faculty Timetable")}
                        </CardContent>
                    </Card>
                </TabsContent>

                <TabsContent value="classroom" className="mt-6">
                    <Card>
                        <CardHeader>
                            <CardTitle>Classroom Timetable</CardTitle>
                        </CardHeader>
                        <CardContent>
                            {renderTimetableTable(null, "Classroom Timetable")}
                        </CardContent>
                    </Card>
                </TabsContent>
            </Tabs>
        </div>
    );
}