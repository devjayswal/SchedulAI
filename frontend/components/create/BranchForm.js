"use client";
import { useState } from "react";
import { Dialog, DialogTrigger, DialogContent, DialogTitle,DialogDescription } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { AlertCircle, X, ChevronDown, ChevronUp } from "lucide-react";
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert";
import { Card, CardContent } from "@/components/ui/card";

export default function BranchForm({setBranches , branches}) {
  const [branchName, setBranchName] = useState("");
  const [branchSem, setBranchSem] = useState(1); // Default to 1
  const [error, setError] = useState("");
  const [open, setOpen] = useState(false);
  const [showAll, setShowAll] = useState(false);

  const handleAddBranch = () => {
    if (!branchName.trim()) {
      setError("Branch name cannot be empty.");
      return;
    }

    if (branchName.length > 6) {
      setError("Branch name cannot be more than 6 letters.");
      return;
    }

    const semNumber = Number(branchSem);
    if (isNaN(semNumber) || semNumber < 1 || semNumber > 8) {
      setError("Semester must be between 1 and 8.");
      return;
    }

    setError(""); // Clear any previous error
    const newBranch = {
      branch_name: branchName.trim(),
      semester: semNumber,
      courses: [] // Initialize empty courses array
    }
    setBranches(prev => [...prev, newBranch]);
    setBranchName("");
    setBranchSem(1);
    setOpen(false);
  };

  const handleRemoveBranch = (index) => {
    setBranches(prev => prev.filter((_, i) => i !== index));
  };

  const displayedBranches = showAll ? branches : branches.slice(0, 3);

  return (
    <div className="space-y-4">
      {/* Display existing branches */}
      {branches.length > 0 && (
        <div className="space-y-2">
          <h4 className="text-sm font-medium text-gray-700">Added Branches ({branches.length})</h4>
          <div className="space-y-2 max-h-60 overflow-y-auto">
            {displayedBranches.map((branch, index) => (
              <Card key={index} className="p-3">
                <CardContent className="p-0 flex items-center justify-between">
                  <div className="flex-1">
                    <span className="font-medium">{branch.branch_name}</span>
                    <span className="text-gray-500 ml-2">(Sem {branch.semester})</span>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => handleRemoveBranch(index)}
                    className="h-6 w-6 p-0 text-red-500 hover:text-red-700"
                  >
                    <X className="h-4 w-4" />
                  </Button>
                </CardContent>
              </Card>
            ))}
          </div>
          {branches.length > 3 && (
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
                  Show All ({branches.length - 3} more)
                </>
              )}
            </Button>
          )}
        </div>
      )}

      {/* Add Branch Dialog */}
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogTrigger asChild>
          <Button>Add Branch</Button>
        </DialogTrigger>
        <DialogContent>
          <DialogTitle>Add Branch</DialogTitle>
          <DialogDescription>
            Enter the branch short name and semester. 
          </DialogDescription>
          <Input
            type="text"
            placeholder="Enter branch short name"
            value={branchName}
            onChange={(e) => setBranchName(e.target.value)}
          />
          
          <Input
            type="number"
            placeholder="Enter sem (1-8)"
            value={branchSem}
            onChange={(e) => setBranchSem(e.target.value)}
            min={1}
            max={8}
          />

          {error && (
            <Alert variant="destructive" className="mt-2">
              <AlertCircle className="h-4 w-4" />
              <AlertTitle>Warning</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          <Button onClick={handleAddBranch} className="mt-2">
            Save
          </Button>
        </DialogContent>
      </Dialog>
    </div>
  );
}
