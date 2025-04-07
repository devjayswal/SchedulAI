"use client";
import { useState } from "react";
import { Dialog, DialogTrigger, DialogContent, DialogTitle } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { AlertCircle } from "lucide-react";
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert";

export default function BranchForm({ onBranchAdd }) {
  const [branchName, setBranchName] = useState("");
  const [branchSem, setBranchSem] = useState(1); // Default to 1
  const [error, setError] = useState("");

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
    onBranchAdd(`${branchName} sem ${branchSem}`); // Pass the branch name and semester to the parent component
    setBranchName("");
    setBranchSem("1");
    
  };

  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button>Add Branch</Button>
      </DialogTrigger>
      <DialogContent>
        <DialogTitle>Add Branch</DialogTitle>

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
  );
}
