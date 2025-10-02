"use client";
import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogTrigger, DialogTitle } from "@/components/ui/dialog";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Loader2, AlertCircle, RefreshCw } from "lucide-react";
import { timetableApi, statusApi, apiUtils } from "@/lib/api";

export default function HomePage() {
  const router = useRouter();
  const [history, setHistory] = useState([]);
  const [selectedTimetable, setSelectedTimetable] = useState(null);
  const [isDeleteDialogOpen, setIsDeleteDialogOpen] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [apiAvailable, setApiAvailable] = useState(false);

  // Fetch timetable history from API
  const fetchTimetableHistory = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Check if API is available
      const isAvailable = await apiUtils.isApiAvailable();
      setApiAvailable(isAvailable);
      
      if (!isAvailable) {
        throw new Error('Backend API is not available. Please ensure the server is running.');
      }

      // Fetch all timetables
      const response = await timetableApi.getAll();
      const timetableIds = response.timetable_ids || [];
      
      // Fetch details for each timetable
      const timetableDetails = await Promise.all(
        timetableIds.map(async (id) => {
          try {
            const details = await timetableApi.getById(id);
            return {
              id: id,
              name: details.name || `Timetable ${id.slice(-8)}`,
              description: details.description || '',
              created_at: details.created_at || new Date().toISOString()
            };
          } catch (err) {
            console.warn(`Failed to fetch details for timetable ${id}:`, err);
            return {
              id: id,
              name: `Timetable ${id.slice(-8)}`,
              description: 'Details unavailable',
              created_at: new Date().toISOString()
            };
          }
        })
      );
      
      // Sort by creation date (newest first)
      timetableDetails.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
      
      setHistory(timetableDetails);
    } catch (err) {
      console.error('Error fetching timetable history:', err);
      setError(apiUtils.handleError(err, 'Failed to load timetable history'));
      setHistory([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchTimetableHistory();
  }, []);

  const handleDelete = async () => {
    if (!selectedTimetable) return;
    
    try {
      await timetableApi.delete(selectedTimetable);
      setHistory((prev) => prev.filter((item) => item.id !== selectedTimetable));
      setIsDeleteDialogOpen(false);
      setSelectedTimetable(null);
    } catch (err) {
      console.error('Error deleting timetable:', err);
      setError(apiUtils.handleError(err, 'Failed to delete timetable'));
    }
  };

  const handlePrint = (id) => {
    // Navigate to view page for printing
    router.push(`/view?id=${id}`);
  };

  const handleView = (id) => {
    router.push(`/view?id=${id}`);
  };

  const handleModify = (id) => {
    router.push(`/create?id=${id}`);
  };

  return (
    <div className="p-6 max-w-4xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">📅 SchedulAI</h1>
          <p className="text-gray-600">
            Welcome to SchedulAI! Easily create and manage timetables for faculties, courses, and classrooms.
          </p>
        </div>
        <div className="flex space-x-2">
          <Button
            variant="outline"
            onClick={fetchTimetableHistory}
            disabled={loading}
            className="flex items-center space-x-2"
          >
            <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
            <span>Refresh</span>
          </Button>
          <Button onClick={() => router.push("/create")}>
            Create New
          </Button>
        </div>
      </div>

      {/* API Status */}
      {!apiAvailable && (
        <Alert className="mb-6">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            Backend API is not available. Please ensure the server is running on {apiUtils.getBaseUrl()}
          </AlertDescription>
        </Alert>
      )}

      {/* Error Message */}
      {error && (
        <Alert variant="destructive" className="mb-6">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Loading State */}
      {loading ? (
        <div className="flex items-center justify-center py-8">
          <Loader2 className="h-8 w-8 animate-spin mr-2" />
          <span>Loading timetables...</span>
        </div>
      ) : (
        /* Timetable History */
        <div className="mt-6">
          <h2 className="text-lg font-semibold mb-4">Timetable History</h2>
          {history.length === 0 ? (
            <div className="text-center py-8">
              <div className="text-gray-400 mb-4">
                <svg className="mx-auto h-12 w-12" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v10a2 2 0 002 2h8a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                </svg>
              </div>
              <p className="text-gray-500 mb-4">No timetables found.</p>
              <Button onClick={() => router.push("/create")}>
                Create Your First Timetable
              </Button>
            </div>
          ) : (
            <div className="space-y-3">
              {history.map((item) => (
                <div
                  key={item.id}
                  className="p-4 bg-white border border-gray-200 rounded-lg shadow-sm hover:shadow-md transition-shadow"
                >
                  <div className="flex justify-between items-start">
                    <div className="flex-1">
                      <h3 className="font-medium text-gray-900">{item.name}</h3>
                      {item.description && (
                        <p className="text-sm text-gray-600 mt-1">{item.description}</p>
                      )}
                      <p className="text-xs text-gray-500 mt-2">
                        Created: {new Date(item.created_at).toLocaleDateString()}
                      </p>
                    </div>
                    <div className="flex space-x-2 ml-4">
                      <Button onClick={() => handleView(item.id)} size="sm" variant="outline">
                        View
                      </Button>
                      <Button onClick={() => handleModify(item.id)} size="sm">
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
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

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
