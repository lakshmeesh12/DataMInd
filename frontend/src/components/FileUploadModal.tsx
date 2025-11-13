// src/components/FileModal.tsx
import { useState, useCallback } from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { useToast } from '@/hooks/use-toast';
import { uploadExcelFiles, FileMetadata, IngestResponse } from '@/lib/api';
import { saveFiles, loadFiles } from '@/lib/storage';
import { Upload, X, FileSpreadsheet } from 'lucide-react'; // Removed Loader2
import AppleSpinner from './AppleSpinner'; // Import the spinner

interface FileUploadModalProps {
  open: boolean;
  onClose: () => void;
  onUploadComplete: () => void;
}

const FileUploadModal = ({ open, onClose, onUploadComplete }: FileUploadModalProps) => {
  const [files, setFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0); // We still use this to track status
  const [dragActive, setDragActive] = useState(false);
  const { toast } = useToast();

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    const droppedFiles = Array.from(e.dataTransfer.files).filter(
      (file) => file.name.endsWith('.xlsx') || file.name.endsWith('.xls')
    );

    if (droppedFiles.length === 0) {
      toast({
        title: 'Invalid files',
        description: 'Please upload Excel files (.xlsx, .xls) only',
        variant: 'destructive',
      });
      return;
    }

    setFiles((prev) => [...prev, ...droppedFiles]);
  }, [toast]);

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const selectedFiles = Array.from(e.target.files);
      setFiles((prev) => [...prev, ...selectedFiles]);
    }
  };

  const removeFile = (index: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const handleUpload = async () => {
    if (files.length === 0) return;

    const totalSize = files.reduce((sum, file) => sum + file.size, 0);
    if (totalSize > 50 * 1024 * 1024) { // 50MB limit
      toast({
        title: 'Files too large',
        description: 'Total size must be under 50MB',
        variant: 'destructive',
      });
      return;
    }

    setUploading(true);
    setProgress(0);

    try {
      const results = await uploadExcelFiles(files, (percent) => {
        setProgress(percent);
      });

      const successfulUploads: IngestResponse[] = [];
      const failedUploads: IngestResponse[] = [];
      
      results.forEach(res => {
        if (res.status === 'success') {
          successfulUploads.push(res);
        } else {
          failedUploads.push(res);
        }
      });

      const newFilesForStorage: FileMetadata[] = successfulUploads.map((res) => ({
        id: res.document_id,
        name: res.filename,
        size: res.size,
        uploadDate: Date.now(),
        sheetsCount: res.sheets_count,
        status: 'ready',
      }));

      if (newFilesForStorage.length > 0) {
        const existingFiles = loadFiles();
        saveFiles([...existingFiles, ...newFilesForStorage]);
      }

      if (failedUploads.length === 0) {
        toast({
          title: 'Upload complete!',
          description: `${successfulUploads.length} file${
            successfulUploads.length > 1 ? 's' : ''
          } uploaded and processed.`,
        });
      } else {
        toast({
          title: 'Partial upload',
          description: `${successfulUploads.length} succeeded, ${failedUploads.length} failed.`,
          variant: 'destructive',
        });
      }

      // We still call onClose, but the state change will happen first
      // The finally block will set uploading to false, allowing modal to close
      onUploadComplete();
      onClose();
      setFiles([]);
      setProgress(0);

    } catch (error: any) {
      toast({
        title: 'Upload failed',
        description: error.message || 'An unknown error occurred. Please try again.',
        variant: 'destructive',
      });
    } finally {
      setUploading(false); // This will re-enable closing the modal
    }
  };

  // Prevents closing the modal while uploading
  const handleModalInteraction = (open: boolean) => {
    if (!open && uploading) {
      return; // Do nothing, prevent closing
    }
    if (!open) {
      onClose(); // Allow closing
    }
  };

  return (
    <Dialog open={open} onOpenChange={handleModalInteraction}>
      <DialogContent className="sm:max-w-lg">
        {/* We hide the header text during upload for a cleaner look */}
        {!uploading && (
          <DialogHeader>
            <DialogTitle>Upload Excel Files</DialogTitle>
            <DialogDescription>
              Upload your BOQ Excel files (.xlsx, .xls). Maximum 50MB total.
            </DialogDescription>
          </DialogHeader>
        )}

        <div className="space-y-4">
          {uploading ? (
            // --- UPLOADING STATE ---
            // Show only the big spinner and status text
            <div className="flex flex-col items-center justify-center space-y-3 py-16">
              <AppleSpinner size="lg" /> {/* <-- Made spinner larger */}
              <span className="text-sm font-medium text-muted-foreground">
                {progress < 100 ? 'Uploading files...' : 'Processing...'}
              </span>
              <span className="text-xs text-muted-foreground">Please wait, this may take a moment.</span>
            </div>
          ) : (
            // --- DEFAULT STATE ---
            // Show the drag-drop area and file list
            <>
              <div
                className={`relative rounded-xl border-2 border-dashed p-8 text-center transition-colors ${
                  dragActive ? 'border-primary bg-primary/5' : 'border-border'
                }`}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
              >
                <input
                  type="file"
                  multiple
                  accept=".xlsx,.xls"
                  onChange={handleFileInput}
                  className="absolute inset-0 cursor-pointer opacity-0"
                  disabled={uploading} // Always false here, but good practice
                />
                <Upload className="mx-auto mb-4 h-10 w-10 text-muted-foreground" />
                <p className="mb-2 font-medium">Drop files here or click to browse</p>
                <p className="text-sm text-muted-foreground">Excel files only (.xlsx, .xls)</p>
              </div>

              {files.length > 0 && (
                <div className="space-y-2">
                  <p className="text-sm font-medium">Selected files ({files.length}):</p>
                  <div className="max-h-40 space-y-2 overflow-y-auto rounded-lg border p-3">
                    {files.map((file, index) => (
                      <div key={index} className="flex items-center justify-between gap-2 text-sm">
                        <div className="flex items-center gap-2 overflow-hidden">
                          <FileSpreadsheet className="h-4 w-4 shrink-0 text-primary" />
                          <span className="truncate">{file.name}</span>
                          <span className="shrink-0 text-xs text-muted-foreground">
                            ({(file.size / 1024).toFixed(0)} KB)
                          </span>
                        </div>
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => removeFile(index)}
                          className="h-6 w-6 p-0"
                        >
                          <X className="h-4 w-4" />
                        </Button>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          )}

          {/* --- HIDE BUTTONS WHEN UPLOADING --- */}
          {!uploading && (
            <div className="flex gap-2">
              <Button variant="outline" onClick={onClose} className="flex-1">
                Cancel
              </Button>
              <Button
                onClick={handleUpload}
                disabled={files.length === 0}
                className="flex-1"
              >
                {`Upload ${files.length} file${files.length !== 1 ? 's' : ''}`}
              </Button>
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default FileUploadModal;