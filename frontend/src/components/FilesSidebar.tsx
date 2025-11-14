// src/components/FilesSidebar.tsx
import { FileMetadata, visualizeDocument } from '@/lib/api';
import { FileSpreadsheet, Trash2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { useState } from 'react';
import VisualizationModal from './VisualizationModal';

interface FilesSidebarProps {
  files: FileMetadata[];
  onDeleteFile: (fileId: string) => void;
  onUploadClick: () => void;
}

const FilesSidebar = ({ files, onDeleteFile, onUploadClick }: FilesSidebarProps) => {
  const [selectedFile, setSelectedFile] = useState<FileMetadata | null>(null);
  const [visModalOpen, setVisModalOpen] = useState(false);
  const [visData, setVisData] = useState<any>(null);
  const [visLoading, setVisLoading] = useState(false);
  const [visError, setVisError] = useState<string | null>(null);

  const formatDate = (ts: number) =>
    new Date(ts).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });

  const formatSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const openVisualization = async (file: FileMetadata) => {
    setSelectedFile(file);
    setVisModalOpen(true);
    setVisLoading(true);
    setVisError(null);
    setVisData(null);

    try {
      const data = await visualizeDocument(file.id);
      setVisData(data);
    } catch (err: any) {
      setVisError(err.message || 'Failed to load visualization');
    } finally {
      setVisLoading(false);
    }
  };

  return (
    <>
      <div className="flex h-full w-64 flex-col border-r bg-sidebar">
        <div className="border-b px-4 py-3">
          <h2 className="text-sm font-semibold">Documents</h2>
          <p className="text-xs text-muted-foreground">
            {files.length} file{files.length !== 1 ? 's' : ''}
          </p>
        </div>

        <ScrollArea className="flex-1">
          <div className="space-y-1.5 p-3">
            {files.length === 0 ? (
              // Empty state
              <div className="rounded-lg border border-dashed p-6 text-center">
                <FileSpreadsheet className="mx-auto mb-2 h-8 w-8 text-muted-foreground" />
                <p className="text-xs font-medium text-muted-foreground">No files uploaded</p>
                <p className="mt-1 text-xs text-muted-foreground">Use the upload button below</p>
              </div>
            ) : (
              files.map((file) => (
                <div
                  key={file.id}
                  className="group rounded-lg border bg-card p-2.5 transition-colors hover:bg-accent"
                >
                  <div className="mb-1.5 flex items-start justify-between gap-2">
                    <div className="flex items-start gap-2 flex-1 min-w-0">
                      <FileSpreadsheet className="mt-0.5 h-3.5 w-3.5 shrink-0 text-primary" />
                      <div className="min-w-0 flex-1">
                        {/* Full filename – wraps to multiple lines */}
                        <p className="text-xs font-medium break-words leading-tight">{file.name}</p>
                        <div className="mt-0.5 flex flex-wrap items-center gap-1.5 text-xs text-muted-foreground">
                          <span>{formatSize(file.size)}</span>
                          <span>•</span>
                          <span>{formatDate(file.uploadDate)}</span>
                        </div>
                        {file.sheetsCount && (
                          <p className="mt-0.5 text-xs text-muted-foreground">
                            {file.sheetsCount} sheets
                          </p>
                        )}
                      </div>
                    </div>

                    <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                      {/* Visualize Button */}
                      <Button
                        size="icon"
                        variant="ghost"
                        className="h-6 w-6 p-0"
                        onClick={(e) => {
                          e.stopPropagation();
                          openVisualization(file);
                        }}
                        title="Visualize Data"
                      >
                        <img
                          src="/assets/visualize-icon.svg"
                          alt="Visualize"
                          className="h-4 w-4"
                        />
                      </Button>

                      {/* Delete Button */}
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={(e) => {
                          e.stopPropagation();
                          onDeleteFile(file.id);
                        }}
                        className="h-6 w-6 p-0"
                      >
                        <Trash2 className="h-3 w-3 text-destructive" />
                      </Button>
                    </div>
                  </div>

                  <div
                    className={`inline-flex items-center rounded-full px-1.5 py-0.5 text-xs font-medium ${
                      file.status === 'ready'
                        ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400'
                        : 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400'
                    }`}
                  >
                    {file.status === 'ready' ? 'Ready' : 'Processing'}
                  </div>
                </div>
              ))
            )}
          </div>
        </ScrollArea>

        <div className="border-t p-3">
          <Button onClick={onUploadClick} size="sm" className="h-8 w-full text-xs">
            <FileSpreadsheet className="mr-1.5 h-3.5 w-3.5" />
            Upload Files
          </Button>
        </div>
      </div>

      {/* Visualization Modal */}
      {selectedFile && (
        <VisualizationModal
          open={visModalOpen}
          onClose={() => setVisModalOpen(false)}
          file={selectedFile}
          visualizationData={visData}
          loading={visLoading}
          error={visError}
        />
      )}
    </>
  );
};

export default FilesSidebar;