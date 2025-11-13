// src/pages/Dashboard.tsx
import { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import TextareaAutosize from 'react-textarea-autosize';
import { useToast } from '@/hooks/use-toast';
import { getCurrentUser, logout } from '@/lib/auth';
import { searchAPI, deleteFileAPI, Message, FileMetadata, SearchResponse } from '@/lib/api';
import { loadFiles } from '@/lib/storage'; // Only files
import ChatMessage from '@/components/ChatMessage';
import TypingIndicator from '@/components/TypingIndicator';
import FileUploadModal from '@/components/FileUploadModal';
import FilesSidebar from '@/components/FilesSidebar';
import {
  Send,
  Trash2,
  LogOut,
  Menu,
  X,
  ChevronLeft,
  ChevronRight,
} from 'lucide-react';
import { Avatar, AvatarFallback } from '@/components/ui/avatar';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';

const Dashboard = () => {
  const navigate = useNavigate();
  const { toast } = useToast();

  // Messages are in-memory only (cleared on refresh)
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [files, setFiles] = useState<FileMetadata[]>([]);
  const [uploadModalOpen, setUploadModalOpen] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const [autoScroll, setAutoScroll] = useState(true);
  const [hasNew, setHasNew] = useState(false);
  const user = getCurrentUser();
  const [loadingMessage, setLoadingMessage] = useState('Thinking...');
  const loadingTimerRef = useRef<NodeJS.Timeout | null>(null);

  /* ---------------------- Lifecycle ---------------------- */
  useEffect(() => {
    if (!user) {
      navigate('/login');
      return;
    }
    // Only load files — messages stay empty on refresh
    setFiles(loadFiles());
  }, [user, navigate]);

  useEffect(() => {
    if (autoScroll && messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, autoScroll]);

  useEffect(() => {
    if (loading) {
      setLoadingMessage('Analyzing the query...');
      if (loadingTimerRef.current) clearTimeout(loadingTimerRef.current);
      loadingTimerRef.current = setTimeout(() => setLoadingMessage('Thinking...'), 10_000);
    } else {
      if (loadingTimerRef.current) clearTimeout(loadingTimerRef.current);
    }
    return () => {
      if (loadingTimerRef.current) clearTimeout(loadingTimerRef.current);
    };
  }, [loading]);

  /* ---------------------- Scroll ---------------------- */
  const handleScroll = () => {
    if (!messagesContainerRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = messagesContainerRef.current;
    const isAtBottom = scrollHeight - scrollTop - clientHeight < 10;
    setAutoScroll(isAtBottom);
    if (isAtBottom) setHasNew(false);
  };

  /* ---------------------- Send ---------------------- */
  const handleSend = async () => {
    if (!input.trim() || loading) return;

    const userMessage: Message = {
      id: `msg_${Date.now()}`,
      role: 'user',
      content: input.trim(),
      timestamp: Date.now(),
    };

    const updatedMessages = [...messages, userMessage];
    setMessages(updatedMessages);
    // Messages are NOT saved to localStorage

    const currentQuery = input.trim();
    setInput('');
    setLoading(true);
    setAutoScroll(true);

    try {
      const response: SearchResponse = await searchAPI(currentQuery);

      let finalContent = response.answer;
      if (response.citations?.length) {
        finalContent += '\n\n---\n### Citations\n';
        finalContent += response.citations.map((c) => `* \`${c}\``).join('\n');
      }
      finalContent += `\n\n**Confidence:** ${response.confidence} | **Completeness:** ${response.completeness}`;

      const assistantMessage: Message = {
        id: `msg_${Date.now()}_assistant`,
        role: 'assistant',
        content: finalContent,
        visualization: response.visualization || undefined,
        timestamp: Date.now(),
      };

      const finalMessages = [...updatedMessages, assistantMessage];
      if (!autoScroll) setHasNew(true);
      setMessages(finalMessages);
      // No persistence
    } catch (error: any) {
      toast({
        title: 'Query failed',
        description: error.message || 'Please try again',
        variant: 'destructive',
      });
      const errorMessage: Message = {
        id: `msg_${Date.now()}_error`,
        role: 'assistant',
        content: `Sorry, I ran into an error: ${error.message}`,
        timestamp: Date.now(),
      };
      setMessages([...updatedMessages, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  /* ---------------------- Handlers ---------------------- */
  const handleClearChat = () => {
    setMessages([]);
    toast({ title: 'Chat cleared', description: 'Conversation history has been cleared' });
  };

  const handleLogout = () => {
    logout();
    navigate('/login');
    toast({ title: 'Logged out', description: 'See you next time!' });
  };

  const handleDeleteFile = async (fileId: string) => {
    try {
      await deleteFileAPI(fileId);
      setFiles(loadFiles());
      toast({ title: 'File deleted', description: 'File has been removed' });
    } catch (error: any) {
      toast({
        title: 'Delete failed',
        description: error.message || 'Could not delete file',
        variant: 'destructive',
      });
    }
  };

  const handleUploadComplete = () => setFiles(loadFiles());

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    setAutoScroll(true);
    setHasNew(false);
  };

  if (!user) return null;

  return (
    <div className="flex h-screen w-full flex-col bg-background">
      {/* ---------------------- Header ---------------------- */}
      <header className="flex h-14 items-center justify-between border-b bg-card px-4 shadow-sm">
        <div className="flex items-center gap-3">
          {/* Mobile menu toggle */}
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="lg:hidden"
          >
            {sidebarOpen ? <X className="h-4 w-4" /> : <Menu className="h-4 w-4" />}
          </Button>

          {/* Larger Horizontal Logo */}
          <img
            src="/assets/Quadrant Horizontal Logo@4x.png"
            alt="DataMind"
            className="h-12"
          />
        </div>

        {/* User dropdown */}
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" className="relative h-8 w-8 rounded-full">
              <Avatar className="h-8 w-8">
                <AvatarFallback className="bg-primary text-xs text-primary-foreground">
                  {user.name.charAt(0).toUpperCase()}
                </AvatarFallback>
              </Avatar>
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="w-48">
            <div className="px-2 py-1.5">
              <p className="text-xs font-medium">{user.name}</p>
              <p className="text-xs text-muted-foreground">{user.email}</p>
            </div>
            <DropdownMenuSeparator />
            <DropdownMenuItem onClick={handleLogout} className="text-xs">
              <LogOut className="mr-2 h-3.5 w-3.5" />
              Log out
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </header>

      {/* ---------------------- Main Content ---------------------- */}
      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar */}
        <div
          className={`${
            sidebarOpen ? 'w-64' : 'w-0'
          } transition-all duration-300 overflow-hidden hidden lg:block`}
        >
          {sidebarOpen && (
            <FilesSidebar
              files={files}
              onDeleteFile={handleDeleteFile}
              onUploadClick={() => setUploadModalOpen(true)}
            />
          )}
        </div>

        {/* Chat Area */}
        <div className="flex flex-1 flex-col overflow-hidden relative">
          {/* Sidebar Collapse Button */}
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="absolute left-0 top-4 z-20 hidden lg:flex h-8 w-8 rounded-r-md bg-card shadow-md"
          >
            {sidebarOpen ? (
              <ChevronLeft className="h-4 w-4" />
            ) : (
              <ChevronRight className="h-4 w-4" />
            )}
          </Button>

          {/* Messages */}
          <div
            ref={messagesContainerRef}
            className="flex-1 overflow-y-auto p-4 pl-12 lg:pl-4"
            onScroll={handleScroll}
            onWheel={() => setAutoScroll(false)}
            onTouchStart={() => setAutoScroll(false)}
          >
            <div className="mx-auto max-w-3xl space-y-4">
              {/* Welcome Screen */}
              {messages.length === 0 && !loading && (
                <div className="flex h-full flex-col items-center justify-center py-16 text-center">
                  <div className="mb-4 rounded-2xl bg-primary/10 p-4">
                    <img
                      src="/assets/Logo Icon.png"
                      alt="DataMind"
                      className="h-10 w-10"
                    />
                  </div>
                  <h2 className="mb-2 text-lg font-semibold">Welcome to DataMind</h2>
                  <p className="mb-6 max-w-md text-sm text-muted-foreground">
                    Upload your Excel, PDF, or any data files and ask questions to get instant insights.
                  </p>
                  <div className="space-y-1.5 text-left">
                    <p className="text-xs font-medium">Try asking:</p>
                    <div className="space-y-0.5 text-xs text-muted-foreground">
                      <p>• "Show me all revenue from Q3"</p>
                      <p>• "What are the top 5 products by sales?"</p>
                      <p>• "Summarize the project budget"</p>
                    </div>
                  </div>
                </div>
              )}

              {messages.map((msg) => (
                <ChatMessage key={msg.id} message={msg} />
              ))}

              {loading && <TypingIndicator message={loadingMessage} />}
              <div ref={messagesEndRef} />
            </div>
          </div>

          {/* Jump to latest */}
          {!autoScroll && hasNew && (
            <div className="pointer-events-auto fixed bottom-24 right-6 z-10">
              <Button size="sm" variant="secondary" onClick={scrollToBottom} className="shadow">
                Jump to latest
              </Button>
            </div>
          )}

          {/* Input Bar */}
          <div className="border-t bg-card p-3 shadow-lg">
            <div className="mx-auto flex max-w-3xl items-end gap-2">
              <Button
                variant="outline"
                size="icon"
                onClick={() => setUploadModalOpen(true)}
                className="h-9 w-9 shrink-0"
                title="Upload files"
              >
                <img src="/assets/Logo Icon.png" alt="Upload" className="h-4 w-4" />
              </Button>

              <TextareaAutosize
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSend();
                  }
                }}
                placeholder="Ask about your data..."
                className="flex w-full resize-none rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
                maxRows={8}
                rows={1}
                disabled={loading}
              />

              {messages.length > 0 && (
                <Button
                  variant="outline"
                  size="icon"
                  onClick={handleClearChat}
                  className="h-9 w-9 shrink-0"
                  title="Clear chat"
                  disabled={loading}
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
              )}

              <Button
                onClick={handleSend}
                disabled={!input.trim() || loading}
                className="h-9 w-9 shrink-0"
                size="icon"
              >
                <Send className="h-4 w-4" />
              </Button>
            </div>

            <p className="mx-auto mt-1.5 max-w-3xl text-center text-xs text-muted-foreground">
              {files.length} file{files.length !== 1 ? 's' : ''} available
            </p>
          </div>
        </div>
      </div>

      <FileUploadModal
        open={uploadModalOpen}
        onClose={() => setUploadModalOpen(false)}
        onUploadComplete={handleUploadComplete}
      />
    </div>
  );
};

export default Dashboard;