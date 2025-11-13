// src/pages/Index.tsx
import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { isAuthenticated } from '@/lib/auth';
import { ArrowRight } from 'lucide-react';

const Index = () => {
  const navigate = useNavigate();

  useEffect(() => {
    if (isAuthenticated()) {
      navigate('/dashboard');
    }
  }, [navigate]);

  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-background px-4">
      <div className="w-full max-w-4xl text-center animate-fade-in">
        {/* Icon Logo - No Background */}
        <div className="mx-auto mb-6 flex h-20 w-20 items-center justify-center">
          <img
            src="/assets/Logo Icon.png"
            alt="DataMind"
            className="h-16 w-16 object-contain"
          />
        </div>

        {/* Title - Smaller */}
        <h1 className="mb-2 text-3xl font-bold tracking-tight">DataMind</h1>

        {/* Subtitle - Smaller */}
        <p className="mb-8 text-lg text-muted-foreground">
          A universal mind for structured & unstructured data
        </p>

        {/* Feature Cards - Smaller Text */}
        <div className="mb-10 grid gap-5 md:grid-cols-3">
          <div className="rounded-xl border bg-card p-5 shadow-sm">
            <div className="mb-3 flex h-11 w-11 items-center justify-center rounded-lg bg-primary/10">
              <img src="/assets/Logo Icon.png" alt="Upload" className="h-5 w-5 object-contain" />
            </div>
            <h3 className="mb-1.5 text-sm font-semibold">Upload Any File</h3>
            <p className="text-xs text-muted-foreground">
              Excel, PDF, Word, CSV — bring all your data in any format
            </p>
          </div>

          <div className="rounded-xl border bg-card p-5 shadow-sm">
            <div className="mb-3 flex h-11 w-11 items-center justify-center rounded-lg bg-primary/10">
              <svg className="h-5 w-5 text-primary" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
              </svg>
            </div>
            <h3 className="mb-1.5 text-sm font-semibold">Ask in Plain English</h3>
            <p className="text-xs text-muted-foreground">
              Query your data naturally — no SQL, no code, just ask
            </p>
          </div>

          <div className="rounded-xl border bg-card p-5 shadow-sm">
            <div className="mb-3 flex h-11 w-11 items-center justify-center rounded-lg bg-primary/10">
              <svg className="h-5 w-5 text-primary" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            </div>
            <h3 className="mb-1.5 text-sm font-semibold">Get Actionable Insights</h3>
            <p className="text-xs text-muted-foreground">
              Instant summaries, trends, and citations from your files
            </p>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex flex-col items-center gap-3 sm:flex-row sm:justify-center">
          <Button size="lg" className="h-10 px-5 text-sm gap-2" onClick={() => navigate('/signup')}>
            Get Started
            <ArrowRight className="h-3.5 w-3.5" />
          </Button>
          <Button size="lg" variant="outline" className="h-10 px-5 text-sm" onClick={() => navigate('/login')}>
            Sign In
          </Button>
        </div>

        {/* Footer */}
        <p className="mt-8 text-xs text-muted-foreground">
          Enterprise-grade data intelligence • Demo mode
        </p>
      </div>
    </div>
  );
};

export default Index;