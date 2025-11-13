// src/lib/api.ts
import axios from 'axios';
import { loadFiles, saveFiles } from './storage';

// Set your FastAPI backend URL
const API_BASE_URL = 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
});

// --- Types ---

// --- Types ---

export interface FileMetadata {
  id: string;
  name: string;
  size: number;
  uploadDate: number;
  sheetsCount?: number;
  status: 'ready' | 'processing';
}

// --- NEW VISUALIZATION TYPES ---
export type VisualizationType = 'table' | 'bar_chart' | 'pie_chart';

export interface ChartData {
  labels: string[];
  datasets: {
    label: string;
    data: number[];
    backgroundColor?: string | string[]; // For charts
    borderColor?: string | string[];
  }[];
}

export interface TableData {
  headers: string[];
  rows: (string | number)[][];
}

export interface VisualizationData {
  type: VisualizationType;
  title: string;
  data: ChartData | TableData;
}
// --- END NEW VISUALIZATION TYPES ---


export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string; // This will hold the text part
  timestamp: number;
  visualization?: VisualizationData; // <-- NEW FIELD
}

export interface IngestResponse {
  status: 'success' | 'failed';
  filename: string;
  document_id: string;
  size: number;
  sheets_count: number;
  message: string;
}

// This interface matches the JSON object from your /search endpoint
export interface SearchResponse {
  answer: string; // The text answer
  visualization: VisualizationData | null; // <-- MODIFIED FIELD
  citations: string[];
  confidence: 'high' | 'medium' | 'low';
  completeness: 'full' | 'partial';
  matched_context: string;
  sections_covered: string[];
  total_extractions: number;
  query_analysis: {
    query_type: string;
    document_context_keywords: string[];
    section_keywords: string[];
    filter_keywords: string[];
    expected_format: string;
    search_strategy: string;
    estimated_scope: string;
  };
}


// --- API Functions ---

/**
 * Uploads Excel files to the /ingest/excel endpoint.
 */
export const uploadExcelFiles = async (
  files: File[],
  onUploadProgress: (progress: number) => void
): Promise<IngestResponse[]> => {
  const formData = new FormData();
  files.forEach((file) => {
    formData.append('files', file);
  });

  try {
    const response = await apiClient.post<IngestResponse[]>('/ingest/excel', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        if (progressEvent.total) {
          const percentCompleted = Math.round(
            (progressEvent.loaded * 100) / progressEvent.total
          );
          onUploadProgress(percentCompleted);
        }
      },
    });
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      console.error('API Error:', error.response?.data);
      throw new Error(error.response?.data?.detail || 'Upload failed');
    } else {
      console.error('Network Error:', error);
      throw new Error('Upload failed. Please check your connection.');
    }
  }
};

/**
 * Calls the /search endpoint.
 * Returns the full JSON response object.
 */
export const searchAPI = async (query: string): Promise<SearchResponse> => {
  try {
    const response = await apiClient.get<SearchResponse>('/search', {
      params: { q: query },
    });
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      console.error('Search API Error:', error.response?.data);
      throw new Error(error.response?.data?.detail || 'Query failed');
    } else {
      console.error('Network Error:', error);
      throw new Error('Query failed. Please check your connection.');
    }
  }
};

// ... (deleteFileAPI is the same) ...
export const deleteFileAPI = async (fileId: string): Promise<void> => {
    console.warn(`Attempting to delete file ${fileId} via API...`);
    try {
      const files = loadFiles();
      const filtered = files.filter(f => f.id !== fileId);
      saveFiles(filtered);
    } catch (error) {
       if (axios.isAxiosError(error)) {
        console.error('Delete API Error:', error.response?.data);
        throw new Error(error.response?.data?.detail || 'Could not delete file');
      } else {
        console.error('Network Error:', error);
        throw new Error('Could not delete file. Please check your connection.');
      }
    }
};