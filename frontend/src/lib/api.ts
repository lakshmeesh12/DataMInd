// src/lib/api.ts
import axios from 'axios';
import { loadFiles, saveFiles } from './storage';

const API_BASE_URL = 'http://localhost:8000';
const apiClient = axios.create({ baseURL: API_BASE_URL });

// ----------------------------------------------------------------------
// Types
// ----------------------------------------------------------------------
export interface FileMetadata {
  id: string;
  name: string;
  size: number;
  uploadDate: number;
  sheetsCount?: number;
  status: 'ready' | 'processing';
}

export type VisualizationType = 'table' | 'bar_chart' | 'pie_chart';

export interface ChartData {
  labels: string[];
  datasets: {
    label: string;
    data: number[];
    backgroundColor?: string | string[];
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

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: number;
  visualization?: VisualizationData;
}

export interface IngestResponse {
  status: 'success' | 'failed';
  filename: string;
  document_id: string;
  size: number;
  sheets_count: number;
  message: string;
}

export interface SearchResponse {
  answer: string;
  visualization: VisualizationData | null;
  citations: string[];
  confidence: 'high' | 'medium' | 'low';
  completeness: 'full' | 'partial';
  matched_context?: string;
  sections_covered?: string[];
  total_extractions?: number;
  query_analysis?: any;
}

// ----------------------------------------------------------------------
// Upload
// ----------------------------------------------------------------------
export const uploadExcelFiles = async (
  files: File[],
  onUploadProgress: (progress: number) => void
): Promise<IngestResponse[]> => {
  const formData = new FormData();
  files.forEach((f) => formData.append('files', f));

  const response = await apiClient.post<IngestResponse[]>('/ingest/excel', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: (e) => {
      if (e.total) onUploadProgress(Math.round((e.loaded * 100) / e.total));
    },
  });
  return response.data;
};

// ----------------------------------------------------------------------
// DELETE (kept as-is)
// ----------------------------------------------------------------------
export const deleteFileAPI = async (fileId: string): Promise<void> => {
  const files = loadFiles();
  const filtered = files.filter((f) => f.id !== fileId);
  saveFiles(filtered);
};

// ----------------------------------------------------------------------
// STREAMING SEARCH (replaces the old searchAPI)
// ----------------------------------------------------------------------
export const searchAPI = async function* (
  query: string
): AsyncGenerator<
  { type: 'token'; content: string } | { type: 'finish'; payload: SearchResponse },
  void
> {
  const resp = await fetch(`${API_BASE_URL}/search-stream?q=${encodeURIComponent(query)}`);

  if (!resp.ok) {
    const txt = await resp.text();
    throw new Error(txt || 'Query failed');
  }

  const reader = resp.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop()!;

    for (const line of lines) {
      if (!line.trim()) continue;
      const obj = JSON.parse(line);
      if (obj.type === 'token') {
        yield obj;
      } else if (obj.type === 'finish') {
        yield { type: 'finish', payload: obj };
      }
    }
  }
};

export interface VisualizeResponse {
  document: {
    id: string;
    filename: string;
    uploaded_at: string;
    sheet_count: number;
  };
  visualizations: {
    cost_breakdown_pie?: any;
    cost_by_section_bar?: any;
    item_type_doughnut?: any;
    top_expensive_items_bar?: any;
    knowledge_graph?: { nodes: any[]; edges: any[] };
  };
  summary?: any;
}

/**
 * Calls `/visualize?doc_id=…` (or `filename=…`).
 * Returns the full JSON payload from the backend.
 */
export const visualizeDocument = async (
  docId?: string,
  filename?: string
): Promise<VisualizeResponse> => {
  if (!docId && !filename) {
    throw new Error('Either docId or filename is required');
  }

  const params: Record<string, string> = {};
  if (docId) params.doc_id = docId;
  if (filename) params.filename = filename;

  const response = await apiClient.get<VisualizeResponse>('/visualize', { params });
  return response.data;
};