// src/lib/storage.ts

import { FileMetadata } from './api';

const FILES_KEY = 'boq_files';

export const saveFiles = (files: FileMetadata[]) => {
  localStorage.setItem(FILES_KEY, JSON.stringify(files));
};

export const loadFiles = (): FileMetadata[] => {
  const stored = localStorage.getItem(FILES_KEY);
  if (!stored) return [];
  try {
    return JSON.parse(stored);
  } catch {
    return [];
  }
};

export const addFile = (file: FileMetadata) => {
  const files = loadFiles();
  files.push(file);
  saveFiles(files);
};

export const deleteFile = (fileId: string) => {
  const files = loadFiles();
  const filtered = files.filter(f => f.id !== fileId);
  saveFiles(filtered);
};