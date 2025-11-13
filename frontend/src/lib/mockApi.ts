// Mock API responses for BOQ queries

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: number;
}

export interface FileMetadata {
  id: string;
  name: string;
  size: number;
  uploadDate: number;
  status: 'processing' | 'ready' | 'error';
  sheetsCount?: number;
}

const LOADING_MESSAGES = [
  'Analyzing your query...',
  'Searching through documents...',
  'Processing BOQ data...',
  'Cross-referencing sheets...',
  'Finalizing answer...',
];

const SAMPLE_RESPONSES = [
  {
    keywords: ['flooring', 'floor', 'tile'],
    response: `Based on your BOQ files, here are the flooring items found:\n\n**5F Building - Flooring Items:**\n- Ceramic tiles (60x60 cm): 250 m² @ $45/m² = $11,250\n- Porcelain tiles (80x80 cm): 180 m² @ $62/m² = $11,160\n- Marble flooring: 95 m² @ $120/m² = $11,400\n\n**Total Flooring Cost:** $33,810\n\n*Data extracted from Sheet: "5F-Finishes", Rows 23-28*`,
  },
  {
    keywords: ['painting', 'paint', 'wall'],
    response: `**Painting Rates Summary:**\n\nInterior walls:\n- Primer + 2 coats emulsion: $8.50/m²\n- Premium paint (Dulux): $12.00/m²\n\nExterior walls:\n- Weather-resistant paint: $15.00/m²\n- Textured finish: $18.50/m²\n\n*These rates include labor and materials. Source: "Rates-Master" sheet, updated Q1 2024.*`,
  },
  {
    keywords: ['electrical', 'wiring', 'circuit'],
    response: `**Electrical Installation Details:**\n\n- Main distribution board (400A): 1 unit @ $2,800\n- Sub-distribution boards: 4 units @ $850 each = $3,400\n- Conduit wiring (PVC): 2,400 m @ $5.50/m = $13,200\n- LED fixtures: 180 units @ $65 each = $11,700\n- Emergency lights: 24 units @ $95 each = $2,280\n\n**Subtotal:** $33,380\n\n*Reference: "Electrical-BOQ" sheet*`,
  },
  {
    keywords: ['plumbing', 'pipe', 'sanitary'],
    response: `**Plumbing & Sanitary Summary:**\n\nWater supply:\n- CPVC pipes (various sizes): $8,900\n- Fixtures (taps, valves): $4,200\n\nDrainage:\n- PVC drainage pipes: $6,500\n- Sanitary ware (WC, basins): $12,400\n\n**Total Plumbing Cost:** $32,000\n\n*Confidence: High | Source: "Plumbing-5F" sheet*`,
  },
];

export const mockQueryAPI = async (query: string): Promise<string> => {
  // Simulate network delay
  await new Promise(resolve => setTimeout(resolve, 1500));
  
  const lowerQuery = query.toLowerCase();
  
  // Find matching response
  const match = SAMPLE_RESPONSES.find(r => 
    r.keywords.some(keyword => lowerQuery.includes(keyword))
  );
  
  if (match) {
    return match.response;
  }
  
  // Default response
  return `I found information related to your query in the BOQ files:\n\n**Summary:**\nYour construction project contains detailed specifications across multiple sheets including structural work, finishes, MEP (Mechanical, Electrical, Plumbing), and external works.\n\n**Suggested queries:**\n- "Show me all flooring items in building 5F"\n- "What are the painting rates?"\n- "List electrical installation costs"\n- "Summarize plumbing requirements"\n\n*Try being more specific about the item, location, or category you're interested in.*`;
};

export const getLoadingMessage = (index: number): string => {
  return LOADING_MESSAGES[index % LOADING_MESSAGES.length];
};

export const mockFileUpload = async (files: File[]): Promise<FileMetadata[]> => {
  // Simulate upload delay
  await new Promise(resolve => setTimeout(resolve, 2000));
  
  return files.map(file => ({
    id: `file_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
    name: file.name,
    size: file.size,
    uploadDate: Date.now(),
    status: 'ready' as const,
    sheetsCount: Math.floor(Math.random() * 8) + 3,
  }));
};
