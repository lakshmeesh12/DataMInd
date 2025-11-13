// src/lib/formatters.ts
import { SearchResponse } from './api';

/**
 * Converts the raw JSON response from the /search API
 * into a human-friendly Markdown string.
 */
export const formatSearchResponse = (response: SearchResponse): string => {
  let md = '';

  // 1. The main answer (already Markdown)
  md += response.answer;

  // 2. Citations
  if (response.citations && response.citations.length > 0) {
    md += '\n\n---\n'; // Horizontal rule
    md += '### 📚 Citations\n';
    md += response.citations
      .map(citation => `* \`${citation}\``)
      .join('\n');
  }

  // 3. Details (Confidence & Completeness)
  md += '\n\n';
  md += `**Confidence:** ${response.confidence} | **Completeness:** ${response.completeness}`;

  // You can add more fields from response.query_analysis if needed
  // Example:
  // md += `\n\n#### Query Analysis\n`;
  // md += `* **Type:** ${response.query_analysis.query_type}\n`;
  // md += `* **Strategy:** ${response.query_analysis.search_strategy}\n`;

  return md;
};