// src/components/charts/DataTable.tsx
import { TableData } from '@/lib/api';

interface DataTableProps {
  data: TableData;
}

const DataTable = ({ data }: DataTableProps) => {
  return (
    <div className="my-2 rounded-lg border border-neutral-300 dark:border-neutral-700 overflow-hidden">
      <table className="w-full border-collapse">
        {/* ---------- HEADER ---------- */}
        <thead>
          <tr className="border-b border-neutral-300 dark:border-neutral-700 bg-muted/70">
            {data.headers.map((header, index) => (
              <th
                key={index}
                className="p-3 text-left text-xs font-medium uppercase tracking-wider text-muted-foreground
                           [&:not(:last-child)]:border-r [&:not(:last-child)]:border-neutral-300
                           dark:[&:not(:last-child)]:border-neutral-700"
              >
                {header}
              </th>
            ))}
          </tr>
        </thead>

        {/* ---------- BODY ---------- */}
        <tbody>
          {data.rows.map((row, rowIndex) => (
            <tr
              key={rowIndex}
              className="border-b border-neutral-200 dark:border-neutral-800 last:border-b-0 hover:bg-muted/30"
            >
              {row.map((cell, cellIndex) => (
                <td
                  key={cellIndex}
                  className="p-3 align-top text-xs text-foreground/90
                             [&:not(:last-child)]:border-r [&:not(:last-child)]:border-neutral-200
                             dark:[&:not(:last-child)]:border-neutral-800"
                >
                  {String(cell)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default DataTable;