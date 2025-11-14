// src/components/VisualizationModal.tsx
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { FileMetadata } from '@/lib/api';
import {
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import CytoscapeComponent from 'react-cytoscapejs';
import AppleSpinner from './AppleSpinner';
import { X } from 'lucide-react';

interface VisualizationModalProps {
  open: boolean;
  onClose: () => void;
  file: FileMetadata;
  visualizationData: any;   // from visualizeDocument()
  loading: boolean;
  error: string | null;
}

const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899'];

const VisualizationModal = ({
  open,
  onClose,
  file,
  visualizationData,
  loading,
  error,
}: VisualizationModalProps) => {
  const layout = { name: 'cose', animate: true, fit: true, padding: 20 };

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="max-w-6xl max-h-[90vh] overflow-y-auto p-0">
        {/* Header */}
        <DialogHeader className="sticky top-0 z-10 border-b bg-background p-4">
          <DialogTitle className="flex items-center justify-between">
            <span>Visualization: {file.name}</span>
            <Button variant="ghost" size="icon" onClick={onClose}>
              <X className="h-4 w-4" />
            </Button>
          </DialogTitle>
        </DialogHeader>

        {/* Body */}
        <div className="p-6 space-y-8">
          {/* Loading */}
          {loading && (
            <div className="flex flex-col items-center justify-center py-16">
              <AppleSpinner size="lg" />
              <p className="mt-4 text-sm text-muted-foreground">Generating visualization…</p>
            </div>
          )}

          {/* Error */}
          {error && !loading && (
            <div className="text-center py-16 text-destructive">{error}</div>
          )}

          {/* Charts – only render when data exists */}
          {visualizationData && !loading && !error && (
            <>
              {/* 1. Cost Breakdown – Pie */}
              {visualizationData.visualizations?.cost_breakdown_pie && (
                <div>
                  <h3 className="mb-3 text-lg font-semibold">
                    {visualizationData.visualizations.cost_breakdown_pie.title}
                  </h3>
                  <ResponsiveContainer width="100%" height={300}>
                    <PieChart>
                      <Pie
                        data={visualizationData.visualizations.cost_breakdown_pie.data.datasets[0].data.map(
                          (d: number, i: number) => ({
                            name: visualizationData.visualizations.cost_breakdown_pie.data.labels[i],
                            value: d,
                          })
                        )}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                        outerRadius={80}
                        dataKey="value"
                      >
                        {visualizationData.visualizations.cost_breakdown_pie.data.datasets[0].data.map(
                          (_: any, idx: number) => (
                            <Cell key={`cell-${idx}`} fill={COLORS[idx % COLORS.length]} />
                          )
                        )}
                      </Pie>
                      <Tooltip />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              )}

              {/* 2. Cost by Section – Bar */}
              {visualizationData.visualizations?.cost_by_section_bar && (
                <div>
                  <h3 className="mb-3 text-lg font-semibold">
                    {visualizationData.visualizations.cost_by_section_bar.title}
                  </h3>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart
                      data={visualizationData.visualizations.cost_by_section_bar.data.labels.map(
                        (label: string, i: number) => ({
                          name: label,
                          cost: visualizationData.visualizations.cost_by_section_bar.data.datasets[0].data[i],
                        })
                      )}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" />
                      <YAxis />
                      <Tooltip />
                      <Bar dataKey="cost" fill="#3b82f6" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )}

              {/* 3. Item Type – Doughnut */}
              {visualizationData.visualizations?.item_type_doughnut && (
                <div>
                  <h3 className="mb-3 text-lg font-semibold">
                    {visualizationData.visualizations.item_type_doughnut.title}
                  </h3>
                  <ResponsiveContainer width="100%" height={300}>
                    <PieChart>
                      <Pie
                        data={visualizationData.visualizations.item_type_doughnut.data.datasets[0].data.map(
                          (d: number, i: number) => ({
                            name: visualizationData.visualizations.item_type_doughnut.data.labels[i],
                            value: d,
                          })
                        )}
                        cx="50%"
                        cy="50%"
                        innerRadius={60}
                        outerRadius={80}
                        dataKey="value"
                      >
                        {visualizationData.visualizations.item_type_doughnut.data.datasets[0].data.map(
                          (_: any, idx: number) => (
                            <Cell key={`cell-${idx}`} fill={COLORS[idx % COLORS.length]} />
                          )
                        )}
                      </Pie>
                      <Tooltip />
                      <Legend />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              )}

              {/* 4. Knowledge Graph – safe iteration */}
              {visualizationData.visualizations?.knowledge_graph && (
                <div>
                  <h3 className="mb-3 text-lg font-semibold">
                    {visualizationData.visualizations.knowledge_graph.title}
                  </h3>
                  <div className="border rounded-lg overflow-hidden" style={{ height: 400 }}>
                    <CytoscapeComponent
                      elements={[
                        ...(Array.isArray(visualizationData.visualizations.knowledge_graph.nodes)
                          ? visualizationData.visualizations.knowledge_graph.nodes
                          : []),
                        ...(Array.isArray(visualizationData.visualizations.knowledge_graph.edges)
                          ? visualizationData.visualizations.knowledge_graph.edges
                          : []),
                      ]}
                      style={{ width: '100%', height: '100%' }}
                      layout={layout}
                      stylesheet={[
                        {
                          selector: 'node',
                          style: {
                            'background-color': '#3b82f6',
                            label: 'data(label)',
                            color: '#fff',
                            'text-valign': 'center',
                            'text-halign': 'center',
                            'font-size': 10,
                            width: 20,
                            height: 20,
                          },
                        },
                        {
                          selector: 'edge',
                          style: {
                            width: 1,
                            'line-color': '#ccc',
                            'target-arrow-color': '#ccc',
                            'target-arrow-shape': 'triangle',
                            'curve-style': 'bezier',
                          },
                        },
                      ]}
                    />
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default VisualizationModal;