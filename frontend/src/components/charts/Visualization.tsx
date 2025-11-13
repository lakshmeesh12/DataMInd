// src/components/charts/Visualization.tsx
import { VisualizationData, ChartData, TableData } from '@/lib/api';
import DynamicBarChart from './DynamicBarChart';
import DynamicPieChart from './DynamicPieChart';
import DataTable from './DataTable';

interface VisualizationProps {
  visData: VisualizationData;
}

const Visualization = ({ visData }: VisualizationProps) => {
  const { type, data, title } = visData;

  switch (type) {
    case 'table':
      return <DataTable data={data as TableData} />;
    case 'bar_chart':
      return <DynamicBarChart data={data as ChartData} title={title} />;
    case 'pie_chart':
      return <DynamicPieChart data={data as ChartData} title={title} />;
    default:
      console.warn('Unknown visualization type:', type);
      return null;
  }
};

export default Visualization;