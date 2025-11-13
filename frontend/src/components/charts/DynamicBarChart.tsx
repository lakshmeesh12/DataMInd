// src/components/charts/DynamicBarChart.tsx
import { ChartData } from '@/lib/api';
import { Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

interface BarChartProps {
  data: ChartData;
  title: string;
}

const DynamicBarChart = ({ data, title }: BarChartProps) => {
  const options = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top' as const,
      },
      title: {
        display: true,
        text: title,
        font: { size: 16 },
      },
      // --- THIS IS THE FIX ---
      tooltip: {
        animation: {
          duration: 0, // Disables the animation, making it instant
        },
      },
      // --- END FIX ---
    },
  };

  // Add default colors if not provided
  const processedData = {
    ...data,
    datasets: data.datasets.map((ds) => ({
      ...ds,
      backgroundColor: ds.backgroundColor || 'rgba(54, 162, 235, 0.6)',
      borderColor: ds.borderColor || 'rgba(54, 162, 235, 1)',
    })),
  };

  return (
    <div className="my-2 rounded-lg border bg-card p-4">
      <Bar options={options} data={processedData} />
    </div>
  );
};

export default DynamicBarChart;