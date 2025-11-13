// src/components/charts/DynamicPieChart.tsx
import { ChartData } from '@/lib/api';
import { Pie } from 'react-chartjs-2';
import { Chart as ChartJS, ArcElement, Tooltip, Legend, Title } from 'chart.js';

ChartJS.register(ArcElement, Tooltip, Legend, Title);

interface PieChartProps {
  data: ChartData;
  title: string;
}

// ... (defaultColors array is unchanged) ...
const defaultColors = [
  'rgba(255, 99, 132, 0.6)',
  'rgba(54, 162, 235, 0.6)',
  'rgba(255, 206, 86, 0.6)',
  'rgba(75, 192, 192, 0.6)',
  'rgba(153, 102, 255, 0.6)',
  'rgba(255, 159, 64, 0.6)',
];


const DynamicPieChart = ({ data, title }: PieChartProps) => {
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

  const processedData = {
    ...data,
    datasets: data.datasets.map((ds) => ({
      ...ds,
      backgroundColor: ds.backgroundColor || defaultColors,
      borderColor: ds.borderColor || '#ffffff',
    })),
  };

  return (
    <div className="my-2 max-h-[400px] rounded-lg border bg-card p-4">
      <Pie options={options} data={processedData} />
    </div>
  );
};

export default DynamicPieChart;