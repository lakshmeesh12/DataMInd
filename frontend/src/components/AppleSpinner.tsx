// src/components/AppleSpinner.tsx
import React from 'react';

// We will create the CSS for this in the next step
import '@/styles/spinner.css'; 

interface AppleSpinnerProps {
  size?: 'sm' | 'md' | 'lg';
}

const AppleSpinner: React.FC<AppleSpinnerProps> = ({ size = 'md' }) => {
  return (
    <div className={`apple-spinner ${size}`}>
      <div className="bar bar-1"></div>
      <div className="bar bar-2"></div>
      <div className="bar bar-3"></div>
      <div className="bar bar-4"></div>
      <div className="bar bar-5"></div>
      <div className="bar bar-6"></div>
      <div className="bar bar-7"></div>
      <div className="bar bar-8"></div>
      <div className="bar bar-9"></div>
      <div className="bar bar-10"></div>
      <div className="bar bar-11"></div>
      <div className="bar bar-12"></div>
    </div>
  );
};

export default AppleSpinner;