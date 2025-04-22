import React from 'react';
import { BrainCircuit } from 'lucide-react';

const Header: React.FC = () => {
  return (
    <header className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white py-4 px-6 shadow-md">
      <div className="container mx-auto flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <BrainCircuit className="h-8 w-8" />
          <h1 className="text-2xl font-bold tracking-tight">HealthGuru</h1>
        </div>
      </div>
    </header>
  );
};

export default Header;