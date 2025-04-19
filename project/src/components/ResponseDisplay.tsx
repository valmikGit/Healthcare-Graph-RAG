import React from 'react';

interface ResponseDisplayProps {
  response: string | null;
  isLoading: boolean;
  error: string | null;
}

const ResponseDisplay: React.FC<ResponseDisplayProps> = ({ response, isLoading, error }) => {
  if (isLoading) {
    return (
      <div className="w-full mt-6 bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700 min-h-[200px] animate-pulse">
        <div className="flex flex-col space-y-3">
          <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-3/4"></div>
          <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-1/2"></div>
          <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-5/6"></div>
          <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-2/3"></div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="w-full mt-6 bg-red-50 dark:bg-red-900/20 rounded-xl p-6 shadow-sm border border-red-200 dark:border-red-800/50 min-h-[100px] text-red-800 dark:text-red-300">
        <h3 className="font-medium mb-2">Error</h3>
        <p>{error}</p>
      </div>
    );
  }

  if (!response) {
    return (
      <div className="w-full mt-6 bg-blue-50 dark:bg-blue-900/10 rounded-xl p-6 shadow-sm border border-blue-100 dark:border-blue-800/30 min-h-[200px] flex items-center justify-center text-center text-gray-500 dark:text-gray-400">
        <p className="max-w-md">Ask a question above to get started. I'm here to help answer your questions as accurately as possible.</p>
      </div>
    );
  }

  return (
    <div className="w-full mt-6 bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700 min-h-[200px] animate-fade-in text-gray-800 dark:text-gray-200">
      <div className="prose dark:prose-invert max-w-none">
        {/* {response.split('\n').map((paragraph, index) => (
          <p key={index}>{paragraph}</p>
        ))} */}
        {
          response
        }
      </div>
    </div>
  );
};

export default ResponseDisplay;