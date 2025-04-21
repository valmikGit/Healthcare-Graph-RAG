import React, { useState } from 'react';
import Header from './components/Header';
import QuestionInput from './components/QuestionInput';
import ResponseDisplay from './components/ResponseDisplay';
import QuestionHistory from './components/QuestionHistory';
import ThemeToggle from './components/ThemeToggle';
import { useTheme } from './hooks/useTheme';
import { askQuestion } from './services/apiService';

function App() {
  const [currentResponse, setCurrentResponse] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { isDarkMode, toggleTheme } = useTheme();

  const [questionHistory, setQuestionHistory] = useState<{ question: string; model: number }[]>([]);

  const handleQuestionSubmit = async (question: string, qaModel: number) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch("http://127.0.0.1:5000/ask", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ question, qa_model: qaModel }),
      });

      if (!response.ok) {
        throw new Error("Failed to fetch response from backend");
      }

      const data = await response.json();
      setCurrentResponse(data.answer);

      const alreadyExists = questionHistory.some(q => q.question === question && q.model === qaModel);
      if (!alreadyExists) {
        setQuestionHistory(prev => [{ question, model: qaModel }, ...prev].slice(0, 5));
      }
    } catch (err) {
      setError('Sorry, there was an error processing your question. Please try again.');
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSelectFromHistory = (question: string, model: number) => {
    handleQuestionSubmit(question, model);
  };


  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-gray-100 transition-colors duration-300">
      <Header />
      
      <main className="container mx-auto px-4 py-8 max-w-4xl">
        <div className="space-y-6">
          <div className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700 transition-all duration-300">
            <h2 className="text-2xl font-semibold mb-6 text-center">Ask Me Anything</h2>
            
            <QuestionInput 
              onSubmit={handleQuestionSubmit} 
              isLoading={isLoading} 
            />
            
            <ResponseDisplay 
              response={currentResponse} 
              isLoading={isLoading} 
              error={error} 
            />
            
            <QuestionHistory 
              questions={questionHistory} 
              onSelectQuestion={handleSelectFromHistory} 
            />
          </div>
        </div>
      </main>
      
      <ThemeToggle isDarkMode={isDarkMode} toggleTheme={toggleTheme} />
    </div>
  );
}

export default App;