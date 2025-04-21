import React, { useState } from 'react';
import { Send } from 'lucide-react';

interface QuestionInputProps {
  onSubmit: (question: string, qaModel: number) => void;
  isLoading: boolean;
}

const QuestionInput: React.FC<QuestionInputProps> = ({ onSubmit, isLoading }) => {
  const [question, setQuestion] = useState('');
  const [model, setModel] = useState<'bert' | 'gpt2'>('bert');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (question.trim() && !isLoading) {
      onSubmit(question.trim(), model === 'bert' ? 0 : 1);
      setQuestion('');
    }
  };

  return (
    <form onSubmit={handleSubmit} className="w-full">
      <div className="relative">
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask me anything..."
          className="w-full p-4 pr-36 rounded-xl border border-gray-200 bg-white shadow-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200 text-gray-800 dark:bg-gray-800 dark:border-gray-700 dark:text-white"
          disabled={isLoading}
        />

        {/* Toggle Menu */}
        <select
          className="absolute right-12 top-1/2 transform -translate-y-1/2 bg-gray-200 dark:bg-gray-700 text-sm p-1 rounded-md text-gray-800 dark:text-white focus:outline-none"
          value={model}
          onChange={(e) => setModel(e.target.value as 'bert' | 'gpt2')}
          disabled={isLoading}
        >
          <option value="bert">BERT</option>
          <option value="gpt2">GPT-2</option>
        </select>

        <button
          type="submit"
          disabled={isLoading || !question.trim()}
          className={`absolute right-2 top-1/2 transform -translate-y-1/2 p-2 rounded-lg ${
            isLoading || !question.trim()
              ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
              : 'bg-blue-600 text-white hover:bg-blue-700'
          } transition-colors duration-200`}
          aria-label="Send question"
        >
          <Send className="h-5 w-5" />
        </button>
      </div>
    </form>
  );
};

export default QuestionInput;
