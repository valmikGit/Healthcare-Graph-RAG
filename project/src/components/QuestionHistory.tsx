import React from 'react';
import { Clock, ArrowRight } from 'lucide-react';

interface QuestionHistoryProps {
  questions: string[];
  onSelectQuestion: (question: string) => void;
}

const QuestionHistory: React.FC<QuestionHistoryProps> = ({ questions, onSelectQuestion }) => {
  if (questions.length === 0) {
    return null;
  }

  return (
    <div className="w-full mt-8">
      <div className="flex items-center mb-3 text-gray-700 dark:text-gray-300">
        <Clock className="w-4 h-4 mr-2" />
        <h3 className="text-sm font-medium">Recent Questions</h3>
      </div>
      <div className="space-y-2">
        {questions.map((question, index) => (
          <button
            key={index}
            onClick={() => onSelectQuestion(question)}
            className="w-full text-left px-4 py-2 rounded-lg text-sm bg-gray-100 hover:bg-gray-200 dark:bg-gray-800 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300 transition-colors duration-200 flex items-center group"
          >
            <span className="truncate">{question}</span>
            <ArrowRight className="w-4 h-4 ml-auto opacity-0 group-hover:opacity-100 transition-opacity duration-200" />
          </button>
        ))}
      </div>
    </div>
  );
};

export default QuestionHistory;