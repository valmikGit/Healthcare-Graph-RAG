import React from 'react';
import { Clock, ArrowRight } from 'lucide-react';

interface QuestionHistoryProps {
  questions: { question: string; model: number }[];
  onSelectQuestion: (question: string, model: number) => void;
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
        {questions.map(({ question, model }, index) => (
          <button
            key={index}
            onClick={() => onSelectQuestion(question, model)}
            className="w-full text-left px-4 py-2 rounded-lg text-sm bg-gray-100 hover:bg-gray-200 dark:bg-gray-800 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300 transition-colors duration-200 flex items-center justify-between group"
          >
            <span className="truncate max-w-[75%]">{question}</span>
            <span className="text-xs text-gray-500 dark:text-gray-400 ml-2">{model === 0 ? 'BERT' : 'GPT-2'}</span>
            <ArrowRight className="w-4 h-4 ml-2 opacity-0 group-hover:opacity-100 transition-opacity duration-200" />
          </button>
        ))}
      </div>
    </div>
  );
};

export default QuestionHistory;