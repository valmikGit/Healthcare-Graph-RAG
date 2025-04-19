// This is a mock service that simulates API calls to a backend
// In a real application, this would make actual fetch calls to your backend

const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

export const askQuestion = async (question: string): Promise<string> => {
  // Simulate network request
  await delay(1500);
  
  // For demonstration purposes, return mock responses based on the question
  if (question.toLowerCase().includes('hello') || question.toLowerCase().includes('hi')) {
    return "Hello! I'm your AI assistant. How can I help you today?";
  }
  
  if (question.toLowerCase().includes('name')) {
    return "I'm InsightAI, a question answering model designed to provide helpful information on a wide range of topics.";
  }
  
  if (question.toLowerCase().includes('time') || question.toLowerCase().includes('date')) {
    return `The current time is ${new Date().toLocaleTimeString()} and the date is ${new Date().toLocaleDateString()}.`;
  }
  
  if (question.toLowerCase().includes('weather')) {
    return "I'm sorry, I don't have access to real-time weather data. In a production environment, this would connect to a weather API to provide current conditions.";
  }
  
  if (question.toLowerCase().includes('how are you')) {
    return "I'm functioning well, thank you for asking! I'm here to assist you with information and answers to your questions.";
  }

  // Default response for other questions
  return `Thank you for your question: "${question}". In a production environment, this would be processed by a backend API that would generate a detailed and helpful response. This is a placeholder response to demonstrate the UI functionality.`;
};