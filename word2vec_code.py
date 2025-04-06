import numpy as np
import re
import random
from collections import Counter

# Sample sentences for training
corpus = [
    "The patient has diabetes and needs insulin.",
    "Insulin is used to treat diabetes.",
    "A symptom of diabetes is blurred vision.",
    "Blurred vision can also be caused by cataracts.",
    "Cataracts affect vision and are treated with surgery."
]

# Tokenize and clean text
def preprocess_text(corpus):
    sentences = []
    for sentence in corpus:
        sentence = sentence.lower()  # Convert to lowercase
        sentence = re.sub(r"[^a-zA-Z0-9\s]", "", sentence)  # Remove punctuation
        sentences.append(sentence.split())  # Tokenize (split by space)
    return sentences

sentences = preprocess_text(corpus)
print(sentences)

# Build vocabulary
word_counts = Counter([word for sentence in sentences for word in sentence])
vocab = list(word_counts.keys())
word2idx = {word: idx for idx, word in enumerate(vocab)}
idx2word = {idx: word for word, idx in word2idx.items()}

print("Vocabulary:", word2idx)

# Create skip-gram training data
def generate_training_data(sentences, window_size=2):
    training_data = []
    
    for sentence in sentences:
        for i, word in enumerate(sentence):
            target_word = word
            context_words = sentence[max(0, i - window_size): i] + sentence[i + 1: i + window_size + 1]
            
            for context_word in context_words:
                training_data.append((word2idx[target_word], word2idx[context_word]))

    return np.array(training_data)

training_data = generate_training_data(sentences)
print("Sample training data:", training_data[:5])  # (center_word, context_word)

# Model parameters
VOCAB_SIZE = len(vocab)
EMBEDDING_DIM = 10  # Size of word vectors

# Initialize weight matrices
W1 = np.random.uniform(-0.8, 0.8, (VOCAB_SIZE, EMBEDDING_DIM))
W2 = np.random.uniform(-0.8, 0.8, (EMBEDDING_DIM, VOCAB_SIZE))

def forward_pass(center_word_idx):
    h = W1[center_word_idx]  # Hidden layer activation
    u = np.dot(h, W2)  # Output layer
    y_hat = np.exp(u) / np.sum(np.exp(u))  # Softmax
    return y_hat, h

def backward_pass(y_hat, h, target_word_idx, learning_rate=0.01):
    global W1, W2
    error = y_hat
    error[target_word_idx] -= 1  # Derivative of cross-entropy loss

    # Compute gradients
    dW2 = np.outer(h, error)  # Gradient for W2
    dW1 = np.dot(W2, error)  # Gradient for W1

    # Update weights
    W2 -= learning_rate * dW2
    W1[target_word_idx] -= learning_rate * dW1

# Training loop
def train_word2vec(epochs=100, learning_rate=0.01):
    for epoch in range(epochs):
        loss = 0
        np.random.shuffle(training_data)  # Shuffle data
        
        for center_word_idx, context_word_idx in training_data:
            y_hat, h = forward_pass(center_word_idx)
            backward_pass(y_hat, h, context_word_idx, learning_rate)

            # Compute loss (negative log likelihood)
            loss -= np.log(y_hat[context_word_idx])

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Train
train_word2vec(epochs=100)

# Get word embedding
def get_embedding(word):
    return W1[word2idx[word]]

# Example: Get embedding for 'diabetes'
print("Embedding for 'diabetes':", get_embedding("diabetes"))