## Research Paper Classifier

A deep learning NLP model that automatically classifies research paper abstracts 
into their correct academic category. Built using DistilBERT and PyTorch, 
trained on the Cornell University arXiv dataset containing 3 million papers.

## Project Overview

Every day thousands of research papers are published across dozens of academic 
fields. This project builds an automated classifier that reads a paper abstract 
and instantly predicts which field it belongs to — such as Computer Science, 
Mathematics, Physics, Statistics, and  other categories.

This project was built as a  demonstrating the 
application of natural language processing and deep learning to academic 
document classification.

## Results

| Metric | Score |
|--------|-------|
| Test Accuracy | 82% |
| Weighted F1 Score | 0.8133 |
| Training samples | ~100,000 (1 chunk) |
| Categories | 20 |
| Training environment | VS Code with local GPU |

## Best Performing Categories

| Category | F1 Score |
|----------|----------|
| Astrophysics | 0.94 |
| Computer Science | 0.89 |
| Mathematics | 0.89 |
| Quantum Physics | 0.80 |

## Dataset
- Source: Cornell University arXiv dataset 
- Total available: 3 million papers
- Used for training: 1 chunk of approximately 100,000 papers
- Input: Abstract text
- Labels: 20 academic categories

## Model
- Base model: DistilBERT base uncased
- Fine-tuned for sequence classification
- Max token length: 128
- Training epochs: 3
- Learning rate: 2e-5

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers and Trainer
- Scikit-learn


## Author

Vighnesh Khot 
