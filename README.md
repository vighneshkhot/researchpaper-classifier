# 📄 Research Paper Classifier

A deep learning NLP model that automatically classifies research paper abstracts into their correct academic category. Built using DistilBERT and PyTorch, trained on the Cornell University arXiv dataset.

---

## 🚀 Project Overview

Every day, thousands of research papers are published across multiple academic fields. This project builds an automated classifier that reads a paper abstract and instantly predicts its category — such as Computer Science, Mathematics, Physics, Statistics, and more.

This project demonstrates the application of Natural Language Processing (NLP) and deep learning for academic document classification.

---

## 📊 Results

| Metric | Score |
|------|------|
| Test Accuracy | 82% |
| Weighted F1 Score | 0.8133 |
| Training Samples | ~100,000 |
| Categories | 20 |
| Training Environment | VS Code (Local GPU) |

---

## 🏆 Best Performing Categories

| Category | F1 Score |
|---------|--------|
| Astrophysics | 0.94 |
| Computer Science | 0.89 |
| Mathematics | 0.89 |
| Quantum Physics | 0.80 |

---

## 📂 Dataset

- Source: Cornell University arXiv dataset  
- Total Available: ~3 million papers  
- Used for Training: ~100,000 samples  
- Input: Abstract text  
- Output: 20 academic categories  

---

## 🤖 Model Details

- Base Model: DistilBERT (base-uncased)  
- Task: Sequence Classification  
- Max Token Length: 128  
- Epochs: 3  
- Learning Rate: 2e-5  

---

## 🛠️ Tech Stack

- Python  
- PyTorch  
- Hugging Face Transformers  
- Scikit-learn
- VS Code (Local GPU) 

---
