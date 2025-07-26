# Multilingual RAG System (Bangla + English)

## 🔍 Objective
A Retrieval-Augmented Generation system that understands Bangla and English questions and answers using content from the HSC Bangla 1st Paper book.

## 📚 Tools Used
- EasyOCR (Text Extraction)
- SentenceTransformers (`distiluse-base-multilingual-cased-v2`)
- HuggingFace mT5 (`google/mt5-small`)

## ⚙️ Setup Guide

```bash
pip install sentence-transformers transformers easyocr
