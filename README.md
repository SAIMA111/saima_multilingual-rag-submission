# Multilingual RAG System (Bangla + English)

## Objective
A Retrieval-Augmented Generation (RAG) system that understands Bangla and English questions and provides answers using OCR-processed HSC Bangla 1st Paper textbook content.

## Key Features
- Works with both Bangla and English questions.
- Uses EasyOCR for extracting mixed-language text.
- Uses Sentence Transformers for multilingual embeddings.
- Utilizes Hugging Face's mT5 model (`google/mt5-small`) for question answering.

## Tools Used
- `EasyOCR` for OCR
- `sentence-transformers` (`distiluse-base-multilingual-cased-v2`) for chunk and query embeddings
- `transformers` (HuggingFace mT5) for QA generation

## Setup Instructions

```bash
pip install -r requirements.txt
```

Run the scripts in the following order:

```bash
python ocr_extractor.py          # Extract text from PDF/images using EasyOCR
python chunking/text_splitter.py # Split extracted text into chunks
python embedder.py               # Create embeddings and save them
python retriever.py              # Ask your question and get answers
```

> Ensure the data/ folder contains either extracted_text_easyocr.txt (if you've already run OCR) or textbook.pdf (to extract OCR text) before starting.


## Evaluation
Note: The current system was tested only on 3 sample pages from the textbook due to time and resource constraints.

Despite this limited scope, the system generated nearly accurate answers for sample questions:

Query 1: অনুপমের বয়স কত বছর?
Predicted Answer: পঁচিশ

Query 2: কে ছিলেন পরিবারের কর্তা?
Predicted Answer: লক্ষ্মী

This shows promising potential for accurate multilingual QA with further refinement and full dataset coverage.



## Project Structure

```
multilingual_rag_project/
│
├── ocr/
│   └── ocr_extractor.py
├── chunking/
│   └── text_splitter.py
├── embedding/
│   └── embedder.py
├── retriever/
│   └── retriever.py
├── data/
│   ├── textbook.pdf
│   ├── extracted_text_easyocr.txt
│   ├── chunks.txt
│   └── embeddings.pkl
├── utils/
│
└── README.md
```

## Implementation Questions

1. What method or library did you use to extract the text, and why? Did you face any formatting challenges with the PDF content?
We used EasyOCR for text extraction from the textbook (textbook.pdf) because it supports Bangla and English, unlike many traditional OCR engines that only work well with English or Latin scripts.

Reason: EasyOCR offers multilingual support, making it suitable for our bilingual document.

However, we did face formatting challenges such as:

Misrecognized characters (e.g., Bangla letters interpreted as numbers).

Incorrect line breaks and symbol clutter.

Some tick marks (✓) were misread as Bangla numbers like "৮".

To mitigate these, we performed basic cleaning in the chunking step.

2. What chunking strategy did you choose (e.g. paragraph-based, sentence-based, character limit)? Why do you think it works well for semantic retrieval?
We used a line-based chunking approach, where each chunk is created by splitting the extracted text using newline characters, with minor adjustments for length.

Reason: This lightweight method helps isolate idea-level content from OCR, especially since full paragraphs are not always clearly formatted due to OCR artifacts.

It works reasonably well for semantic retrieval because short text lines reduce noise and often capture distinct concepts or exam-like statements.

3. What embedding model did you use? Why did you choose it? How does it capture the meaning of the text?
We used the distiluse-base-multilingual-cased-v2 model from the sentence-transformers library.

Reason: It's a multilingual model fine-tuned for semantic similarity tasks and works well for both Bangla and English. It offers good performance while being lightweight.

This model converts sentences into dense vectors that capture semantic meaning based on context rather than just keywords. It performs cosine similarity comparisons in vector space, allowing it to match even semantically similar but not identical phrases.

4. How are you comparing the query with your stored chunks? Why did you choose this similarity method and storage setup?
We use cosine similarity between the embedding of the user query and each document chunk. This is computed using vector math via NumPy.

Reason: Cosine similarity is a standard, effective method to measure the angle between two high-dimensional vectors, making it ideal for semantic similarity.

The embedding vectors are stored in a simple .pkl file (Python pickle), and all computation is done in-memory for fast retrieval.

5. How do you ensure that the question and the document chunks are compared meaningfully? What would happen if the query is vague or missing context?
We ensure meaningful comparison by:

Using a multilingual sentence embedding model that supports both Bangla and English.

Ensuring the chunking process retains logical units of information (short statements or lines).

If a query is too vague or lacks context, the system may retrieve semantically weak or unrelated chunks. This is a limitation of our current chunking granularity and lack of advanced context tracking.

Future improvements may include:

Contextual filtering

Reranking with generative models

Better pre-processing to merge logical paragraph blocks

6. Do the results seem relevant? If not, what might improve them (e.g. better chunking, better embedding model, larger document)?
Yes, the results are partially relevant.

For example, on a 3-page sample test from the textbook, the system produced:

"অনুপমের বয়স কত বছর?" →  পঁচিশ

"কে ছিলেন পরিবারের কর্তা?" →  লক্ষ্মী

However, some answers were off due to:

OCR noise

Limited document size

Overly short chunks

To improve accuracy:

Use paragraph-level chunking (with OCR corrections)

Adopt more robust embedding models (like LaBSE, bge-m3)

Expand the document set beyond 3 pages

Implement context windowing or reranking using mT5/mBERT

## Submission Date

26 July 2025

