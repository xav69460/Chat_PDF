# Chat PDF
This is an end to end LLM project that enables users to interact with the content of their PDF files using Google Gemini Pro and Langchain technologies.

## Overview
This project provides a streamlined interface for users to ask questions about the content of their PDF files. It leverages Streamlit for the user interface, PyPDF2 for PDF parsing, Google Generative AI Embeddings for text embeddings, Langchain for text splitting and vector storage, and Google Gemini Pro for conversational AI capabilities. 


## Features
- **PDF Text Extraction:** The project extracts text from uploaded PDF files to enable content interaction.
- **Text Chunking:** Extracted text is chunked for efficient processing and storage.
- **Vectorization:** Text chunks are converted into vectors using Google Generative AI Embeddings for semantic similarity calculations. The vectors are stored in a Facebook AI Similarity Search (FAISS) vector store.
- **Conversational AI**: Users can ask questions about the PDF content, and the system provides detailed answers based on the context. This uses Google's LLM: Gemini Pro



## Potential Applications
- **Customer Support:** Imagine empowering your customer support team with Chat PDF. Customers can ask questions, and the system can instantly retrieve relevant information from your knowledge base, providing quick and accurate support.
- **Knowledge Management:** Organizations can utilize Chat PDF to access and navigate through vast knowledge bases stored in PDF documents. From employee handbooks to technical manuals, extracting actionable insights becomes effortless.
- **Educational Resources:** Students and educators alike can leverage Chat PDF to delve deeper into educational materials, facilitating interactive learning experiences and enhancing comprehension.


References & Credits: Krish Naik:
