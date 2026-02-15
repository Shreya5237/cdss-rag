Clinical Decision Support System (CDSS) – RAG Edition

An AI-powered Rare Disease Differential Diagnosis Assistant built entirely using free and open-source tools.

This project uses Retrieval-Augmented Generation (RAG) to retrieve peer-reviewed PubMed literature and assist doctors by suggesting possible differential diagnoses based on complex symptom input.

Disclaimer
This system is intended for research and educational purposes only. It does not replace professional medical judgment.

Project Overview

This system acts as an AI second-opinion assistant. A doctor can enter scattered or complex patient symptoms, and the system retrieves relevant medical literature from PubMed and generates evidence-backed differential diagnosis suggestions.

The goal of this project is to demonstrate how RAG systems can be applied in healthcare settings using only free resources.

How It Works

PubMed abstracts are fetched using the NCBI Entrez API.

The text is converted into embeddings using a local HuggingFace model.

The embeddings are stored in a local ChromaDB vector database.

A doctor enters patient symptoms into the Streamlit dashboard.

The system retrieves the most relevant medical literature.

Gemini 1.5 Flash generates differential diagnosis suggestions with reasoning.

Architecture

LLM
Google Gemini 1.5 Flash (Free Tier)

Embeddings
HuggingFace all-MiniLM-L6-v2 (Local and Free)

Vector Database
ChromaDB (Local and Free)

Framework
LlamaIndex and Streamlit

Data Source
PubMed via NCBI Entrez API

Quick Start Guide
1. Prerequisites

Python 3.9 or higher

Internet connection

Virtual environment activated

2. Install Dependencies

Run the following command inside your virtual environment:

.\venv\Scripts\pip install -r requirements.txt

3. Setup API Key

Get a free API key from Google AI Studio
https://aistudio.google.com/app/apikey

Rename .env.example to .env

Open .env and add your key:

GOOGLE_API_KEY=your_api_key_here

4. Build the Knowledge Base

Fetch PubMed abstracts and build the vector database:

.\venv\Scripts\python setup_knowledge_base.py

This will:

Download abstracts

Create embeddings

Store vectors in the chroma_db folder

5. Run the Application

Launch the Streamlit dashboard:

.\venv\Scripts\streamlit run app.py

Open the local URL shown in the terminal to access the application.

Project Structure

cdss-rag/

app.py
Main Streamlit dashboard

setup_knowledge_base.py
Script for ingestion and vector database creation

pubmed_fetch.py
Helper module to fetch abstracts from PubMed

chroma_db/
Local vector database created after ingestion

requirements.txt
Project dependencies

README.md
Project documentation

Use Case

This system is designed for:

Rare disease differential diagnosis

Complex and non-specific symptom cases

AI-assisted clinical research workflows

Future Improvements

Hybrid search using keyword and vector search

Structured medical output format (JSON)

Confidence scoring system

ICD-10 mapping integration

RAG evaluation using RAGAS

Deployment on HuggingFace Spaces

Author

Shreya R
Vetri Selvi M
Divyadharshiny SP
M.Tech CSE (Integrated)
AI and Clinical Decision Support Enthusiast

