# Veritas AI

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)

Veritas AI is a comprehensive tool designed to identify fake news articles using a combination of machine learning (Logistic Regression + TF-IDF), sentiment analysis (VADER), and similarity search. It now includes **LLM-powered analysis** using local models (TinyLlama, Qwen, Phi-2) for deeper insights and **internet search** via DuckDuckGo to verify claims against live web sources.

## Features

-   **Fake News Detection**: Uses a trained Logistic Regression model to classify news as Real or Fake.
-   **Sentiment Analysis**: Analyzes the emotional tone of the text.
-   **Web Scraping**: Automatically verifies facts by searching for relevant articles on DuckDuckGo.
-   **LLM Analysis**: Optional deep-dive analysis using local Large Language Models.
-   **Docker Support**: Fully containerized for easy deployment.

## Structure

-   `backend/`: Flask/FastAPI application serving the model and API.
-   `frontend/`: HTML/CSS/JS frontend for user interaction.
-   `data/`: Dataset files (`True.csv`, `Fake.csv`) and processed data.
-   `models/`: Saved model artifacts (`model_pipeline.pkl`, `tfidf_vectorizer.pkl`).
-   `train_model.py`: Script to train the machine learning model.

## Setup

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/your-username/veritas-ai.git
    cd veritas-ai
    ```

2.  **Create Virtual Environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r backend/requirements.txt
    python -m spacy download en_core_web_sm
    ```

4.  **Data & Models**:
    -   Place your `True.csv` and `Fake.csv` datasets in the `data/` directory.
    -   Run training to generate the model and processed data:
        ```bash
        python train_model.py
        ```

5.  **Run Application**:
    ```bash
    # Run with Uvicorn (FastAPI)
    uvicorn backend.app:app --host 0.0.0.0 --port 5001
    ```
    Access at `http://127.0.0.1:5001`.

## Docker

1.  **Build Image**:
    ```bash
    docker build -t veritas-ai .
    ```

2.  **Run Container**:
    ```bash
    # Normal run
    docker run -p 5001:5001 veritas-ai

    # Run with LLMs disabled (for low-memory environments)
    docker run -p 5001:5001 -e ENABLE_LLM=false veritas-ai
    ```

## Troubleshooting

-   **Container Crashes (OOM)**: If the Docker container crashes immediately or during analysis, it's likely due to the Large Language Models (LLMs) running out of memory. Try running with `-e ENABLE_LLM=false`.
-   **Port Conflicts**: If port 5001 is busy, change the `-p` flag, e.g., `-p 8080:5001`.

## Deployment

This project is configured for deployment on Vercel via the `vercel.json` file.
