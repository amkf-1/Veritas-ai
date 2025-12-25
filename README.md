# Veritas AI

Veritas AI is a comprehensive tool designed to identify fake news articles using a combination of machine learning (Logistic Regression + TF-IDF), sentiment analysis (VADER), and similarity search. It now includes **LLM-powered analysis** using local models (TinyLlama, Qwen, Phi-2) for deeper insights and **internet search** via DuckDuckGo to verify claims against live web sources.

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

4.  **Run Application**:
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
    docker run -p 5001:5001 veritas-ai
    ```

## Deployment

This project is configured for deployment on Vercel via the `vercel.json` file.
