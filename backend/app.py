from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import os
import traceback
import re
import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import TfidfVectorizer
from duckduckgo_search import DDGS

# --- Configuration ---
MAX_SCRAPED_RESULTS = 5
SCRAPE_TIMEOUT = 5 # Seconds
WORDS_FOR_SCRAPING_QUERY = 10 # Use first 10 words for query
SIMILARITY_THRESHOLD = 0.5 # Minimum cosine similarity to be considered 'similar'
MAX_SIMILAR_RESULTS = 3 # Show top N similar fake/true
MAX_EXCESSIVE_PUNCT = 5 # Threshold for flagging punctuation
MAX_ALL_CAPS_WORDS = 5 # Threshold for flagging all caps words

# LLM Configuration
PRIMARY_LLM_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
SECONDARY_LLM_MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"
ALTERNATIVE_LLM_MODEL_NAME = "microsoft/phi-2"
LLM_MAX_NEW_TOKENS = 250
LLM_TEMPERATURE = 0.7
USE_SECONDARY_MODEL = True
USE_ALTERNATIVE_MODEL = True

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
except ImportError:
    print("ERROR: transformers or torch not found. Please install dependencies.")
    AutoModelForCausalLM = None
    AutoTokenizer = None
    torch = None

# --- FastAPI App Setup ---
app = FastAPI(title="Veritas AI API")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Paths ---
backend_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(backend_dir, '..', 'models')
data_dir = os.path.join(backend_dir, '..', 'data')
model_path = os.path.join(models_dir, 'model_pipeline.pkl')
vectorizer_path = os.path.join(models_dir, 'tfidf_vectorizer.pkl')
processed_data_path = os.path.join(data_dir, 'processed_news_data.pkl')
frontend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../frontend'))

# --- Global Models ---
vader_analyzer = SentimentIntensityAnalyzer()
classifier = None
vectorizer = None
processed_data_df = None
all_text_vectors = None
llm_primary_model = None
llm_primary_tokenizer = None
llm_secondary_model = None
llm_secondary_tokenizer = None

# --- Pydantic Models ---
class NewsRequest(BaseModel):
    text: str

class SentimentResult(BaseModel):
    label: str
    score: float
    details: dict

class ScrapeResult(BaseModel):
    status: str
    message: str
    data: list

class ScrapedAnalysis(BaseModel):
    keywords: list
    sources: list

class PredictionResponse(BaseModel):
    analyzed_text: str
    classification: str | None
    classification_confidence: float | None
    sentiment: SentimentResult
    basic_linguistic_flags: list[str]
    similarity_results: dict
    scraped_results: ScrapeResult
    scraped_analysis: ScrapedAnalysis
    llm_analysis: str

# --- Load Resources (Startup) ---
@app.on_event("startup")
async def load_resources():
    global classifier, vectorizer, processed_data_df, all_text_vectors
    global llm_primary_model, llm_primary_tokenizer, llm_secondary_model, llm_secondary_tokenizer

    # Load Classifier
    try:
        print(f"Loading classifier from: {model_path}")
        if os.path.exists(model_path):
            _pipeline_loaded = joblib.load(model_path)
            if len(_pipeline_loaded.steps) >= 2:
                classifier = _pipeline_loaded.steps[1][1]
                print("Classifier loaded.")
            else:
                print("Error: Pipeline structure invalid.")
        else:
            print("Error: Classifier file not found.")
    except Exception as e:
        print(f"Error loading classifier: {e}")

    # Load Vectorizer
    try:
        print(f"Loading vectorizer from: {vectorizer_path}")
        if os.path.exists(vectorizer_path):
            vectorizer = joblib.load(vectorizer_path)
            print("Vectorizer loaded.")
        else:
            print("Error: Vectorizer file not found.")
    except Exception as e:
        print(f"Error loading vectorizer: {e}")

    # Load Processed Data
    try:
        print(f"Loading processed data from: {processed_data_path}")
        if os.path.exists(processed_data_path):
            processed_data_df = joblib.load(processed_data_path)
            print(f"Processed data loaded ({len(processed_data_df)} articles).")
            if vectorizer is not None and not processed_data_df.empty:
                print("Calculating TF-IDF vectors...")
                all_text_vectors = vectorizer.transform(processed_data_df['text'])
                print("Vectors calculated.")
        else:
            print("Error: Processed data file not found.")
    except Exception as e:
        print(f"Error loading processed data: {e}")

    # Load LLMs
    if AutoModelForCausalLM is not None and torch is not None and os.environ.get("ENABLE_LLM", "True").lower() == "true":
        load_llms()
    else:
        print("LLM loading skipped (disabled via env var or dependencies missing).")

def load_llms():
    global llm_primary_model, llm_primary_tokenizer, llm_secondary_model, llm_secondary_tokenizer
    
    print("Loading LLMs...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Using device: {device}, dtype: {dtype}")

    # Primary Model
    try:
        llm_primary_tokenizer = AutoTokenizer.from_pretrained(PRIMARY_LLM_MODEL_NAME)
        llm_primary_model = AutoModelForCausalLM.from_pretrained(
            PRIMARY_LLM_MODEL_NAME, torch_dtype=dtype, device_map="auto"
        )
        llm_primary_model.eval()
        print(f"Primary model ({PRIMARY_LLM_MODEL_NAME}) loaded.")
    except Exception as e:
        print(f"Error loading primary LLM: {e}")

    # Secondary Model
    if USE_SECONDARY_MODEL:
        try:
            print(f"Loading secondary model: {SECONDARY_LLM_MODEL_NAME}")
            llm_secondary_tokenizer = AutoTokenizer.from_pretrained(SECONDARY_LLM_MODEL_NAME)
            
            # Try 4-bit quantization
            try:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16
                )
                llm_secondary_model = AutoModelForCausalLM.from_pretrained(
                    SECONDARY_LLM_MODEL_NAME, quantization_config=bnb_config, device_map="auto"
                )
                print("Secondary model loaded (4-bit).")
            except Exception:
                print("4-bit quantization failed, falling back to standard.")
                llm_secondary_model = AutoModelForCausalLM.from_pretrained(
                    SECONDARY_LLM_MODEL_NAME, torch_dtype=dtype, device_map="auto", low_cpu_mem_usage=True
                )
                print("Secondary model loaded (standard).")
            
            llm_secondary_model.eval()
        except Exception as e:
            print(f"Error loading secondary LLM: {e}")
            # Try Alternative
            if USE_ALTERNATIVE_MODEL:
                try:
                    print(f"Loading alternative model: {ALTERNATIVE_LLM_MODEL_NAME}")
                    llm_secondary_tokenizer = AutoTokenizer.from_pretrained(ALTERNATIVE_LLM_MODEL_NAME)
                    llm_secondary_model = AutoModelForCausalLM.from_pretrained(
                        ALTERNATIVE_LLM_MODEL_NAME, torch_dtype=dtype, device_map="auto", low_cpu_mem_usage=True
                    )
                    llm_secondary_model.eval()
                    print("Alternative model loaded.")
                except Exception as alt_e:
                    print(f"Error loading alternative LLM: {alt_e}")

# --- Helper Functions ---
def analyze_sentiment(text):
    try:
        vs = vader_analyzer.polarity_scores(text)
        compound = vs['compound']
        if compound >= 0.05: label = "Positive"
        elif compound <= -0.05: label = "Negative"
        else: label = "Neutral"
        return {"label": label, "score": compound, "details": vs}
    except Exception as e:
        print(f"Sentiment error: {e}")
        return {"label": "Error", "score": 0.0, "details": {}}

def check_basic_linguistics(text):
    excessive_punct = len(re.findall(r'[!\?]{2,}', text))
    all_caps_words = len(re.findall(r'\b[A-Z]{2,}\b', text))
    flags = []
    if excessive_punct >= MAX_EXCESSIVE_PUNCT:
        flags.append(f"High amount of excessive punctuation found ({excessive_punct}).")
    if all_caps_words >= MAX_ALL_CAPS_WORDS:
        flags.append(f"High amount of all-caps words found ({all_caps_words}).")
    return flags if flags else ["No basic linguistic flags detected."]

def find_similar_articles(input_text):
    if vectorizer is None or all_text_vectors is None or processed_data_df is None or processed_data_df.empty:
        return {"similar_true": [], "similar_fake": []}
    try:
        input_vector = vectorizer.transform([input_text])
        similarities = cosine_similarity(input_vector, all_text_vectors)[0]
        # Use a copy to avoid SettingWithCopyWarning on the global df
        temp_df = processed_data_df.copy()
        temp_df['similarity'] = similarities
        
        similar_df = temp_df[temp_df['similarity'] >= SIMILARITY_THRESHOLD]
        top_true = similar_df[similar_df['label'] == 1].nlargest(MAX_SIMILAR_RESULTS, 'similarity')
        top_fake = similar_df[similar_df['label'] == 0].nlargest(MAX_SIMILAR_RESULTS, 'similarity')

        format_result = lambda row: f"{row['title']} (Similarity: {row['similarity']:.2f}) - Snippet: {str(row['text'])[:100]}..."

        similar_true_list = list(top_true.apply(format_result, axis=1)) if not top_true.empty else []
        similar_fake_list = list(top_fake.apply(format_result, axis=1)) if not top_fake.empty else []

        return {"similar_true": similar_true_list, "similar_fake": similar_fake_list}
    except Exception as e:
        print(f"Similarity search error: {e}")
        return {"similar_true": ["Error"], "similar_fake": ["Error"]}

def scrape_duckduckgo(query):
    if not query:
        return {"status": "error", "message": "No query provided.", "data": []}
    print(f"Scraping DDG for: {query}")
    try:
        results = []
        ddgs_gen = DDGS().text(query, max_results=MAX_SCRAPED_RESULTS)
        for r in ddgs_gen:
            results.append({'title': r['title'], 'link': r['href'], 'snippet': r['body']})
        
        if not results:
            return {"status": "success", "message": "No results found.", "data": []}
        return {"status": "success", "message": f"{len(results)} results found.", "data": results}
    except Exception as e:
        print(f"Scraping error: {e}")
        return {"status": "error", "message": str(e), "data": []}

def get_llm_analysis(text, scraped_snippets):
    if not torch: return "LLM disabled (torch missing)."
    
    # Select model
    model = llm_secondary_model if (USE_SECONDARY_MODEL and llm_secondary_model) else llm_primary_model
    tokenizer = llm_secondary_tokenizer if (USE_SECONDARY_MODEL and llm_secondary_model) else llm_primary_tokenizer
    
    if not model or not tokenizer: return "LLM disabled (no models loaded)."

    print(f"Running LLM analysis using {model.config._name_or_path}...")
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant analyzing news credibility."},
        {"role": "user", "content": f"Analyze the following news text. Is it likely true or fake? Provide a thorough justification.\n\nNews Text: '{text}'\n"}
    ]
    if scraped_snippets:
        messages[1]["content"] += "\nRelevant Search Snippets:\n" + "\n".join([f"{i+1}. {s}" for i, s in enumerate(scraped_snippets[:3])])

    try:
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=LLM_MAX_NEW_TOKENS,
                temperature=LLM_TEMPERATURE,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        output_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return output_text.strip()
    except Exception as e:
        print(f"LLM inference error: {e}")
        return "Error during LLM analysis."

# --- Endpoints ---

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: NewsRequest):
    text_to_analyze = request.text
    
    # 1. Classification
    classification = None
    confidence = None
    if classifier and vectorizer:
        try:
            vectorized_input = vectorizer.transform([text_to_analyze])
            prediction_label = classifier.predict(vectorized_input)[0]
            prediction_proba = classifier.predict_proba(vectorized_input)[0]
            classification = "Likely True" if prediction_label == 1 else "Likely Fake"
            confidence = prediction_proba[1] if prediction_label == 1 else prediction_proba[0]
        except Exception as e:
            print(f"Classification error: {e}")
            classification = "Error"
            confidence = 0.0
    else:
        classification = "Model not loaded"
        confidence = 0.0

    # 2. Sentiment
    sentiment = analyze_sentiment(text_to_analyze)

    # 3. Linguistics
    linguistics = check_basic_linguistics(text_to_analyze)

    # 4. Similarity
    similarity = find_similar_articles(text_to_analyze)

    # 5. Scraping
    scraped_results = {"status": "skipped", "message": "Empty text", "data": []}
    scraped_analysis = {"keywords": [], "sources": []}
    
    if text_to_analyze.strip():
        words = text_to_analyze.split()
        query = " ".join(words[:WORDS_FOR_SCRAPING_QUERY])
        if query:
            scraped_results = scrape_duckduckgo(query)
            
            # 5b. Snippet Analysis
            if scraped_results['status'] == 'success' and scraped_results['data']:
                snippets = [item['snippet'] for item in scraped_results['data']]
                scraped_analysis['sources'] = scraped_results['data']
                if len(snippets) > 1:
                    try:
                        snippet_vectorizer = TfidfVectorizer(stop_words='english', max_features=50)
                        tfidf_matrix = snippet_vectorizer.fit_transform(snippets)
                        feature_names = snippet_vectorizer.get_feature_names_out()
                        total_scores = tfidf_matrix.sum(axis=0).A1
                        top_indices = np.argsort(total_scores)[-10:][::-1]
                        scraped_analysis['keywords'] = [feature_names[i] for i in top_indices]
                    except Exception as e:
                        print(f"Snippet analysis error: {e}")
                        scraped_analysis['keywords'] = ["Error"]
                else:
                    scraped_analysis['keywords'] = ["Insufficient data"]

    # 6. LLM
    llm_input_snippets = [item['snippet'] for item in scraped_results.get('data', [])]
    llm_result = get_llm_analysis(text_to_analyze, llm_input_snippets)

    return PredictionResponse(
        analyzed_text=text_to_analyze,
        classification=classification,
        classification_confidence=confidence,
        sentiment=SentimentResult(**sentiment),
        basic_linguistic_flags=linguistics,
        similarity_results=similarity,
        scraped_results=ScrapeResult(**scraped_results),
        scraped_analysis=ScrapedAnalysis(**scraped_analysis),
        llm_analysis=llm_result
    )

# Serve Static Files
app.mount("/", StaticFiles(directory=frontend_path, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)