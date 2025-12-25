import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# Define paths relative to the script location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(BASE_DIR, 'data')
models_dir = os.path.join(BASE_DIR, 'models')
true_csv_path = os.path.join(data_dir, 'True.csv')
fake_csv_path = os.path.join(data_dir, 'Fake.csv')
model_save_path = os.path.join(models_dir, 'model_pipeline.pkl')
vectorizer_save_path = os.path.join(models_dir, 'tfidf_vectorizer.pkl') # Path for vectorizer
processed_data_save_path = os.path.join(data_dir, 'processed_news_data.pkl') # Path for processed data

# Ensure models directory exists
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

print("Loading datasets...")
try:
    df_true = pd.read_csv(true_csv_path)
    df_fake = pd.read_csv(fake_csv_path)
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    print(f"Please ensure '{os.path.basename(true_csv_path)}' and '{os.path.basename(fake_csv_path)}' exist in the '{data_dir}' directory.")
    exit()
except Exception as e:
    print(f"An error occurred while reading the CSV files: {e}")
    exit()

print("Preprocessing data...")
# Add labels: 1 for True, 0 for Fake
df_true['label'] = 1
df_fake['label'] = 0

# Combine the dataframes
df_combined = pd.concat([df_true, df_fake], ignore_index=True, sort=False)

# Select features (text) and target (label)
# Keep title for context in similarity search results
df_combined = df_combined[['title', 'text', 'label']].copy() # Keep title
df_combined['text'].fillna('', inplace=True) # Fill NaN text with empty string
df_combined['title'].fillna('No Title', inplace=True) # Fill NaN titles
df_combined = df_combined.astype({'text': 'str', 'title': 'str'}) # Ensure text/title columns are string
df_combined.dropna(subset=['label'], inplace=True) # Drop rows where label is missing if any

X = df_combined['text']
y = df_combined['label']

print(f"Total samples: {len(df_combined)}")
if len(df_combined) == 0:
    print("Error: No data available after preprocessing. Check CSV files and column names.")
    exit()

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

print("Setting up model pipeline...")
# Create a pipeline with TF-IDF Vectorizer and Logistic Regression
# We will fit the vectorizer separately first to save it
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7) # Ignore words appearing in > 70% of docs

print("Fitting TF-IDF Vectorizer...")
X_train_tfidf = vectorizer.fit_transform(X_train)

print("Training classifier...")
# Train classifier on the transformed training data
classifier = LogisticRegression(solver='liblinear', random_state=42)
classifier.fit(X_train_tfidf, y_train)

print("Building final pipeline (with fitted classifier)...")
# Note: For prediction, the pipeline needs the unfitted vectorizer step
# The loaded pipeline in app.py will use this structure.
pipeline_for_saving = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.7)), # Unfitted step for pipeline structure
    ('clf', classifier) # Use the classifier already fitted above
])
# Manually set the fitted vectorizer attributes to the pipeline step for saving IF NEEDED
# This ensures the saved pipeline *could* be used directly, although we load vectorizer separately in app.py now.
# pipeline_for_saving.steps[0][1].vocabulary_ = vectorizer.vocabulary_
# pipeline_for_saving.steps[0][1].idf_ = vectorizer.idf_

print("Evaluating model...")
# Evaluate using the fitted vectorizer and classifier on the test set
X_test_tfidf = vectorizer.transform(X_test)
y_pred = classifier.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Fake (0)', 'True (1)'])

print(f"\nModel Accuracy: {accuracy:.4f}\n")
print("Classification Report:")
print(report)

print(f"Saving model pipeline to {model_save_path}...")
joblib.dump(pipeline_for_saving, model_save_path)

print(f"Saving fitted TF-IDF vectorizer to {vectorizer_save_path}...")
joblib.dump(vectorizer, vectorizer_save_path)

print(f"Saving processed data for similarity search to {processed_data_save_path}...")
# Save relevant columns (text, title, label) for fast loading in app
df_combined_minimal = df_combined[['title', 'text', 'label']].copy()
joblib.dump(df_combined_minimal, processed_data_save_path)

print("\nModel training complete and assets saved!") 