import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Download stopwords (if not already available)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and special characters
    text = re.sub(r'[' + string.punctuation + ']', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Tokenize and remove stopwords
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)

# Step 1: Load the CSV Dataset
df = pd.read_csv("imdb_dataset.csv")
print("Dataset Preview:")
print(df.head())

# Step 2: Preprocess the text
df['cleaned_text'] = df['review'].apply(preprocess_text)

# Map sentiment labels to binary values (1 for positive, 0 for negative)
df['Sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Step 3: Feature Extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_text'])
y = df['Sentiment']

# Step 4: Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 6: Save the trained model and vectorizer using joblib
joblib.dump(model, 'logistic_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
print("\nModel and vectorizer have been saved to disk.")
