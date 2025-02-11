import re
import string
import nltk
from nltk.corpus import stopwords
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

# Load the saved model and vectorizer
loaded_model = joblib.load('logistic_model.pkl')
loaded_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Define new comments to predict
new_comments = [
    "This product is amazing!",
    "I did not like the service.",
    "I wouldnt suggest this movie",
    "Enjoyed every bit of it",
    " a bad experience",
    "disappointing",
    "great",
    "trash"
]

# Preprocess the new comments
new_comments_cleaned = [preprocess_text(comment) for comment in new_comments]
# Convert the cleaned comments using the loaded TF-IDF vectorizer
new_comments_tfidf = loaded_vectorizer.transform(new_comments_cleaned)
# Predict sentiments using the loaded model
new_predictions = loaded_model.predict(new_comments_tfidf)

print("\nNew Comments Predictions:")
for comment, pred in zip(new_comments, new_predictions):
    label = "Positive" if pred == 1 else "Negative"
    print(f"Comment: {comment} | Sentiment: {label}")
