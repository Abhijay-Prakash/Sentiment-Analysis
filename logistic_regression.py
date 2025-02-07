import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Download stopwords once
nltk.download('stopwords')
# Create a global stopwords set
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation and special characters
    text = re.sub(r'[' + string.punctuation + ']', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Tokenize and remove stopwords (using the preloaded stop_words set)
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    return ' '.join(filtered_tokens)

# Step 1: Load the CSV Dataset (adjust the file path as needed)
df = pd.read_csv("imdb_dataset.csv")
print("Dataset Preview:")
print(df.head())

# Step 2: Preprocess the text using the optimized function
df['cleaned_text'] = df['review'].apply(preprocess_text)

# (The rest of your code continues here...)
# Convert sentiment labels to binary
df['Sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Feature Extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_text'])
y = df['Sentiment']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Making Predictions on New Comments (Optional)
new_comments = [
    "This product is amazing!",
    "I did not like the service.",
    "I wouldnt suggest this movie",
    "Enjoyed every bit of it",
    "not worth the time"
    ""
]
new_comments_cleaned = [preprocess_text(comment) for comment in new_comments]
new_comments_tfidf = vectorizer.transform(new_comments_cleaned)
new_predictions = model.predict(new_comments_tfidf)

print("\nNew Comments Predictions:")
for comment, pred in zip(new_comments, new_predictions):
    label = "Positive" if pred == 1 else "Negative"
    print(f"Comment: {comment} | Sentiment: {label}")
