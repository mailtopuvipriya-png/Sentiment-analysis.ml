import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# 1. LOAD DATASET
# -----------------------------
data = pd.read_csv(
    r"IMDB Dataset.csv\1429_1.csv",
    low_memory=False
)

# -----------------------------
# 2. SELECT REQUIRED COLUMNS
# -----------------------------
data = data[['reviews.text', 'reviews.rating']].dropna()
data.columns = ['review', 'rating']

# -----------------------------
# 3. REMOVE NEUTRAL CLASS (FOR BETTER ACCURACY)
# -----------------------------
data = data[data['rating'] != 3]

def get_sentiment(rating):
    return "positive" if rating >= 4 else "negative"

data['sentiment'] = data['rating'].apply(get_sentiment)

# -----------------------------
# 4. CLEAN TEXT
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

data['review'] = data['review'].apply(clean_text)

# -----------------------------
# 5. BALANCE DATASET
# -----------------------------
min_size = data['sentiment'].value_counts().min()
data = data.groupby('sentiment').sample(n=min_size, random_state=42)

# -----------------------------
# 6. TRAIN / TEST SPLIT
# -----------------------------
X = data['review']
y = data['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# 7. TF-IDF VECTORISATION
# -----------------------------
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=40000,
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.85
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -----------------------------
# 8. MODEL TRAINING
# -----------------------------
model = LinearSVC(
    class_weight='balanced',
    random_state=42
)

model.fit(X_train_vec, y_train)

# -----------------------------
# 9. EVALUATION
# -----------------------------
y_pred = model.predict(X_test_vec)

print("\n==============================")
print("ACCURACY:", accuracy_score(y_test, y_pred))
print("==============================\n")

print(classification_report(y_test, y_pred))

# -----------------------------
# 10. PREDICTION FUNCTION
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def predict_sentiment(text):
    text = clean_text(text)
    vec = vectorizer.transform([text])
    return model.predict(vec)[0]

# -----------------------------
# 11. SAMPLE TESTS
# -----------------------------
print("\n--- SAMPLE TESTS ---")
print(predict_sentiment("I love this product, it's amazing!"))
print(predict_sentiment("Worst experience ever, very bad quality"))
print(predict_sentiment("It's okay, not great not bad"))

# -----------------------------
# 12. MANUAL TEST MODE (LIVE INPUT)
# -----------------------------
print("\n===== MANUAL TEST MODE =====")
print("Type a review and press Enter (type 'exit' to stop)\n")

while True:
    text = input("Enter review: ")

    if text.lower() == "exit":
        print("Exiting...")
        break

    result = predict_sentiment(text)
    print("Prediction:", result)
    print("-" * 40)