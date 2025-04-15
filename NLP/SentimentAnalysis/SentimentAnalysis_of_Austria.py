import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import requests
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import nltk
import pickle

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Step 1-2: Web scrape Wikipedia page
def scrape_wikipedia(country):
    url = f"https://en.wikipedia.org/wiki/{country.replace(' ', '_')}"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'lxml')
    return soup.get_text(strip=True)

# Step 3: Clean text
def clean_text(text):
    text = re.sub(r'\[\d+\]', ' ', text)  # Remove citations [1]
    text = re.sub(r'\(\w+\)', ' ', text)  # Remove parentheses content
    text = re.sub(r'[0-9]+', ' ', text)  # Remove numbers
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    return text

# Step 5: Sentiment analysis function
def analyze_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return "Positive"
    elif analysis.sentiment.polarity == 0:
        return "Neutral"
    else:
        return "Negative"

# Main processing function
def process_country_data(country="Austria"):
    # Scrape and clean data
    raw_text = scrape_wikipedia(country)
    cleaned_text = clean_text(raw_text)
    
    # Sentence tokenization and sentiment analysis
    sentences = sent_tokenize(cleaned_text)
    data = pd.DataFrame({'sentence': sentences})
    data['sentiment'] = data['sentence'].apply(analyze_sentiment)
    
    # Word tokenization and processing
    words = word_tokenize(cleaned_text.lower())
    words = [w for w in words if w.isalnum() and w not in stopwords.words('english') and len(w) > 2]
    
    return data, words

# Train models
def train_models(X, y):
    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "Naive Bayes": MultinomialNB(),
        "KNN": KNeighborsClassifier()
    }
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = classification_report(y_test, y_pred, output_dict=True)
    
    return models, results

# Streamlit app
def main():
    st.title("🇦🇹 Country Sentiment Analysis")
    st.subheader("Wikipedia Text Analysis with Multiple ML Models")
    
    # Process data (can be cached)
    data, words = process_country_data()
    
    # Show word cloud
    st.subheader("Word Cloud")
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(words))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)
    
    # Show sentiment distribution
    st.subheader("Sentiment Distribution")
    st.bar_chart(data['sentiment'].value_counts())
    
    # Prepare data for modeling (only positive/negative)
    binary_data = data[data['sentiment'] != 'Neutral'].copy()
    tfidf = TfidfVectorizer(max_features=1000)
    X = tfidf.fit_transform(binary_data['sentence'])
    y = binary_data['sentiment'].apply(lambda x: 1 if x == 'Positive' else 0)
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_sm, y_sm = smote.fit_resample(X, y)
    
    # Train models
    models, results = train_models(X_sm, y_sm)
    
    # Save the best model (Random Forest) and vectorizer
    with open("best_model.pkl", "wb") as f:
        pickle.dump(models["Random Forest"], f)
    with open("tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(tfidf, f)
    
    # Model comparison
    st.subheader("Model Performance Comparison")
    metrics = []
    for name, report in results.items():
        metrics.append({
            'Model': name,
            'Accuracy': report['accuracy'],
            'Precision (Positive)': report['1']['precision'],
            'Recall (Positive)': report['1']['recall'],
            'F1 (Positive)': report['1']['f1-score']
        })
    st.dataframe(pd.DataFrame(metrics))
    
    # Prediction interface
    st.subheader("Try It Yourself")
    user_input = st.text_area("Enter a sentence to analyze its sentiment:")
    
    if st.button("Predict Sentiment"):
        if user_input:
            # Preprocess and predict
            processed_input = clean_text(user_input)
            vector = tfidf.transform([processed_input])
            
            # Use Random Forest for prediction
            prediction = models["Random Forest"].predict(vector)[0]
            proba = models["Random Forest"].predict_proba(vector)[0]
            
            sentiment = "Positive" if prediction == 1 else "Negative"
            confidence = max(proba)
            
            st.success(f"Predicted Sentiment: **{sentiment}** (Confidence: {confidence:.2f})")
            
            with st.expander("Details"):
                st.write(f"Positive probability: {proba[1]:.4f}")
                st.write(f"Negative probability: {proba[0]:.4f}")
                st.write(f"Using model: Random Forest")
        else:
            st.warning("Please enter a sentence to analyze")

if __name__ == "__main__":
    main()
