import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Set page config
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="😊",
    layout="wide"
)

# Load saved model and resources
@st.cache_resource
def load_model():
    with open("sentiment_model/best_model.pkl", "rb") as f:
        model_data = pickle.load(f)
    with open("sentiment_model/tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("sentiment_model/class_names.pkl", "rb") as f:
        class_names = pickle.load(f)
    return model_data, vectorizer, class_names

model_data, vectorizer, class_names = load_model()

# App title
st.title("Sentiment Analysis App")
st.markdown(f"""
**Model Used:** {model_data['model_name']}  
**Test Accuracy:** {model_data['accuracy']:.2%}  
""")

# Input section
st.header("🔍 Analyze Text Sentiment")
user_input = st.text_area("Enter text to analyze:", "I really love this product! It's amazing.")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text before analyzing.")
    else:
        # Transform input
        text_vector = vectorizer.transform([user_input])
        expected_features = model_data['model'].n_features_in_

        # Check feature compatibility
        if text_vector.shape[1] != expected_features:
            st.error(
                f"Model expects {expected_features} features but received {text_vector.shape[1]}. "
                f"Ensure your vectorizer and model were saved from the same training session."
            )
        else:
            # Predict
            prediction = model_data['model'].predict(text_vector)[0]
            probabilities = model_data['model'].predict_proba(text_vector)[0]

            # Display results
            st.subheader("Results")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Predicted Sentiment", class_names[prediction])
            with col2:
                confidence = np.max(probabilities) * 100
                st.metric("Confidence", f"{confidence:.1f}%")

            # Probability distribution
            st.subheader("Probability Distribution")
            prob_df = pd.DataFrame({
                'Sentiment': list(class_names.values()),
                'Probability': probabilities
            })
            st.bar_chart(prob_df.set_index('Sentiment'))

# Sidebar
st.sidebar.header("Sample Texts")
sample_texts = {
    "Positive": "I absolutely love this place! The service was excellent and the food was delicious.",
    "Negative": "This was the worst experience ever. The product broke after one day of use."
}
sample_choice = st.sidebar.selectbox("Try a sample:", list(sample_texts.keys()))
if st.sidebar.button("Load Sample"):
    st.experimental_set_query_params(sample=sample_choice)
    st.rerun()

st.sidebar.header("How To Use")
st.sidebar.info("""
1. Enter your text in the main box  
2. Click **'Analyze'**  
3. View the predicted sentiment and confidence  
""")

st.sidebar.header("ℹ️ About")
st.sidebar.info("""
This app uses a machine learning model trained on Wikipedia-style text.  
- Model: Random Forest  
- Feature Extraction: TF-IDF (max 1000 features)  
""")
