import streamlit as st
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

# --- PAGE CONFIG ---
st.set_page_config(page_title="Fake News Detector", page_icon="🔍")

# --- LOAD & TRAIN MODEL (CACHED) ---
@st.cache_resource
def load_and_train():
    # Use the absolute path you already tested
    df = pd.read_csv(r'C:\AICTE EDUNET\news.csv')
    
    # --- CRITICAL FIX: Remove rows with empty text or labels ---
    df = df.dropna(subset=['text', 'label'])
    
    # Split data
    x_train, x_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=7)
    
    # Vectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, max_features=5000)
    tfidf_train = tfidf_vectorizer.fit_transform(x_train)
    
    # Passive Aggressive Classifier
    pac = PassiveAggressiveClassifier(max_iter=50)
    pac.fit(tfidf_train, y_train)
    
    return tfidf_vectorizer, pac

# Initialize model and vectorizer
try:
    vectorizer, model = load_and_train()
except FileNotFoundError:
    st.error("⚠️ 'news.csv' not found. Please add it to your project folder.")
    st.stop()

# --- USER INTERFACE ---
st.title("🛡️ Fake News Detector for Students")
st.markdown("Analyze news articles to assess their credibility and prevent the spread of misinformation.")

# Text Input Area
user_input = st.text_area("Paste the news article or headline below:", height=200)

if st.button("Check Credibility"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        # Pre-process and Predict
        data = vectorizer.transform([user_input])
        prediction = model.predict(data)[0]
        
        # Display Results
        if prediction == 'REAL':
            st.success(f"✅ This news looks **RELIABLE**.")
            st.balloons()
        else:
            st.error(f"🚨 This news looks **FAKE**.")
            st.info("Tip: Always verify sensational headlines with trusted sources.")

# Footer
st.sidebar.info("This app uses a Passive Aggressive Classifier for lightweight, high-speed detection.")