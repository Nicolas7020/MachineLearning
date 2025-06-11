
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os

# Train model function
@st.cache_resource
def train_model():
    """Train the fake news detection model from scratch"""

    try:
        # Load data
        with st.spinner("Loading training data..."):
            fake_news = pd.read_csv('Fake.csv')
            true_news = pd.read_csv('True.csv')

        # Prepare data
        fake_news['label'] = 1  # Fake = 1
        true_news['label'] = 0  # Real = 0

        # Combine datasets
        news_data = pd.concat([
            fake_news[['title', 'text', 'label']], 
            true_news[['title', 'text', 'label']]
        ])

        # Shuffle and remove duplicates
        news_data = news_data.sample(frac=1).reset_index(drop=True)
        news_data = news_data.drop_duplicates()

        # Create vectorizers
        with st.spinner("Creating TF-IDF vectors..."):
            tfidf_text = TfidfVectorizer(max_features=5000, stop_words='english')
            tfidf_title = TfidfVectorizer(max_features=2000, stop_words='english')

            # Fit and transform
            text_vectors = tfidf_text.fit_transform(news_data['text'])
            title_vectors = tfidf_title.fit_transform(news_data['title'])

            # Combine vectors
            combined_vectors = np.hstack((text_vectors.toarray(), title_vectors.toarray()))

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            combined_vectors, news_data['label'], 
            test_size=0.2, random_state=42
        )

        # Train model
        with st.spinner("Training model..."):
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)

            # Calculate accuracy
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

        return model, tfidf_text, tfidf_title, accuracy, len(news_data)

    except FileNotFoundError as e:
        st.error(f"Data files not found: {e}")
        st.info("Please ensure 'Fake.csv' and 'True.csv' are in the same directory")
        return None, None, None, None, None
    except Exception as e:
        st.error(f"Error training model: {e}")
        return None, None, None, None, None

def predict_news(title, text, model, tfidf_text, tfidf_title):
    """Make prediction on news article"""
    try:
        # Transform input
        text_vector = tfidf_text.transform([text])
        title_vector = tfidf_title.transform([title])

        # Combine vectors
        combined_vector = np.hstack((text_vector.toarray(), title_vector.toarray()))

        # Predict
        prediction = model.predict(combined_vector)[0]
        probability = model.predict_proba(combined_vector)[0]

        return prediction, probability

    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

def main():
    # Page config
    st.set_page_config(
        page_title="Fake News Detector",
        page_icon="📰",
        layout="wide"
    )

    # Title
    st.title("🔍 Fake News Detector")
    st.markdown("---")
    st.subheader("AI-Powered News Authenticity Checker")

    # Sidebar info
    st.sidebar.title("📊 Model Info")

    # Train model
    with st.sidebar:
        st.info("Training model from scratch...")
        model, tfidf_text, tfidf_title, accuracy, data_count = train_model()

    if model is None:
        st.error("❌ Could not train model. Please check your data files.")
        return

    # Display model info
    st.sidebar.success("✅ Model trained successfully!")
    st.sidebar.metric("Model Accuracy", f"{accuracy:.3f}")
    st.sidebar.metric("Training Data", f"{data_count:,} articles")

    # Main interface
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 📝 Enter News Article")

        # Initialize session state for examples
        if 'example_title' not in st.session_state:
            st.session_state.example_title = ""
        if 'example_text' not in st.session_state:
            st.session_state.example_text = ""

        # Example section
        with st.expander("💡 Try Example Articles"):
            col_ex1, col_ex2 = st.columns(2)

            with col_ex1:
                if st.button("📰 Real News Example", use_container_width=True):
                    st.session_state.example_title = "U.S. appeals court rejects challenge to Trump voter fraud panel"
                    st.session_state.example_text = "(Reuters) - A U.S. appeals court in Washington on Tuesday upheld a lower court's decision to allow President Donald Trump's commission investigating voter fraud to request data on voter rolls from U.S. states. The commission was created to investigate allegations of voter fraud in the 2016 election."
                    st.success("✅ Real news example loaded! See the input fields below.")

            with col_ex2:
                if st.button("🚨 Suspicious Example", use_container_width=True):
                    st.session_state.example_title = "SHOCKING: Secret Truth About Elections REVEALED!"
                    st.session_state.example_text = "You won't believe what we discovered about recent elections. This incredible revelation will change everything you thought you knew about voting. Mainstream media doesn't want you to see this shocking evidence that proves everything wrong."
                    st.success("✅ Suspicious news example loaded! See the input fields below.")

        # Input fields with session state values
        title = st.text_input(
            "📰 Article Title",
            value=st.session_state.example_title,
            placeholder="Enter the news article title here...",
            help="Provide the complete headline"
        )

        text = st.text_area(
            "📄 Article Text",
            value=st.session_state.example_text,
            placeholder="Paste the full article content here...",
            height=400,
            help="Include the complete article text for better accuracy"
        )

        # Clear button
        col_clear, col_space = st.columns([1, 3])
        with col_clear:
            if st.button("🗑️ Clear", help="Clear the input fields"):
                st.session_state.example_title = ""
                st.session_state.example_text = ""
                st.success("✅ Fields cleared! Please refresh the page to see empty fields.")

        # Predict button
        st.markdown("---")
        predict_button = st.button(
            "🔍 ANALYZE ARTICLE", 
            type="primary", 
            use_container_width=True,
            help="Click to check if this article is real or fake"
        )

        # Prediction logic
        if predict_button:
            if title.strip() and text.strip():
                with st.spinner("🤖 Analyzing article authenticity..."):
                    prediction, probability = predict_news(title, text, model, tfidf_text, tfidf_title)

                    if prediction is not None:
                        st.markdown("---")
                        st.markdown("### 📊 Analysis Results")

                        # Main result
                        if prediction == 0:  # Real
                            st.success("✅ **REAL NEWS** - This appears to be authentic")
                            confidence = probability[0] * 100
                        else:  # Fake  
                            st.error("❌ **FAKE NEWS** - This appears to be suspicious")
                            confidence = probability[1] * 100

                        # Confidence score
                        st.metric(
                            "Confidence Level", 
                            f"{confidence:.1f}%",
                            help="How certain the model is about this prediction"
                        )

                        # Detailed breakdown
                        st.markdown("#### 📈 Probability Breakdown")
                        col_real, col_fake = st.columns(2)

                        with col_real:
                            real_prob = probability[0] * 100
                            st.metric(
                                "Real News", 
                                f"{real_prob:.1f}%",
                                delta=f"{real_prob-50:.1f}%" if real_prob != 50 else None
                            )

                        with col_fake:
                            fake_prob = probability[1] * 100
                            st.metric(
                                "Fake News", 
                                f"{fake_prob:.1f}%",
                                delta=f"{fake_prob-50:.1f}%" if fake_prob != 50 else None
                            )

                        # Interpretation
                        st.markdown("#### 🎯 Interpretation")
                        if confidence > 85:
                            st.success("🎯 **Very High Confidence** - The model is very certain about this classification")
                        elif confidence > 70:
                            st.info("✅ **High Confidence** - The model is quite confident in this result")
                        elif confidence > 60:
                            st.warning("⚖️ **Moderate Confidence** - Consider additional verification from other sources")
                        else:
                            st.warning("❓ **Low Confidence** - The model is uncertain. This article has mixed signals")

            else:
                st.error("⚠️ Please enter both article title and text!")

    with col2:
        st.markdown("### 🤖 How It Works")
        st.info("""
        **Machine Learning Pipeline:**

        📊 **TF-IDF Vectorization**
        - Converts text to numerical features
        - Analyzes word importance and frequency

        🧠 **Logistic Regression**
        - Trained on thousands of real/fake articles
        - Provides probability scores

        🎯 **Dual Analysis**
        - Examines both title and content
        - Combined feature analysis
        """)

        st.markdown("### 📋 Usage Tips")
        st.markdown("""
        **For Best Results:**
        - ✅ Complete article title and text
        - ✅ Full articles (not just summaries)  
        - ✅ English language content
        - ✅ News articles (not opinions)

        **Avoid:**
        - ❌ Very short text snippets
        - ❌ Non-news content
        - ❌ Articles in other languages
        """)

        st.markdown("### ⚠️ Important Disclaimer")
        st.warning("""
        **Educational Purpose Only**

        This tool is for learning and demonstration. 
        Always verify news from multiple reliable 
        sources. AI predictions should not be the 
        sole basis for determining authenticity.
        """)

        st.markdown("### 📈 Model Performance")
        if model is not None:
            st.metric("Training Accuracy", f"{accuracy:.1%}")
            st.caption("Based on train-test split validation")

if __name__ == "__main__":
    main()
