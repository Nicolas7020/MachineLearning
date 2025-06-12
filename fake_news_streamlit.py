import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.sub-header {
    font-size: 1.5rem;
    color: #ff7f0e;
    margin-bottom: 1rem;
}
.prediction-box {
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
}
.real-news {
    background-color: #d4edda;
    border: 2px solid #28a745;
}
.fake-news {
    background-color: #f8d7da;
    border: 2px solid #dc3545;
}
.confidence-bar {
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">📰 Fake News Detector</h1>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <p style="font-size: 1.2rem; color: #666;">Detect fake news using advanced machine learning techniques</p>
    <p style="color: #888;">Enter a news title and article text to analyze its authenticity</p>
</div>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the dataset"""
    try:
        # Load data
        fake_news = pd.read_csv('Fake.csv')
        true_news = pd.read_csv('True.csv')
        
        # Add labels
        fake_news['label'] = 1
        true_news['label'] = 0
        
        # Combine datasets
        news_data = pd.concat([fake_news[['title', 'text', 'label']], true_news[['title', 'text', 'label']]])
        news_data = news_data.sample(frac=1, random_state=42).reset_index(drop=True)
        news_data = news_data.drop_duplicates()
        
        return news_data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def clean_text(text):
    """Clean and preprocess text data"""
    if pd.isna(text) or text == '':
        return ''
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove special characters and digits, keep only letters and spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

@st.cache_resource
def train_model():
    """Train the fake news detection model"""
    try:
        # Load data
        news_data = load_data()
        if news_data is None:
            return None, None, None
        
        # Data preprocessing
        news_data['title'] = news_data['title'].fillna('')
        news_data['text'] = news_data['text'].fillna('')
        
        # Clean text
        news_data['title_cleaned'] = news_data['title'].apply(clean_text)
        news_data['text_cleaned'] = news_data['text'].apply(clean_text)
        
        # Remove empty entries
        news_data = news_data[(news_data['title_cleaned'] != '') | (news_data['text_cleaned'] != '')]
        
        # Feature engineering
        news_data['combined_text'] = news_data['title_cleaned'] + ' ' + news_data['text_cleaned']
        news_data = news_data[news_data['combined_text'].str.len() >= 50]
        
        # Vectorization
        tfidf_text = TfidfVectorizer(
            max_features=5000, 
            stop_words='english',
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2)
        )
        
        tfidf_title = TfidfVectorizer(
            max_features=2000, 
            stop_words='english',
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2)
        )
        
        # Transform text
        text_vectors = tfidf_text.fit_transform(news_data['text_cleaned'])
        title_vectors = tfidf_title.fit_transform(news_data['title_cleaned'])
        
        # Combine vectors
        combined_vectors = hstack([text_vectors, title_vectors])
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            combined_vectors, 
            news_data['label'], 
            test_size=0.2, 
            random_state=42,
            stratify=news_data['label']
        )
        
        # Train model
        model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            C=1.0,
            solver='liblinear'
        )
        
        model.fit(X_train, y_train)
        
        return model, tfidf_text, tfidf_title
        
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None, None, None

def predict_news(title, text, model, tfidf_text, tfidf_title):
    """Predict if news is real or fake"""
    try:
        # Clean input text
        title_cleaned = clean_text(title)
        text_cleaned = clean_text(text)
        
        # Transform text
        title_vector = tfidf_title.transform([title_cleaned])
        text_vector = tfidf_text.transform([text_cleaned])
        
        # Combine vectors
        combined_vector = hstack([text_vector, title_vector])
        
        # Make prediction
        prediction = model.predict(combined_vector)[0]
        probability = model.predict_proba(combined_vector)[0]
        
        # Get confidence scores
        real_confidence = probability[0] * 100
        fake_confidence = probability[1] * 100
        
        result = 'Real' if prediction == 0 else 'Fake'
        
        return {
            'prediction': result,
            'real_confidence': real_confidence,
            'fake_confidence': fake_confidence
        }
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

# Initialize the app
st.markdown('<h2 class="sub-header">🔄 Loading Model...</h2>', unsafe_allow_html=True)

# Load model
with st.spinner('Training model... This may take a few moments.'):
    model, tfidf_text, tfidf_title = train_model()

if model is None:
    st.error("Failed to load model. Please check your data files.")
    st.stop()

st.success("✅ Model loaded successfully!")

# Sample news for testing
sample_news = {
    "Real News Sample": {
        "title": "Bank Indonesia Holds Benchmark Interest Rate at 5.75%",
        "text": "Jakarta – Bank Indonesia (BI) has decided to maintain its benchmark interest rate at 5.75% during the latest Board of Governors' meeting. This decision is aimed at ensuring economic stability and controlling inflation, which is currently at 3.5%. The central bank also emphasized the importance of maintaining a conducive investment climate in the country."

    },
    "Fake News Sample": {
        "title": "Breaking: Aliens Land in Central Park, Demand Pizza",
        "text": "In a shocking turn of events, extraterrestrial beings have reportedly landed their spaceship in Central Park and are demanding the finest New York pizza. Witnesses claim the aliens spoke perfect English and complained about the lack of proper toppings on other planets. The mayor has called for an emergency meeting to discuss pizza negotiations."
    }
}

# Create two columns for input and results
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<h2 class="sub-header">📝 Input News Article</h2>', unsafe_allow_html=True)
    
    # Sample selection first
    st.markdown('<h3 class="sub-header">📋 Quick Test with Samples</h3>', unsafe_allow_html=True)
    selected_sample = st.selectbox("Choose a sample to test:", ["Manual Input"] + list(sample_news.keys()))
    
    # Set default values based on selection
    if selected_sample != "Manual Input":
        default_title = sample_news[selected_sample]["title"]
        default_text = sample_news[selected_sample]["text"]
    else:
        default_title = ""
        default_text = ""
    
    # Input fields
    title_input = st.text_input(
        "News Title",
        value=default_title,
        placeholder="Enter the news headline here...",
        help="Enter the title of the news article you want to analyze"
    )
    
    text_input = st.text_area(
        "News Content",
        value=default_text,
        height=300,
        placeholder="Enter the full news article content here...",
        help="Enter the main content of the news article"
    )
    
    # Predict button
    predict_button = st.button(
        "🔍 Analyze News",
        type="primary",
        use_container_width=True
    )

with col2:
    st.markdown('<h2 class="sub-header">📊 Analysis Results</h2>', unsafe_allow_html=True)
    
    if predict_button and title_input and text_input:
        with st.spinner('Analyzing news article...'):
            result = predict_news(title_input, text_input, model, tfidf_text, tfidf_title)
            
            if result:
                # Display prediction result
                prediction = result['prediction']
                real_conf = result['real_confidence']
                fake_conf = result['fake_confidence']
                
                # Determine box style based on prediction
                box_class = "real-news" if prediction == "Real" else "fake-news"
                icon = "✅" if prediction == "Real" else "❌"
                
                st.markdown(f"""
                <div class="prediction-box {box_class}">
                    <h3 style="margin: 0; text-align: center;">
                        {icon} Prediction: <strong>{prediction} News</strong>
                    </h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence scores
                st.markdown("### Confidence Scores")
                
                # Real news confidence
                st.markdown("**Real News Confidence:**")
                st.progress(real_conf / 100)
                st.write(f"{real_conf:.2f}%")
                
                # Fake news confidence
                st.markdown("**Fake News Confidence:**")
                st.progress(fake_conf / 100)
                st.write(f"{fake_conf:.2f}%")
                
                # Additional information
                st.markdown("### Analysis Details")
                st.info(f"""
                **Article Length:** {len(text_input)} characters

                **Title Length:** {len(title_input)} characters

                **Analysis Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                """)
                
                # Explanation
                if prediction == "Real":
                    st.success("""
                    ✅ **This article appears to be REAL news**

                    The model has high confidence that this is legitimate news content.
                    """)
                else:
                    st.error("""
                    ❌ **This article appears to be FAKE news**

                    The model has detected patterns commonly associated with misinformation.
                    """)
    
    elif predict_button:
        st.warning("⚠️ Please enter both title and content to analyze the news article.")
    
    else:
        st.info("""
        👆 **How to use:**

        1. Choose "Manual Input" to enter your own news
        2. Or select a sample from the dropdown to test
        3. Enter/modify the news title and content
        4. Click 'Analyze News' to get results
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>🤖 Powered by Machine Learning | Built with Streamlit</p>
    <p>⚠️ Disclaimer: This tool is for educational purposes. Always verify news from multiple sources.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar with additional information
with st.sidebar:
    st.markdown("### 📊 Model Information")
    st.info("""
    **Algorithm:** Logistic Regression

    **Features:** TF-IDF Vectorization

    **Text Processing:** Title + Content

    **Training Data:** 40,000+ articles
    """)
    
    st.markdown("### 🎯 How it Works")
    st.markdown("""
    1. **Text Preprocessing:** Clean and normalize text
    2. **Feature Extraction:** Convert text to numerical features
    3. **Model Prediction:** Use trained ML model
    4. **Confidence Scoring:** Calculate prediction certainty
    """)
    
    st.markdown("### 📈 Performance Tips")
    st.markdown("""
    - Provide complete article text for better accuracy
    - Include the original headline
    - Longer articles generally yield more reliable results
    - Use english language content for optimal performance
    - Avoid using slang or overly complex language
    """)
