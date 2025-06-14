# 📰 Fake News Detector

A machine learning-powered application to detect fake news using Natural Language Processing and Logistic Regression. This project includes both a Jupyter notebook for model development and an interactive Streamlit web application for real-time news analysis.

## 🎯 Project Overview

This project aims to combat misinformation by providing an automated tool to classify news articles as either **Real** or **Fake**. The system uses advanced text preprocessing, TF-IDF vectorization, and machine learning to analyze news content and provide confidence scores for predictions.

## ✨ Features

- **Machine Learning Model**: Logistic Regression with TF-IDF features
- **Text Analysis**: Advanced preprocessing and feature extraction
- **Interactive UI**: Beautiful Streamlit web interface
- **Confidence Scores**: Detailed prediction confidence visualization
- **Sample Testing**: Pre-loaded examples for immediate testing
- **Modern Design**: Responsive and user-friendly interface
- **Real-time Analysis**: Instant prediction results

## 🗂️ Project Structure

```
MachineLearning/
├── AOL_Machine_Learning.ipynb    # Main development notebook
├── DEPLOY.ipynb                  # Deployment setup notebook
├── fake_news_streamlit.py        # Streamlit web application
├── README.md                     # Project documentation
├── Fake.csv                      # Fake news dataset
├── True.csv                      # Real news dataset
└── deploy apps/                  # Code with much simple preprocessing/tuning
    ├── fake_news_app.py
    ├── Fake.csv
    └── True.csv
```

## 🚀 Quick Start

### Prerequisites

Make sure you have Python 3.7+ installed on your system.

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd MachineLearning
   ```

2. **Install required packages**
   
   **Option A: Using pip (recommended)**
   ```bash
   pip install streamlit scikit-learn pandas numpy matplotlib seaborn scipy
   ```
   
   **Option B: Using conda**
   ```bash
   conda install streamlit scikit-learn pandas numpy matplotlib seaborn scipy
   ```
   
   **Option C: Using requirements.txt (if available)**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```bash
   python -c "import streamlit, sklearn, pandas, numpy; print('All packages installed successfully!')"
   ```

4. **Run the application**
   ```bash
   streamlit run fake_news_streamlit.py
   ```

5. **Open in browser**
   - Navigate to `http://localhost:8501`
   - Start analyzing news articles!

## 🛠️ Usage

### Web Application

1. **Launch the app** using the command above
2. **Choose input method**:
   - Select "Manual Input" to enter your own news
   - Choose from pre-loaded samples for quick testing
3. **Enter news details**:
   - Add the news title in the title field
   - Paste the article content in the text area
4. **Analyze**: Click the "🔍 Analyze News" button
5. **View results**: See the prediction with confidence scores

### Jupyter Notebook

Open `AOL_Machine_Learning.ipynb` to:
- Explore the data preprocessing pipeline
- Understand the model training process
- Analyze model performance metrics
- Experiment with different parameters

## 🚀 Deployment Options

### Local Development
Follow the Quick Start guide above for local development and testing.

### Cloud Deployment

#### Streamlit Cloud (Recommended)
1. Push your code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy with one click

#### Heroku
1. Create a `Procfile`:
   ```
   web: streamlit run fake_news_streamlit.py --server.port=$PORT --server.address=0.0.0.0
   ```
2. Deploy using Heroku CLI or GitHub integration

#### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "fake_news_streamlit.py", "--server.address", "0.0.0.0"]
```

## 🔬 Technical Details

### Data Processing Pipeline

1. **Data Loading**
   - Combines fake and real news datasets
   - Handles missing values and duplicates
   - Balances classes if needed

2. **Text Preprocessing**
   - Converts text to lowercase
   - Removes URLs, emails, and HTML tags
   - Eliminates special characters and digits
   - Normalizes whitespace

3. **Feature Engineering**
   - TF-IDF vectorization for title and content
   - N-gram extraction (unigrams and bigrams)
   - Feature combination and selection

4. **Model Training**
   - Logistic Regression classifier
   - Stratified train-test split
   - Hyperparameter optimization

### Model Performance

- **Algorithm**: Logistic Regression
- **Features**: TF-IDF vectors (title + content)
- **Training Data**: 40,000+ news articles
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC

#### Performance Benchmarks
- **Accuracy**: ~95%+ on test data
- **Precision**: High precision for both real and fake news detection
- **Recall**: Balanced recall across classes
- **Processing Speed**: <1 second per article analysis
- **Memory Usage**: ~50MB for model and vectorizers

## 📊 Dataset

The project uses two datasets:

- **Fake.csv**: Collection of fake news articles
- **True.csv**: Collection of verified real news articles

Each dataset contains:
- `title`: News headline
- `text`: Article content
- `subject`: News category
- `date`: Publication date

## 🎨 User Interface

The Streamlit application features:

- **Split Layout**: Input on the left, results on the right
- **Color Coding**: Green for real news, red for fake news
- **Progress Bars**: Visual confidence score representation
- **Responsive Design**: Works on desktop and mobile devices
- **Information Sidebar**: Model details and usage tips

## 📈 Model Evaluation

The notebook includes comprehensive evaluation:

- Confusion matrix visualization
- ROC curve analysis
- Classification report
- Feature importance analysis
- Performance metrics comparison

## 🔧 Configuration

### Model Parameters

```python
# TF-IDF Configuration
max_features_text = 5000
max_features_title = 2000
min_df = 2
max_df = 0.8
ngram_range = (1, 2)

# Logistic Regression Configuration
max_iter = 1000
C = 1.0
solver = 'liblinear'
random_state = 42
```

### Customization

You can modify the model by:
- Adjusting TF-IDF parameters in the notebook
- Changing the classification algorithm
- Adding new preprocessing steps
- Including additional features

## 🚨 Important Notes

- **Data Requirements**: Ensure `Fake.csv` and `True.csv` are in the project directory
- **Model Training**: First run may take a few minutes to train the model
- **Compatibility**: Works with Streamlit versions 0.84.0+
- **Educational Use**: This tool is for educational purposes; always verify news from multiple sources

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 🔧 Troubleshooting

### Common Issues

**Issue**: `AttributeError: module 'streamlit' has no attribute 'rerun'`
- **Solution**: This app is compatible with older Streamlit versions and doesn't use `st.rerun()`. Update to Streamlit 1.28+ for best experience

**Issue**: `FileNotFoundError: Fake.csv or True.csv`
- **Solution**: Ensure both CSV files are in the same directory as `fake_news_streamlit.py`

**Issue**: Model training takes too long
- **Solution**: 
  - Reduce the dataset size for testing
  - Adjust TF-IDF `max_features` parameters
  - Use a smaller sample of data during development

**Issue**: Low prediction accuracy
- **Solution**: 
  - Check data quality and preprocessing steps
  - Verify dataset integrity
  - Consider feature engineering improvements

**Issue**: Streamlit app won't start
- **Solution**:
  ```bash
  # Check Streamlit installation
  streamlit --version
  
  # Reinstall if needed
  pip uninstall streamlit
  pip install streamlit
  
  # Run with verbose output
  streamlit run fake_news_streamlit.py --logger.level=debug
  ```

**Issue**: Memory errors during model training
- **Solution**:
  - Reduce `max_features` in TF-IDF configuration
  - Use sparse matrices efficiently
  - Consider training on a subset of data first

**Issue**: Package compatibility issues
- **Solution**: Use the provided `requirements.txt` with specific versions:
  ```bash
  pip install -r requirements.txt
  ```

### Performance Optimization

- **Caching**: The app uses `@st.cache_data` and `@st.cache_resource` for optimal performance
- **Memory Management**: Efficient sparse matrix handling for large datasets
- **Loading Time**: First load may take 30-60 seconds to train the model

## Future Enhancements

- [ ] Support for multiple languages
- [ ] Real-time news scraping
- [ ] Advanced deep learning models
- [ ] Batch processing capabilities
- [ ] API endpoint development
- [ ] Mobile application
- [ ] Browser extension

## Application Preview

The Streamlit application provides an intuitive interface with:

### Main Features
- **Input Section**: Clean form for entering news title and content
- **Sample Selection**: Quick testing with pre-loaded examples
- **Real-time Analysis**: Instant prediction with confidence visualization
- **Results Display**: Color-coded predictions with detailed confidence metrics

### Sample News Examples
The app includes carefully curated examples of both real and fake news for testing:
- Real news from reputable sources with verifiable information
- Fake news examples with common misinformation patterns
- Diverse topics covering politics, technology, health, and more

---

## 💡 Usage Tips

1. **For Best Results**: Enter complete news articles with clear titles
2. **Testing**: Use the sample news to understand how the model works
3. **Verification**: Always cross-check results with multiple reliable sources
4. **Educational Purpose**: Use this tool to understand how AI can help identify misinformation patterns

## Educational Values

This project demonstrates:
- **Text Processing**: Advanced NLP techniques for news analysis
- **Machine Learning**: Classification algorithms and model evaluation
- **Web Development**: Interactive applications with Streamlit
- **Data Science Pipeline**: From raw data to deployed application

---

*Last updated: June 2025*
