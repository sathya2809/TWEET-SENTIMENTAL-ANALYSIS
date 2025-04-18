# Tweet Sentiment Analysis Project
![Image](https://github.com/user-attachments/assets/54a1cd86-26b1-4408-b370-1265b4f4663e)
![Image](https://github.com/user-attachments/assets/9c07a51b-1dbd-4e96-9e4e-3fdcf5a31c7f)
![Image](https://github.com/user-attachments/assets/3617e7e2-c15c-44c1-b164-12bc0524855e)

## Overview
This project implements a machine learning model to analyze the sentiment of tweets, classifying them as positive, negative, or neutral. The system uses TF-IDF vectorization and Logistic Regression to achieve high accuracy in sentiment prediction.

## Features
- Real-time sentiment analysis of text input
- Support for three sentiment categories: Positive, Neutral, and Negative
- Interactive web interface using Streamlit
- Visualization of sentiment probabilities
- Pre-trained model with high accuracy

## Technical Stack
- Python 3.x
- Scikit-learn
- NLTK
- Streamlit
- Pandas
- NumPy
- Joblib

## Model Performance
The sentiment analysis model achieves the following metrics:
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-score: 1.0000

## Project Structure
```
TWEET SENTIMENTAL ANALYSIS/
├── main.py                      # Streamlit web application
├── sentiment_analysis_model.pkl # Trained model
├── tfidf_vectorizer.pkl        # TF-IDF vectorizer
├── TWEET_SENTIMENT_CLASSIFICATION.ipynb # Training notebook
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/tweet-sentiment-analysis.git
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit application:
```bash
streamlit run main.py
```

## Usage
1. Open the web interface at http://localhost:8501
2. Enter text in the input field
3. View the sentiment analysis results with confidence scores
4. Explore example statements provided in the interface

## Example Code
```python
import joblib

# Load the model and vectorizer
model = joblib.load('sentiment_analysis_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Example text
text = "This is an amazing product! I love it."

# Transform text using vectorizer
text_vectorized = vectorizer.transform([text])

# Predict sentiment
prediction = model.predict(text_vectorized)[0]
```

## Model Training
The model was trained on a cleaned dataset of labeled tweets using:
- TF-IDF vectorization for feature extraction
- Logistic Regression for classification
- 80-20 train-test split
- Stratified sampling to handle class imbalance

## Contributing
Feel free to contribute to this project by:
1. Forking the repository
2. Creating a new branch
3. Making your changes
4. Submitting a pull request

## License
This project is licensed under the MIT License - see the LICENSE file for details.



## Contact
[Sathya S] - [22am056@kpriet.ac.in]
