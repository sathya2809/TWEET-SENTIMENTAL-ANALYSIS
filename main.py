import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go
from PIL import Image

# Set page config
st.set_page_config(
    page_title="Tweet Sentiment Analysis",
    page_icon="🎭",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTextInput > div > div > input {
        font-size: 1.2rem;
    }
    .sentiment-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    model = joblib.load('sentiment_analysis_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer

# Function to predict sentiment
def predict_sentiment(text, model, vectorizer):
    # Transform text
    text_vectorized = vectorizer.transform([text])
    # Predict
    prediction = model.predict(text_vectorized)[0]
    # Get probability scores
    proba = model.predict_proba(text_vectorized)[0]
    return prediction, proba

def create_gauge_chart(probability, sentiment):
    colors = {'Positive': 'green', 'Neutral': 'gray', 'Negative': 'red'}
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Confidence Score for {sentiment}"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': colors.get(sentiment, 'gray')},
            'steps': [
                {'range': [0, 33], 'color': "lightgray"},
                {'range': [33, 66], 'color': "gray"},
                {'range': [66, 100], 'color': "darkgray"}
            ],
        }
    ))
    
    fig.update_layout(height=250)
    return fig

def main():
    # Load models
    model, vectorizer = load_models()
    
    # Header
    st.title("🎭 Tweet Sentiment Analysis")
    st.markdown("Analyze the sentiment of your text in real-time!")
    
    # Text input
    text_input = st.text_area("Enter your text here:", height=100)
    
    if text_input:
        # Get prediction
        sentiment, probabilities = predict_sentiment(text_input, model, vectorizer)
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Sentiment Analysis Result")
            sentiment_color = {
                'Positive': 'background-color: #90EE90',
                'Neutral': 'background-color: #F0F0F0',
                'Negative': 'background-color: #FFB6C1'
            }
            
            st.markdown(f"""
                <div style="{sentiment_color.get(sentiment)}; padding: 20px; border-radius: 10px;">
                    <h3 style="text-align: center; margin: 0;">
                        {sentiment}
                    </h3>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            # Get the highest probability
            max_prob = max(probabilities)
            # Create and display gauge chart
            fig = create_gauge_chart(max_prob, sentiment)
            st.plotly_chart(fig, use_container_width=True)
        
        # Display probability distribution
        st.subheader("Sentiment Distribution")
        labels = ['Negative', 'Neutral', 'Positive']
        fig = go.Figure(data=[go.Bar(
            x=labels,
            y=probabilities * 100,
            marker_color=['#FFB6C1', '#F0F0F0', '#90EE90']
        )])
        fig.update_layout(
            yaxis_title="Probability (%)",
            xaxis_title="Sentiment",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Add example section
    st.write('---')
    st.write('### Example Statements:')

    examples = {
        'Positive Statements': [
            "I absolutely love this product! It's amazing!",
            "The weather is perfect today, feeling blessed!",
            "Just got promoted at work, so excited!",
            "This is the best day of my life!",
            "The team did an outstanding job on the project!"
        ],
        'Neutral Statements': [
            "The weather is cloudy today.",
            "I'm going to the store.",
            "The meeting is scheduled for tomorrow.",
            "The train arrives at 3 PM.",
            "This book has 200 pages."
        ],
        'Negative Statements': [
            "This is the worst experience ever!",
            "I'm really disappointed with the service.",
            "The product broke after one day.",
            "I hate waiting in long queues!",
            "This movie was a complete waste of time."
        ]
    }

    for sentiment, statements in examples.items():
        st.write(f"#### {sentiment}")
        for statement in statements:
            # Get sentiment prediction for example
            vec = vectorizer.transform([statement])
            pred = model.predict(vec)[0]
            prob = np.max(model.predict_proba(vec))
            
            # Display example with its prediction
            expander = st.expander(f"'{statement}'")
            with expander:
                st.write(f"Predicted Sentiment: {pred.upper()}")
                st.write(f"Confidence: {prob:.2%}")

if __name__ == "__main__":
    main()
