# Customer-churn-and-retention-analysis-C&R

# MAIN PROJECT PROTOTYPE
!pip cache purge  
!pip install scikit-learn==1.3.0 nltk pandas google-colab mlxtend  

import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from google.colab import files
import re
from nltk.corpus import stopwords
from mlxtend.frequent_patterns import apriori, association_rules 


# Download NLTK VADER resources
nltk.download('vader_lexicon')
nltk.download('stopwords')

# Initialize the VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

# Step 1: Upload CSV file containing customer feedback
uploaded = files.upload()
csv_path = list(uploaded.keys())[0]

# Step 2: Load CSV and automatically identify the feedback column
feedback_df = pd.read_csv(csv_path)
feedback_column_name = None
for col in feedback_df.columns:
    if any(keyword in col.lower() for keyword in ['feedback', 'review', 'comment', 'text', 'response']):
        feedback_column_name = col
        break
if feedback_column_name:
    feedback_texts = feedback_df[feedback_column_name].dropna().tolist()
else:
    raise ValueError("Feedback column not found. Ensure it includes keywords like 'feedback', 'review', or 'comment'.")

# Step 3: Preprocess feedback texts
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
    text = text.lower()  # Convert to lowercase
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

feedback_texts = [preprocess_text(text) for text in feedback_texts]

# Step 4: Perform Sentiment Analysis using VADER
feedback_sentiments = []
for text in feedback_texts:
    sentiment_scores = sid.polarity_scores(text)
    if sentiment_scores['compound'] >= 0.05:
        sentiment_label = 'Positive'
    elif sentiment_scores['compound'] <= -0.05:
        sentiment_label = 'Negative'
    else:
        sentiment_label = 'Neutral'
    feedback_sentiments.append((text, sentiment_label, sentiment_scores))

# Step 5: Analyze feedback data for churn and retention insights
churn_feedbacks = [fs for fs in feedback_sentiments if fs[1] == 'Negative']
retention_feedbacks = [fs for fs in feedback_sentiments if fs[1] == 'Positive']

# Step 6: Generate Insights for the Product Manager
print("Customer Feedback Analysis and Marketing Insights:")

# Churn Analysis
print("\n1. Customer Churn Analysis:")
if churn_feedbacks:
    print(f" - Total negative feedbacks: {len(churn_feedbacks)}")
    churn_texts = [cf[0] for cf in churn_feedbacks]
    churn_vectorizer = CountVectorizer(max_features=10, stop_words='english')
    churn_word_counts = churn_vectorizer.fit_transform(churn_texts).toarray().sum(axis=0)
    churn_common_words = churn_vectorizer.get_feature_names_out()
    churn_word_freqs = dict(zip(churn_common_words, churn_word_counts))
    print(" - Common themes in negative feedbacks (Potential Reasons for Churn):")
    for word, freq in churn_word_freqs.items():
        print(f"   * {word.capitalize()}: {freq} mentions")
    print(" - Recommendation: Focus on improving these areas to reduce churn.")
else:
    print(" - No significant negative feedbacks detected.")

# Retention Analysis
print("\n2. Customer Retention Analysis:")
if retention_feedbacks:
    print(f" - Total positive feedbacks: {len(retention_feedbacks)}")
    retention_texts = [rf[0] for rf in retention_feedbacks]
    retention_vectorizer = CountVectorizer(max_features=10, stop_words='english')
    retention_word_counts = retention_vectorizer.fit_transform(retention_texts).toarray().sum(axis=0)
    retention_common_words = retention_vectorizer.get_feature_names_out()
    retention_word_freqs = dict(zip(retention_common_words, retention_word_counts))
    print(" - Common themes in positive feedbacks (Strengths to Retain):")
    for word, freq in retention_word_freqs.items():
        print(f"   * {word.capitalize()}: {freq} mentions")
    print(" - Recommendation: Emphasize these strengths in marketing to improve retention.")
else:
    print(" - No significant positive feedbacks detected.")

# Overall Insights
print("\n3. Overall Sentiment Analysis:")
positive_count = sum(1 for fs in feedback_sentiments if fs[1] == 'Positive')
negative_count = sum(1 for fs in feedback_sentiments if fs[1] == 'Negative')
neutral_count = sum(1 for fs in feedback_sentiments if fs[1] == 'Neutral')
total_feedbacks = len(feedback_sentiments)

print(f" - Total feedbacks analyzed: {total_feedbacks}")
print(f" - Positive: {positive_count} ({(positive_count / total_feedbacks) * 100:.2f}%)")
print(f" - Negative: {negative_count} ({(negative_count / total_feedbacks) * 100:.2f}%)")
print(f" - Neutral: {neutral_count} ({(neutral_count / total_feedbacks) * 100:.2f}%)")
print("\nFinal Recommendation: Address churn factors to reduce negative sentiment and strengthen marketing around positive attributes to enhance retention.")
