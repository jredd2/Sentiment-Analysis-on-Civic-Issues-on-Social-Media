# ===========================
# UNC HOUSING SENTIMENT DASHBOARD
# ===========================

# --- Imports ---
import snscrape.modules.twitter as sntwitter
import pandas as pd
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px
from dash import Dash, dcc, html

# --- (1) Tweet Scraping ---
search_query = (
    "(housing OR dorm OR residence OR 'student housing' OR 'dorm shortage' OR 'roommate' OR 'off campus' OR 'housing' OR 'no housing') "
    "(UNC OR 'UNC Chapel Hill' OR #UNCC OR #uncc OR 'UNC Charlotte' OR #ncsu OR #NCSU OR 'NC State' OR 'North Carolina Central' OR #NCCU OR #nccu OR 'App State') "
    "since:2025-01-01 until:2025-10-15"
)

tweets = []
for i, tweet in enumerate(sntwitter.TwitterSearchScraper(search_query).get_items()):
    if i > 1200:  # limit to avoid excess
        break
    tweets.append([tweet.date, tweet.user.username, tweet.content, tweet.url])

df = pd.DataFrame(tweets, columns=['Date', 'User', 'Text', 'URL'])

# --- (2) Text Preprocessing with Stopwords ---
import nltk
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = [word for word in text.split() if word not in stop_words]
    return " ".join(tokens)

df['Clean_Text'] = df['Text'].apply(clean_text)

# --- (3) Sentiment Analysis ---
analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    scores = analyzer.polarity_scores(text)
    if scores['compound'] >= 0.05:
        return 'Positive'
    elif scores['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

df['Sentiment'] = df['Clean_Text'].apply(get_sentiment)
df['Compound'] = df['Clean_Text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

(4) Campus Detection ---
def detect_campus(text):
    text = text.lower()
    if 'chapel hill' in text or 'unc ' in text or 'UNC' in text:
        return 'UNC Chapel Hill'
    elif 'charlotte' in text or 'uncc' in text or 'UNCC' in text:
        return 'UNC Charlotte'
    elif 'nc state' in text or 'ncsu' in text or 'NCSU' in text:
        return 'NC State'
    elif 'central' in text or 'nccu' in text or 'NCCU' in text:
        return 'NC Central'
    elif 'app state' in text or 'appalachian' or appstate in text:
        return 'Appalachian State'
    else:
        return 'Other'

df['Campus'] = df['Text'].apply(detect_campus)

df['Campus'] = df['Text'].apply(detect_campus)

# --- (5) Theme Detection ---
def detect_theme(text):
    text = text.lower()
    if any(word in text for word in ['expensive', 'afford', 'rent', 'cost', 'budget']):
        return 'Affordability'
    elif any(word in text for word in ['full', 'crowded', 'no room', 'waitlist', 'shortage', 'overflow']):
        return 'Space Shortage'
    elif any(word in text for word in ['maintenance', 'mold', 'broken', 'leak']):
        return 'Maintenance'
    elif any(word in text for word in ['movein', 'new dorm', 'construction']):
        return 'New/Move-In'
    elif any(word in text for word in ['angry', 'frustrated', 'annoyed', 'mad', 'disappointed']):
        return 'Frustration'
    else:
        return 'General'

df['Theme'] = df['Clean_Text'].apply(detect_theme)

# --- (6) Generate WordClouds ---
for sentiment in ['Positive', 'Neutral', 'Negative']:
    subset = df[df['Sentiment'] == sentiment]
    text = " ".join(subset['Clean_Text'])
    if text:
        wc = WordCloud(width=800, height=400, background_color='white').generate(text)
        plt.figure(figsize=(8, 4))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"{sentiment} Tweets Word Cloud")
        plt.show()

# --- (7) Dashboard with Dash ---
app = Dash(__name__)

fig_sentiment = px.histogram(
    df, x='Campus', color='Sentiment', barmode='group',
    title='Sentiment Distribution by Campus'
)

fig_theme = px.histogram(
    df, x='Theme', color='Sentiment', barmode='group',
    title='Sentiment Distribution by Theme'
)

fig_pie = px.pie(df, names='Sentiment', title='Overall Sentiment Distribution')

app.layout = html.Div([
    html.H1("UNC Housing Sentiment Dashboard", style={'textAlign': 'center'}),
    html.Div([
        dcc.Graph(figure=fig_sentiment),
        dcc.Graph(figure=fig_theme),
        dcc.Graph(figure=fig_pie)
    ])
])

if __name__ == '__main__':
    app.run_server(debug=True)
