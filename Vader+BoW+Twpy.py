# ================================
# 1. Install and Import Libraries
# ================================
# !pip install tweepy nltk pandas matplotlib wordcloud plotly dash dash-bootstrap-components python-dotenv

import os
import re
import pandas as pd
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
import tweepy

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from dotenv import load_dotenv

import dash
from dash import dcc, html, dash_table, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px

# Download NLTK data
nltk.download('vader_lexicon')

# ================================
# 2. Load Environment Variables
# ================================
# .env file should have:
# TWITTER_CONSUMER_KEY=your_key
# TWITTER_CONSUMER_SECRET=your_secret
# TWITTER_ACCESS_TOKEN=your_token
# TWITTER_ACCESS_SECRET=your_token_secret

# Load .env variables
load_dotenv()

consumer_key = os.getenv("TWITTER_CONSUMER_KEY")
consumer_secret = os.getenv("TWITTER_CONSUMER_SECRET")
access_token = os.getenv("TWITTER_ACCESS_TOKEN")
access_token_secret = os.getenv("TWITTER_ACCESS_SECRET")

# Authenticate using OAuth 1.0a
auth = tweepy.OAuth1UserHandler(consumer_key, consumer_secret, access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)
# ================================
# 3. Connect to Twitter API
# ================================
auth = tweepy.OAuth1UserHandler(
    consumer_key, consumer_secret, access_token, access_token_secret
)
api = tweepy.API(auth)

# ================================
# 4. Collect Tweets
# ================================
query = '#NCCU (housing OR dorms OR central OR shortage OR residential) -is:retweet'
tweets = tweepy.Cursor(api.search_tweets, q=query, lang="en", tweet_mode="extended").items(200)

data = []
for tweet in tweets:
    data.append(tweet.full_text)

df = pd.DataFrame(data, columns=["tweet"])

# ================================
# 5. Preprocess Tweets
# ================================
def clean_tweet(text):
    text = re.sub(r"http\S+", "", text)  # remove links
    text = re.sub(r"@\w+", "", text)     # remove mentions
    text = re.sub(r"#", "", text)        # remove hashtag symbol
    text = re.sub(r"[^A-Za-z\s]", "", text)  # remove punctuation/numbers
    return text.lower().strip()

df["clean_tweet"] = df["tweet"].apply(clean_tweet)

# ================================
# 6. Bag of Words
# ================================
all_words = " ".join(df["clean_tweet"]).split()
word_freq = Counter(all_words)

common_words = pd.DataFrame(word_freq.most_common(15), columns=["word", "count"])
print("Top words:\n", common_words)

# Word Cloud Visualization
wc = WordCloud(width=800, height=400, background_color="white").generate(" ".join(all_words))
plt.figure(figsize=(10,5))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.show()

# ================================
# 7. Sentiment Analysis (VADER)
# ================================
sia = SentimentIntensityAnalyzer()

df["sentiment"] = df["clean_tweet"].apply(lambda x: sia.polarity_scores(x)["compound"])
df["label"] = df["sentiment"].apply(
    lambda x: "positive" if x > 0.05 else ("negative" if x < -0.05 else "neutral")
)

# ================================
# 8. Thematic Coding
# ================================
themes = {
    "Frustration": ["shortage", "waitlist", "overcrowded", "problem", "struggling", "frustrated"],
    "Resilience": ["found", "moving in", "solution", "worked out", "finally"],
    "Community Support": ["roommate", "friend", "helped", "shared"],
    "Facilities": ["dorm", "old", "maintenance", "broken", "quality", "building"]
}

def assign_theme(tweet):
    for theme, keywords in themes.items():
        for kw in keywords:
            if kw in tweet.lower():
                return theme
    return "Other"

df["theme"] = df["clean_tweet"].apply(assign_theme)

# ================================
# 9. Summarize Data
# ================================
theme_summary = (
    df.groupby(["theme", "label"])
    .size()
    .reset_index(name="count")
)

theme_summary["percent"] = theme_summary.groupby("theme")["count"].apply(
    lambda x: (x / x.sum()) * 100
)

print(theme_summary)

# ================================
# 10. Dashboard with Dash
# ================================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

# Figures
overall_fig = px.pie(
    df, names="label", title="Overall Sentiment of NCCU Housing Tweets"
)

theme_fig = px.bar(
    theme_summary,
    x="theme",
    y="percent",
    color="label",
    barmode="group",
    title="Sentiment by Theme (Housing Shortage at NCCU)"
)

# Layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H2("NCCU Housing Sentiment Dashboard"), className="mb-4")
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id="overall", figure=overall_fig), md=6),
        dbc.Col(dcc.Graph(id="theme", figure=theme_fig), md=6)
    ]),
    dbc.Row([
        dbc.Col(html.H4("Sample Tweets by Theme"), className="mt-4")
    ]),
    dbc.Row([
        dbc.Col(dash_table.DataTable(
            id="tweet-table",
            columns=[{"name": c, "id": c} for c in ["clean_tweet", "label", "theme"]],
            page_size=5,
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left'}
        ), md=12)
    ])
])

# Interactivity: Filter tweets by clicking bar in theme chart
@app.callback(
    Output("tweet-table", "data"),
    Input("theme", "clickData")
)
def display_tweets(clickData):
    if clickData:
        selected_theme = clickData["points"][0]["x"]
        return df[df["theme"] == selected_theme][["clean_tweet", "label", "theme"]].to_dict("records")
    else:
        return df[["clean_tweet", "label", "theme"]].sample(5).to_dict("records")

# Run app
if __name__ == "__main__":
    app.run_server(debug=True)
