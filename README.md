**README.md**:

---

# 🏠 Sentiment Analysis of NCCU Housing Tweets

**Combining Tweepy + Bag of Words (BoW) + VADER**

## 📌 Project Goals

This project uses **Twitter data** to analyze student discussions about **housing availability at North Carolina Central University (NCCU)**.

* **Bag of Words (BoW)** identifies common themes and topics (e.g., *housing*, *dorms*, *shortage*, *waitlist*).
* **VADER** (Valence Aware Dictionary for Sentiment Reasoning) captures emotional tone (e.g., *frustration*, *relief*, *excitement*).

🔑 **Together** these methods let us see patterns such as:

* “**shortage**” appearing mostly in **negative tweets**.
* “**moving in**” or “**new dorm**” appearing in **positive tweets**.

---

## ⚙️ Setup & Installation

1. **Install Required Packages**

   ```bash
   pip install tweepy nltk pandas matplotlib wordcloud python-dotenv
   ```

2. **Import Libraries**

   ```python
   import tweepy
   import pandas as pd
   import re
   from nltk.sentiment.vader import SentimentIntensityAnalyzer
   from collections import Counter
   from wordcloud import WordCloud
   import matplotlib.pyplot as plt
   import nltk
   nltk.download('vader_lexicon')
   ```

---

## 🔐 Step 1: Connect to Twitter API

For security, store your API keys in a `.env` file:

```
TWITTER_CONSUMER_KEY=your_key
TWITTER_CONSUMER_SECRET=your_secret
TWITTER_ACCESS_TOKEN=your_token
TWITTER_ACCESS_SECRET=your_token_secret
```

Load environment variables in your notebook:

```python
from dotenv import load_dotenv
import os

load_dotenv()
consumer_key = os.environ["TWITTER_CONSUMER_KEY"]
consumer_secret = os.environ["TWITTER_CONSUMER_SECRET"]
access_token = os.environ["TWITTER_ACCESS_TOKEN"]
access_token_secret = os.environ["TWITTER_ACCESS_SECRET"]
```

---

## 🔎 Step 2: Build Twitter Query

We filter for tweets mentioning NCCU housing concerns:

```text
(#NCCU OR #nccu) (housing OR dorms OR "residential life" OR "on campus housing" OR "lack of housing") (Central) -is:retweet
```

* `#NCCU OR #nccu` → captures both hashtag variations
* Housing keywords → *housing, dorms, residential life, lack of housing*
* `Central` → ensures tweets are about NCCU (often called *Central*)
* `-is:retweet` → removes retweets (to avoid duplicates)

---

## 📥 Step 3: Collect Tweets

```python
query = '#NCCU (housing OR dorms OR central OR shortage OR residential) -is:retweet'
tweets = tweepy.Cursor(api.search_tweets, q=query, lang="en", tweet_mode="extended").items(200)

data = [tweet.full_text for tweet in tweets]
df = pd.DataFrame(data, columns=["tweet"])
```

---

## 🧹 Step 4: Preprocess Tweets

Clean tweets for analysis:

```python
def clean_tweet(text):
    text = re.sub(r"http\S+", "", text)      # remove links
    text = re.sub(r"@\w+", "", text)         # remove mentions
    text = re.sub(r"#", "", text)            # remove hashtags
    text = re.sub(r"[^A-Za-z\s]", "", text)  # remove punctuation/numbers
    return text.lower().strip()

df["clean_tweet"] = df["tweet"].apply(clean_tweet)
```

---

## 📝 Step 5: Bag of Words (BoW)

Extract frequent words to identify main topics:

```python
all_words = " ".join(df["clean_tweet"]).split()
word_freq = Counter(all_words)

common_words = pd.DataFrame(word_freq.most_common(15), columns=["word", "count"])
print(common_words)
```

---

## 💬 Step 6: Sentiment Analysis with VADER

Analyze tone of tweets:

```python
sia = SentimentIntensityAnalyzer()

df["sentiment"] = df["clean_tweet"].apply(lambda x: sia.polarity_scores(x)["compound"])
df["label"] = df["sentiment"].apply(
    lambda x: "positive" if x > 0.05 else ("negative" if x < -0.05 else "neutral")
)
```

---

## 📊 Step 7: Visualization

Plot sentiment distribution:

```python
sentiment_counts = df["label"].value_counts()
sentiment_counts.plot(kind="bar", color=["green", "red", "gray"])
plt.title("Sentiment of NCCU Housing Tweets")
plt.xlabel("Sentiment")
plt.ylabel("Number of Tweets")
plt.show()
```

---

## 🚀 Outcomes

* **BoW** → reveals trending housing-related topics.
* **VADER** → classifies tweets as **positive**, **negative**, or **neutral**.
* **Combined insights** → shows which housing issues spark negative sentiment (e.g., *shortages*) and which generate positive sentiment (e.g., *new dorms*).

---

Would you like me to also add an example **output screenshot** (e.g., sample bar chart + top 15 words table) into the README so it’s presentation-ready?
