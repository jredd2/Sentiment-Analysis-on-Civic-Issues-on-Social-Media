# ================================
# 7. Sentiment Analysis (VADER)
# ================================
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()
df["sentiment"] = df["clean_tweet"].apply(lambda x: sia.polarity_scores(x)["compound"])
df["label"] = df["sentiment"].apply(
    lambda x: "positive" if x > 0.05 else ("negative" if x < -0.05 else "neutral")
)

# ================================
# 8. Thematic Coding
# ================================
# Expanded themes based on student sentiments regarding campus housing:
themes = {
    "Housing Frustration": [
        "shortage", "waitlist", "overcrowded", "problem", 
        "struggling", "frustrated", "cramped", "expensive"
    ],
    "Quality Amenities": [
        "modern", "comfortable", "well-maintained", "clean", 
        "quality", "updated", "nice"
    ],
    "Community Support": [
        "roommate", "friend", "community", "together", 
        "support", "shared", "social"
    ],
    "Effective Administration": [
        "responsive", "helpful", "administration", "policy", 
        "service", "quick", "administrative"
    ],
    "Facilities Issues": [
        "dorm", "maintenance", "broken", "building", 
        "infrastructure", "facilities", "old"
    ]
}

def assign_theme(tweet):
    tweet_lower = tweet.lower()
    # Collect all matching themes; a tweet might touch on multiple themes.
    matched_themes = []
    for theme, keywords in themes.items():
        for kw in keywords:
            if kw in tweet_lower:
                matched_themes.append(theme)
                break  # Stop checking keywords for a theme once a match is found.
    # If no themes were found, return "Other"
    if not matched_themes:
        return "Other"
    # Optionally, join multiple themes; here we join with a comma.
    return ", ".join(matched_themes)

df["theme"] = df["clean_tweet"].apply(assign_theme)
