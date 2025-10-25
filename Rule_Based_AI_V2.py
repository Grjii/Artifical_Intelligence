
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 19:01:17 2025

@author: grgej
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("sentimentdataset.csv")

keywords = ["cute", "gaming", "book", "exploring", "productive", "workout", "milestone", "friends", "language", "growth",
            "morning", "discussion", "today", "blues", "winter",
            "book", "baking", "school", "highschool", "school",
            "newyear", "reflection", "friend", "hiddengems", "newyear", "gaming", "reading", "virtualreality", "challengeaccepted", "brunchw",
            "traffic", "debate", "health", "winterblues", "highschoolreality",
            "highschoolexams", "environmentalefforts", "highschoolbakes","highschoolprojects", "highschool",
            "brightened", "petantics", "playful", "day", "amused",
            "wrong", "highschoolweather", "bad", "tech", "series",
            "celebration", "family", "garden", "laugh", "sunny",
            "challenges", "dissapointment", "lonely", "goodbye", "college"
            ]

def keyword_engagementcheck(row):
    score = 0
    text_content = str(row.get("Text", ""))
    text_words = text_content.lower().split()
    
    matches = sum(1 for word in text_words for kw in keywords if kw in word)
    score += matches * 2
    
    sentiment = str(row["Sentiment"]).strip().lower()
    if sentiment == "happy":
        score += 2
    elif sentiment == "positive":
        score += 1
    elif sentiment == "amusement":
        score += 0.75
    elif sentiment == "neutral":
        score += 0.5
    elif sentiment in ["negative", "bad", "sad"]:
        score -= 1
    elif sentiment == "hate":
        score -= 2

    if score >= 4:
        return "Viral"
    elif score >= 2:
        return "High Engagement"
    elif score >= 1:
        return "Moderate Engagement"
    else:
        return "Low Engagement"

df["EngagementLevel"] = df.apply(keyword_engagementcheck, axis=1)
df["Predicted_Viral"] = df["EngagementLevel"].apply(lambda x: 1 if x in ["Viral", "High Engagement"] else 0)
df["Actual_Viral"] = df["Sentiment"].apply(lambda x: 1 if str(x).strip().lower() == "positive" else 0)

y_true, y_pred = df["Actual_Viral"], df["Predicted_Viral"]
metrics = {
    "Accuracy": accuracy_score(y_true, y_pred),
    "Precision": precision_score(y_true, y_pred, zero_division=0),
    "Recall": recall_score(y_true, y_pred, zero_division=0),
    "F1 Score": f1_score(y_true, y_pred, zero_division=0)
}

# Print metrics to console
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

plt.bar(metrics.keys(), metrics.values())
plt.ylim(0, 1)
plt.show()
