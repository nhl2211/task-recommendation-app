import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from datetime import datetime

# Load dataset
df = pd.read_csv("task_recommendation_dataset.csv")

# TF-IDF Vectorization with N-grams for improved text representation
vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['Description'])
cosine_sim = cosine_similarity(tfidf_matrix)

# Streamlit UI setup
st.set_page_config(page_title="Advanced Task Recommendation System")
st.title("🧠 AI-Powered Task Recommendation System")
st.markdown("This system recommends tasks to team members based on task similarity, sentiment, and past history.")

# Sidebar filters
with st.sidebar:
    st.header("🔍 Filter Options")
    users = df["Assigned To"].unique().tolist()
    selected_user = st.selectbox("Select a user:", users)
    filter_priority = st.multiselect("Select priority levels:", df["Priority"].unique(), default=df["Priority"].unique())
    filter_category = st.multiselect("Select categories:", df["Category"].unique(), default=df["Category"].unique())

# Filtered tasks based on sidebar
filtered_df = df[(df['Priority'].isin(filter_priority)) & (df['Category'].isin(filter_category))]

# Recommend button
if st.button("🔎 Get Personalized Recommendations"):
    user_tasks = df[df['Assigned To'] == selected_user].index.tolist()
    candidate_tasks = filtered_df[filtered_df['Assigned To'] != selected_user].index.tolist()

    recommendations = []
    for i in candidate_tasks:
        sim_scores = [cosine_sim[i][j] for j in user_tasks if j in cosine_sim[i]]
        if sim_scores:
            avg_score = sum(sim_scores) / len(sim_scores)
            recommendations.append((i, avg_score))

    top_recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)[:5]
    result_rows = []

    st.markdown(f"### 📌 Top 5 Task Suggestions for **{selected_user}**")
    for idx, score in top_recommendations:
        task = df.iloc[idx]
        description = task['Description']
        sentiment_score = TextBlob(description).sentiment.polarity
        sentiment = "😊 Positive" if sentiment_score > 0.2 else "😟 Negative" if sentiment_score < -0.2 else "😐 Neutral"

        st.markdown(f"- **Task ID {task['Task ID']}** ({task['Category']} - {task['Priority']})<br>"
                    f"  ➤ *{description}*<br>"
                    f"  🔁 Similarity Score: `{score:.2f}`<br>"
                    f"  💬 Sentiment: **{sentiment}**<br>"
                    f"  📅 Deadline: {task['Deadline']}",
                    unsafe_allow_html=True)

        result_rows.append({
            "Task ID": task['Task ID'],
            "Description": description,
            "Category": task['Category'],
            "Priority": task['Priority'],
            "Similarity": round(score, 2),
            "Sentiment": sentiment,
            "Deadline": task['Deadline']
        })

    result_df = pd.DataFrame(result_rows)
    csv = result_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Recommendations", data=csv, file_name=f"recommendations_{selected_user}.csv", mime="text/csv")

  
