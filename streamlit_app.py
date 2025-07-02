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
st.set_page_config(page_title="AI Task Recommender", layout="wide")
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5em 1em;
    }
    .stMarkdown h3 {
        color: #2c3e50;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("# 🤖 AI-Powered Task Recommendation System")
st.markdown("Effortlessly recommend relevant tasks to your team members based on task content and past work.")

# Sidebar filters
with st.sidebar:
    st.header("🔍 Customize Filters")
    users = df["Assigned To"].unique().tolist()
    selected_user = st.selectbox("👤 Select a team member:", users)
    filter_priority = st.multiselect("🚦 Priority:", df["Priority"].unique(), default=df["Priority"].unique())
    filter_category = st.multiselect("🗂️ Category:", df["Category"].unique(), default=df["Category"].unique())

# Filtered tasks based on sidebar
filtered_df = df[(df['Priority'].isin(filter_priority)) & (df['Category'].isin(filter_category))]

# Recommendation UI layout
col1, col2 = st.columns([3, 2])
with col1:
    if st.button("🎯 Get My Recommendations"):
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

        st.subheader(f"📌 Top 5 Task Suggestions for {selected_user}")
        for idx, score in top_recommendations:
            task = df.iloc[idx]
            description = task['Description']
            sentiment_score = TextBlob(description).sentiment.polarity
            sentiment = "😊 Positive" if sentiment_score > 0.2 else "😟 Negative" if sentiment_score < -0.2 else "😐 Neutral"

            st.markdown(f"""
            <div style='background-color:#ffffff;border:1px solid #dcdcdc;padding:10px;border-radius:10px;margin-bottom:10px;'>
                <b>Task ID:</b> {task['Task ID']}<br>
                <b>Description:</b> {description}<br>
                <b>Category:</b> {task['Category']} | <b>Priority:</b> {task['Priority']}<br>
                <b>Similarity Score:</b> {score:.2f} | <b>Sentiment:</b> {sentiment}
            </div>
            """, unsafe_allow_html=True)

            result_rows.append({
                "Task ID": task['Task ID'],
                "Description": description,
                "Category": task['Category'],
                "Priority": task['Priority'],
                "Similarity": round(score, 2),
                "Sentiment": sentiment
            })

        result_df = pd.DataFrame(result_rows)
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Recommendations", data=csv, file_name=f"recommendations_{selected_user}.csv", mime="text/csv")

with col2:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712109.png", width=250)
    st.markdown("""
        <div style='font-size: 0.9em; padding-top: 1em;'>
            <p>This tool helps project managers and team leads find task matches for their team using machine learning.</p>
            <p><b>Tip:</b> Try changing the priority and category filters for better suggestions!</p>
        </div>
    """, unsafe_allow_html=True)
