import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob

# Load dataset
df = pd.read_csv("task_recommendation_dataset.csv")

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['Description'])
cosine_sim = cosine_similarity(tfidf_matrix)

# Streamlit UI setup
st.set_page_config(page_title="AI Task Recommender", layout="wide")
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #0066cc;
        color: white;
        font-weight: 600;
        border-radius: 6px;
        padding: 0.6em 1.2em;
        transition: 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #004c99;
    }
    .recommendation-box {
        background-color: #ffffff;
        border: 1px solid #d0d0d0;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .header-text {
        font-size: 1.5rem;
        color: #1c3d5a;
    }
    .info-panel {
        font-size: 0.9em;
        padding-top: 1em;
        color: #333333;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='header-text'>🤖 AI-Powered Task Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("Use AI to generate personalized task suggestions for team members based on task descriptions, category, and priority.")

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

        if not user_tasks:
            st.warning(f"{selected_user} has no past tasks. Displaying top global recommendations.")
            fallback_tasks = filtered_df.sample(n=min(6, len(filtered_df)))
            result_rows = []
            for _, task in fallback_tasks.iterrows():
                description = task['Description']
                sentiment_score = TextBlob(description).sentiment.polarity
                sentiment = "😊 Positive" if sentiment_score > 0.2 else "😟 Negative" if sentiment_score < -0.2 else "😐 Neutral"

                st.markdown(f"""
                <div class='recommendation-box'>
                    <b>Task ID:</b> {task['Task ID']}<br>
                    <b>Description:</b> {description}<br>
                    <b>Category:</b> {task['Category']} | <b>Priority:</b> <span style='color:red;font-weight:bold;'>🔴 {task['Priority']}</span><br>
                    <b>Sentiment:</b> {sentiment}
                </div>
                """, unsafe_allow_html=True)

                result_rows.append({
                    "Task ID": task['Task ID'],
                    "Description": description,
                    "Category": task['Category'],
                    "Priority": task['Priority'],
                    "Sentiment": sentiment
                })
        else:
            candidate_tasks = filtered_df[filtered_df['Assigned To'] != selected_user].index.tolist()

            recommendations = []
            for i in candidate_tasks:
                sim_scores = [cosine_sim[i][j] for j in user_tasks if j < cosine_sim.shape[1]]
                if sim_scores:
                    avg_score = sum(sim_scores) / len(sim_scores)
                    recommendations.append((i, avg_score))

            top_recommendations = sorted(recommendations, key=lambda x: (df.iloc[x[0]]['Priority'], -x[1]), reverse=True)[:6]
            result_rows = []

            st.subheader(f"📌 Top 6 Task Suggestions for {selected_user}")
            for idx, score in top_recommendations:
                task = df.iloc[idx]
                description = task['Description']
                sentiment_score = TextBlob(description).sentiment.polarity
                sentiment = "😊 Positive" if sentiment_score > 0.2 else "😟 Negative" if sentiment_score < -0.2 else "😐 Neutral"

                st.markdown(f"""
                <div class='recommendation-box'>
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
        st.download_button("📥 Download Recommendations", data=csv, file_name=f"recommendations_{selected_user}_final.csv", mime="text/csv")

with col2:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712109.png", width=250)
    st.markdown("""
        <div class='info-panel'>
            <p>This tool helps project managers and team leads assign relevant tasks using machine learning recommendations.</p>
            <p><b>Tip:</b> Adjust priority or category filters to explore different task matches.</p>
        </div>
    """, unsafe_allow_html=True)
