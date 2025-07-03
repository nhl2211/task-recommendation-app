import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob

# Load dataset
df = pd.read_csv("task_recommendation_dataset.csv")
df['Description'] = df['Description'].astype(str).str.lower()

# TF-IDF Vectorization
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

import re

def clean_text(text):
    words = re.findall(r'\b\w+\b', text.lower())
    return ' '.join([w for w in words if w not in stop_words])


df['Cleaned_Description'] = df['Description'].apply(clean_text)
vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['Cleaned_Description'])
cosine_sim = cosine_similarity(tfidf_matrix)

# Streamlit UI setup
st.set_page_config(page_title="AI Task Recommender", layout="wide")
st.markdown("""
    <style>
    body {
        background-color: #ffffff;
        color: #333333;
    }
    .main {
        background-color: #ffffff;
    }
    .stButton>button {
        background-color: #4b79a1;
        color: white;
        font-weight: 600;
        border-radius: 6px;
        padding: 0.6em 1.2em;
        transition: 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #3a5e84;
    }
    .recommendation-box {
        background-color: #f7faff;
        border: 1px solid #dde5f1;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        color: #333333;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
    }
    .header-text {
        font-size: 1.7rem;
        color: #2e5c9a;
    }
    .info-panel {
        font-size: 0.9em;
        padding-top: 1em;
        color: #555555;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='header-text'>🤖 AI-Powered Task Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("Use AI to generate personalized task suggestions for team members based on task descriptions, category, and priority.")

# Sidebar filters
with st.sidebar:
    st.markdown("""
    <style>
    .sidebar-content label, .sidebar-content div {
        font-size: 0.92rem;
        color: #2e5c9a;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("## 🎛️ Filters", unsafe_allow_html=True)
    selected_user = st.selectbox("Select a team member:", users)
    filter_priority = st.multiselect("Priority:", df["Priority"].unique(), default=df["Priority"].unique())
    filter_category = st.multiselect("Category:", df["Category"].unique(), default=df["Category"].unique())

    if st.button("🔁 Reset Filters"):
        st.experimental_rerun()
    filter_priority = st.multiselect("Priority:", df["Priority"].unique(), default=df["Priority"].unique())
    filter_category = st.multiselect("Category:", df["Category"].unique(), default=df["Category"].unique())

# Filtered tasks based on sidebar
filtered_df = df[(df['Priority'].isin(filter_priority)) & (df['Category'].isin(filter_category))]

# Recommendation UI layout
col1, col2 = st.columns([3, 2])
with col1:
    if st.button("🎯 Get My Recommendations"):
        user_tasks = df[df['Assigned To'] == selected_user].index.tolist()

        result_rows = []
        if not user_tasks:
            st.warning(f"{selected_user} has no past tasks. Displaying top global recommendations.")
            fallback_tasks = filtered_df.groupby('Priority', group_keys=False).apply(lambda x: x.sample(min(2, len(x)))).sample(n=min(6, len(filtered_df)))
            tasks_to_display = fallback_tasks.to_dict('records')
        else:
            candidate_tasks = filtered_df[filtered_df['Assigned To'] != selected_user].index.tolist()
            recommendations = []
            for i in candidate_tasks:
                sim_scores = [cosine_sim[i, j] for j in user_tasks]
                if sim_scores:
                    avg_score = sum(sim_scores) / len(sim_scores)
                    recommendations.append((i, avg_score))

            top_recommendations = sorted(recommendations, key=lambda x: (df.iloc[x[0]]['Priority'], -x[1]), reverse=True)[:6]
            tasks_to_display = []
            for idx, score in top_recommendations:
                task = df.iloc[idx].to_dict()
                task['Similarity'] = round(score, 2)
                tasks_to_display.append(task)

        st.subheader(f"📌 Top 6 Task Suggestions for {selected_user}")
        for task in tasks_to_display:
            description = task['Description']
            sentiment_score = TextBlob(description).sentiment.polarity
            sentiment = "Positive" if sentiment_score > 0.2 else "Negative" if sentiment_score < -0.2 else "Neutral"
            task_priority_display = f"<span style='color:red;font-weight:bold;'>🔴 {task['Priority']}</span>" if task['Priority'] == "High" else task['Priority']
            similarity_display = f" | <b>Similarity Score:</b> {task['Similarity']}" if 'Similarity' in task else ""

            st.markdown(f"""
            <div class='recommendation-box'>
                <b>Task ID:</b> {task['Task ID']}<br>
                <b>Description:</b> {description}<br>
                <b>Category:</b> {task['Category']} | <b>Priority:</b> {task_priority_display}<br>
                <b>Sentiment:</b> {sentiment}{similarity_display}
            </div>
            """, unsafe_allow_html=True)

            row = {
                "Task ID": task['Task ID'],
                "Description": description,
                "Category": task['Category'],
                "Priority": task['Priority'],
                "Sentiment": sentiment
            }
            if 'Similarity' in task:
                row['Similarity'] = task['Similarity']
            result_rows.append(row)

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
