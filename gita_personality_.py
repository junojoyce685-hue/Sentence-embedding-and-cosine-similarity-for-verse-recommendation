# gita_personality_.py
import streamlit as st
import pandas as pd
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(
    page_title="GitaPersona",
    page_icon="📿",
    layout="wide"
)

st.title("📿 YOGASCOPE: a “lens” into your Karma, Bhakti, and Jnana tendencies")
st.write("Answer the questions below to discover your dominant Yoga type and get personalized Gita wisdom!")

# -------------------------------
# Load questions
# -------------------------------
with open("quiz_questions.json", "r") as f:
    questions = json.load(f)["questions"]

# -------------------------------
# Load dataset
# -------------------------------
verses = pd.read_csv("gita_labeled_dataset.csv")  # Columns: Chapter, Verse, Shloka, EngMeaning, Yoga_Types

# -------------------------------
# Load embedding model
# -------------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------------
# User inputs
# -------------------------------
st.header("📝 Answer the following questions:")

responses = []
for q in questions:
    answer = st.text_area(q["question"], height=80)
    responses.append(answer if answer else "")

if st.button("Submit"):
    st.success("Processing your personality...")

    # -------------------------------
    # Combine responses
    # -------------------------------
    combined_text = " ".join(responses)

    # -------------------------------
    # Yoga profiles
    # -------------------------------
    profiles = {
        "karma": "action duty responsibility discipline work effort service contribution",
        "bhakti": "devotion faith love surrender trust compassion spiritual connection prayer",
        "jnana": "knowledge wisdom reflection truth understanding philosophy awareness inquiry"
    }

    # -------------------------------
    # Embeddings & similarity
    # -------------------------------
    user_embedding = model.encode([combined_text])
    profile_texts = list(profiles.values())
    profile_embeddings = model.encode(profile_texts)

    scores = cosine_similarity(user_embedding, profile_embeddings)[0]
    result_scores = dict(zip(profiles.keys(), scores))

    # -------------------------------
    # Radar chart
    # -------------------------------
    labels = list(result_scores.keys())
    values = list(result_scores.values())
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    # Scale slightly for better visibility
    scale_factor = 1.2
    values_scaled = [v * scale_factor for v in values[:-1]] + [v * scale_factor for v in values[:1]]

    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
    ax.plot(angles, values_scaled, 'o-', linewidth=3, color='blue', label='Your Scores')
    ax.fill(angles, values_scaled, color='skyblue', alpha=0.4)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0, 1.2)
    ax.set_title("Your Gita Personality Radar", fontsize=16)

    st.pyplot(fig)

    # -------------------------------
    # Determine dominant personality
    # -------------------------------
    personality = max(result_scores, key=result_scores.get)
    st.subheader(f"Dominant Personality: {personality.upper()} YOGA")

    if personality == "karma":
        st.write("You are driven by action, discipline, and fulfilling your duty.")
    elif personality == "bhakti":
        st.write("You are guided by devotion, trust, and spiritual connection.")
    else:
        st.write("You seek wisdom, reflection, and deeper understanding.")

    # -------------------------------
    # Recommend verses
    # -------------------------------
    filtered_verses = verses[verses["Yoga_Type"].str.lower().str.contains(personality)]
    if filtered_verses.empty:
        st.warning("No verses found for this yoga type in the dataset.")
    else:
        verse_embeddings = model.encode(filtered_verses["EngMeaning"].tolist())
        verse_scores = cosine_similarity(user_embedding, verse_embeddings)
        top_indices = verse_scores.argsort()[0][-3:][::-1]

        st.subheader("📖 Recommended Gita Verses:\n")
        for idx in top_indices:
            verse = filtered_verses.iloc[idx]
            st.markdown(f"**Chapter {verse['Chapter']} Verse {verse['Verse']}**")
            st.markdown(f"*Shloka:* {verse['Shloka']}")
            st.markdown(f"*Meaning:* {verse['EngMeaning']}")
            st.markdown(f"*Why this verse was recommended:* This verse aligns with your **{personality.upper()} YOGA** tendencies.\n---")
