import streamlit as st
import dill as pickle
import gdown
import os
import pandas as pd

# Load model dari Google Drive
@st.cache_resource
def load_model_from_drive():
    file_id = "1uARTcSmf--15RMbvBxwP7TJFONlISYvK"
    output_path = "recommender_model.pkl"

    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)

    with open(output_path, "rb") as f:
        return pickle.load(f)

# Load dataset lengkap dari CSV
@st.cache_data
def load_full_dataset():
    df = pd.read_csv("netflix_preprocessed.csv")  # Ganti path jika perlu
    return df

# Fungsi rekomendasi manual (ganti fungsi dari pickle)
def content_recommender(title, cosine_similarities, indices, netflix_title, top_n=5):
    if title not in indices:
        return []

    idx = indices[title]
    sim_scores = list(enumerate(cosine_similarities[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n + 1]  # Skip movie itu sendiri

    recommended_indices = [i[0] for i in sim_scores]
    return netflix_title.iloc[recommended_indices].tolist()

# Kolom yang akan ditampilkan
columns_to_show = [
    'type', 'title', 'director', 'cast', 'country', 'date_added',
    'release_year', 'rating', 'listed_in', 'description',
    'duration_minutes', 'duration_seasons'
]

# Load model dan data
model_data = load_model_from_drive()
netflix_title_series = model_data["netflix_title"]  # Series of titles
cosine_similarities = model_data["cosine_similarities"]
indices = model_data["indices"]

full_df = load_full_dataset()

# UI Streamlit
st.title("Netflix Movie Recommender")
st.markdown("Enter a Netflix movie title below to get similar movie recommendations.")

title = st.text_input("Enter a movie title:")
search_clicked = st.button("Get Recommended Movies")

if search_clicked and title:
    # Cek apakah title ada di netflix_title_series
    if title in set(netflix_title_series):
        # Ambil detail dari full_df
        movie_details_df = full_df[full_df['title'] == title][columns_to_show]
        if movie_details_df.empty:
            st.warning("Details not found in the full dataset.")
        else:
            st.subheader("Selected Movie Details")
            st.dataframe(movie_details_df, use_container_width=True)

        # Rekomendasi
        st.subheader("Recommended Titles:")
        recommendations = content_recommender(
            title,
            cosine_similarities,
            indices,
            netflix_title_series
        )

        for i, rec_title in enumerate(recommendations, 1):
            with st.expander(f"{i}. {rec_title}"):
                rec_details_df = full_df[full_df['title'] == rec_title][columns_to_show]
                if not rec_details_df.empty:
                    st.dataframe(rec_details_df, use_container_width=True)
                else:
                    st.warning(f"Details for '{rec_title}' not found.")
    else:
        st.error("❌ Movie title not found in model title list.")
