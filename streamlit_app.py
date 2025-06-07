import streamlit as st
import joblib
import gdown
import os
import pandas as pd

# === SETTING FILE ID GOOGLE DRIVE ===
RECOMMENDER_FILE_ID = "1T51NDCDCiCgOgpXzZ7oVodFcJHeTP4ST"  # content_recommender.pkl
SIMILARITY_FILE_ID = "1fSDXnCN_b1AjZmrFQ-CjLqX9snuRf9cK"   # similarity_data.pkl
DATASET_PATH = "netflix_preprocessed.csv"  # Pastikan file ini tersedia

# === DOWNLOAD & LOAD ===
@st.cache_resource
def load_content_recommender():
    path = "recommender.pkl"
    if not os.path.exists(path):
        gdown.download(f"https://drive.google.com/uc?id={RECOMMENDER_FILE_ID}", path, quiet=False)
    return joblib.load(path)

@st.cache_resource
def load_similarity_data():
    path = "data_similarity.pkl"
    if not os.path.exists(path):
        gdown.download(f"https://drive.google.com/uc?id={SIMILARITY_FILE_ID}", path, quiet=False)
    return joblib.load(path)

@st.cache_data
def load_full_dataset():
    return pd.read_csv(DATASET_PATH)

# Kolom yang akan ditampilkan
columns_to_show = [
    'type', 'title', 'director', 'cast', 'country', 'date_added',
    'release_year', 'rating', 'listed_in', 'description',
    'duration_minutes', 'duration_seasons'
]

# === MAIN FUNCTION ===
def main():
    st.title("Netflix Recommender System 🎬")
    st.markdown("Enter a Netflix movie title below to get similar movie recommendations.")

    title = st.text_input("Enter a movie title:")
    search_clicked = st.button("Get Recommended Movies")

    # Load resources
    content_recommender = load_content_recommender()
    similarity_data = load_similarity_data()
    full_df = load_full_dataset()

    # Extract components
    cosine_similarities = similarity_data["cosine_similarities"]
    indices = similarity_data["indices"]
    netflix_title = similarity_data["netflix_title"]

    if search_clicked and title:
        if title in set(netflix_title):
            movie_details_df = full_df[full_df['title'] == title][columns_to_show]
            if movie_details_df.empty:
                st.warning("Details not found in the full dataset.")
            else:
                st.subheader("Selected Movie Details")
                st.dataframe(movie_details_df, use_container_width=True)

            st.subheader("Recommended Titles:")
            try:
                # Panggil fungsi dengan 3 parameter
                recommendations = content_recommender(title, cosine_similarities, indices)
                for i, rec_title in enumerate(recommendations, 1):
                    with st.expander(f"{i}. {rec_title}"):
                        rec_details_df = full_df[full_df['title'] == rec_title][columns_to_show]
                        if not rec_details_df.empty:
                            st.dataframe(rec_details_df, use_container_width=True)
                        else:
                            st.warning(f"Details for '{rec_title}' not found.")
            except Exception as e:
                st.error(f"❌ Error while generating recommendations: {e}")
        else:
            st.error("❌ Movie title not found in model title list.")

# Run main
if __name__ == "__main__":
    main()
