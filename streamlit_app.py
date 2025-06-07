import streamlit as st
import joblib
import gdown
import os
import pandas as pd

# === Google Drive file IDs ===
RECOMMENDER_FILE_ID = "1iKVh9RmfsBcUzC_xwvfMv5eJwckgVrrL"  # your recommender.pkl
SIMILARITY_FILE_ID = "1p1rY2KoUDbH8ujIAuxb9Nh5GFIckDlwV"   # your similarity_data.pkl
DATASET_PATH = "netflix_preprocessed.csv"  # Pastikan file CSV ini sudah tersedia lokal

# === Download files from Google Drive if not exist ===
@st.cache_resource
def download_file(file_id, filename):
    if not os.path.exists(filename):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, filename, quiet=False)
    return filename

# === Load pickle recommender (class instance) ===
@st.cache_resource
def load_recommender():
    path = download_file(RECOMMENDER_FILE_ID, "recommender.pkl")
    return joblib.load(path)

# === Load similarity data dict ===
@st.cache_resource
def load_similarity_data():
    path = download_file(SIMILARITY_FILE_ID, "similarity_data.pkl")
    return joblib.load(path)

# === Load Netflix dataset ===
@st.cache_data
def load_full_dataset():
    return pd.read_csv(DATASET_PATH)

# Kolom yang akan ditampilkan
columns_to_show = [
    'type', 'title', 'director', 'cast', 'country', 'date_added',
    'release_year', 'rating', 'listed_in', 'description',
    'duration_minutes', 'duration_seasons'
]

def main():
    st.title("Netflix Recommender System 🎬")
    st.markdown("Enter a Netflix movie title below to get similar movie recommendations.")

    title = st.text_input("Enter a movie title:")
    search_clicked = st.button("Get Recommended Movies")

    # Load models and data
    recommender = load_recommender()               # class instance with method recommend()
    similarity_data = load_similarity_data()        # dict with cosine_similarities, indices, netflix_title
    full_df = load_full_dataset()

    cosine_similarities = similarity_data["cosine_similarities"]
    indices = similarity_data["indices"]
    netflix_title = similarity_data["netflix_title"]

    if search_clicked and title:
        if title in set(netflix_title):
            # Tampilkan detail movie yang dicari
            movie_details_df = full_df[full_df['title'] == title][columns_to_show]
            if movie_details_df.empty:
                st.warning("Details not found in the full dataset.")
            else:
                st.subheader("Selected Movie Details")
                st.dataframe(movie_details_df, use_container_width=True)

            # Tampilkan rekomendasi
            st.subheader("Recommended Titles:")
            try:
                recommendations_df = recommender.recommend(title)
                for i, (_, row) in enumerate(recommendations_df.iterrows(), start=1):
                    with st.expander(f"{i}. {row['title']}"):
                        # Tampilkan kolom yang ingin ditampilkan untuk setiap rekomendasi
                        st.markdown(f"""
                        **Genre:** {row['listed_in']}  
                        **Rating:** {row['rating']}  
                        **Description:** {row['description']}
                        """)
            except Exception as e:
                st.error(f"❌ Error while generating recommendations: {e}")
        else:
            st.error("❌ Movie title not found in model title list.")

if __name__ == "__main__":
    main()
