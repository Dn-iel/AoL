import streamlit as st
import joblib
import gdown
import os
import pandas as pd

# === DEFINISI CLASS YANG DIGUNAKAN SAAT MEMBUAT PICKLE ===
class content_recommender:
    def __init__(self, df, cosine_similarities, indices):
        self.df = df
        self.cosine_similarities = cosine_similarities
        self.indices = indices

    def recommend(self, name):
        if name not in self.indices:
            raise ValueError(f"Judul '{name}' tidak ditemukan.")
        idx = self.indices[name]
        sim_scores = list(enumerate(self.cosine_similarities[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:6]
        netflix_indices = [i[0] for i in sim_scores]
        displayed_column = ['title', 'listed_in', 'description', 'rating']
        return self.df.iloc[netflix_indices][displayed_column]

# === FUNGSI DOWNLOAD DARI GOOGLE DRIVE ===
def download_file(file_id, filename):
    if not os.path.exists(filename):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, filename, quiet=False)
    return filename

# === LOAD FILE PICKLE DAN DATASET ===
@st.cache_resource
def load_recommender():
    path = download_file("1iKVh9RmfsBcUzC_xwvfMv5eJwckgVrrL", "recommender.pkl")
    return joblib.load(path)

@st.cache_resource
def load_similarity_data():
    path = download_file("1p1rY2KoUDbH8ujIAuxb9Nh5GFIckDlwV", "data_similarity.pkl")
    return joblib.load(path)

@st.cache_data
def load_full_dataset():
    return pd.read_csv("netflix_preprocessed.csv")

# === STREAMLIT MAIN ===
def main():
    st.title("🎬 Netflix Recommender System")
    st.markdown("Masukkan judul film Netflix untuk mendapatkan rekomendasi film serupa.")

    # Load data
    recommender = load_recommender()
    similarity_data = load_similarity_data()
    full_df = load_full_dataset()

    # Ambil variabel penting dari data_similarity
    cosine_similarities = similarity_data["cosine_similarities"]
    indices = similarity_data["indices"]
    netflix_title = similarity_data["netflix_title"]

    # Input judul
    title = st.text_input("Masukkan judul film:")
    if st.button("Dapatkan Rekomendasi"):
        if title in netflix_title:
            # Tampilkan detail film
            movie_details = full_df[full_df['title'] == title][
                ['type', 'title', 'director', 'cast', 'country', 'date_added',
                 'release_year', 'rating', 'listed_in', 'description',
                 'duration_minutes', 'duration_seasons']
            ]
            st.subheader("🎞️ Detail Film")
            st.dataframe(movie_details, use_container_width=True)

            # Rekomendasi
            st.subheader("🎯 Rekomendasi Serupa")
            try:
                recommendations = recommender.recommend(title)
                for _, row in recommendations.iterrows():
                    with st.expander(f"{row['title']}"):
                        st.markdown(f"**Genre:** {row['listed_in']}")
                        st.markdown(f"**Rating:** {row['rating']}")
                        st.markdown(f"**Deskripsi:** {row['description']}")
            except Exception as e:
                st.error(f"Terjadi error saat mengambil rekomendasi: {e}")
        else:
            st.error("Judul tidak ditemukan dalam data model.")

if __name__ == "__main__":
    main()
