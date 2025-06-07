import streamlit as st
import joblib
import gdown
import os
import pandas as pd

# === DEFINISI CLASS HARUS ADA SEBELUM LOAD PICKLE ===
class content_recommender:
    def __init__(self, df, cosine_similarities, indices):
        self.df = df
        self.cosine_similarities = cosine_similarities
        self.indices = indices

    def recommend(self, name):
        idx = self.indices[name]
        sim_scores = list(enumerate(self.cosine_similarities[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[2:7]  # Ambil rekomendasi top 5 (kecuali diri sendiri)
        netflix_indices = [i[0] for i in sim_scores]
        displayed_columns = ['title', 'listed_in', 'description', 'rating']
        return self.df.iloc[netflix_indices][displayed_columns]

# === FUNGSI DOWNLOAD FILE DARI GOOGLE DRIVE ===
def download_file(file_id, filename):
    if not os.path.exists(filename):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, filename, quiet=False)
    return filename

# === CACHE RESOURCE UNTUK LOAD PICKLE ===
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
    # Pastikan file CSV ada di folder yang sama
    return pd.read_csv("netflix_preprocessed.csv")

# === MAIN STREAMLIT APP ===
def main():
    st.title("Netflix Recommender System 🎬")
    st.markdown("Masukkan judul film Netflix untuk mendapatkan rekomendasi film serupa.")

    # Load data dan model
    recommender = load_recommender()
    similarity_data = load_similarity_data()
    full_df = load_full_dataset()

    cosine_similarities = similarity_data["cosine_similarities"]
    indices = similarity_data["indices"]
    netflix_title = similarity_data["netflix_title"]

    title = st.text_input("Masukkan judul film:")
    if st.button("Dapatkan Rekomendasi"):
        if title in netflix_title:
            # Tampilkan detail film yang dipilih
            movie_details = full_df[full_df['title'] == title][
                ['type', 'title', 'director', 'cast', 'country', 'date_added',
                 'release_year', 'rating', 'listed_in', 'description',
                 'duration_minutes', 'duration_seasons']
            ]
            st.subheader("Detail Film yang Dipilih")
            st.dataframe(movie_details, use_container_width=True)

            # Tampilkan rekomendasi
            st.subheader("Rekomendasi Film Serupa")
            try:
                recommendations = recommender.recommend(title)
                for idx, row in recommendations.iterrows():
                    with st.expander(f"{row['title']}"):
                        st.markdown(f"**Genre:** {row['listed_in']}")
                        st.markdown(f"**Rating:** {row['rating']}")
                        st.markdown(f"**Deskripsi:** {row['description']}")
            except Exception as e:
                st.error(f"Terjadi error saat mengambil rekomendasi: {e}")
        else:
            st.error("Judul film tidak ditemukan di data model.")

if __name__ == "__main__":
    main()
