import streamlit as st
import pickle
import gdown
import os
import pandas as pd

# Load model dari Google Drive
@st.cache_resource
def load_model_from_drive():
    file_id = "1Y6lvZEMwZ9-fMe3TbGZNYDXBklhrOJji"
    output_path = "recommender_model2.pkl"

    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)

    with open(output_path, "rb") as f:
        return pickle.load(f)

# Load dataset setelah proses preprocessing
@st.cache_data
def load_full_dataset():
    df = pd.read_csv("netflix_preprocessed.csv")
    return df

def content_recommender(title, cosine_similarities, indices, df, top_n=5):
    if title not in indices:
        return []

    idx = indices[title]
    sim_scores = list(enumerate(cosine_similarities[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n + 1]  # Skip movie itu sendiri

    recommended_indices = [i[0] for i in sim_scores]
    displayed_column = ['title', 'listed_in', 'description', 'rating']
    return df.iloc[recommended_indices][displayed_column]


columns_to_show = ['title', 'listed_in', 'description', 'rating']

def main():
    # Load model dan data
    model_data = load_model_from_drive()
    netflix_title_series = model_data["netflix_title"]  # Series of titles
    cosine_similarities = model_data["cosine_similarities"]
    indices = model_data["indices"]

    full_df = load_full_dataset()

    st.title("Netflix Movie Recommender")
    st.markdown("Enter a Netflix movie title below to get similar movie recommendations.")

    user_input = st.text_input("Enter a movie title:")
    title = user_input.title() # untuk memastikan input-an dari user selalu diawali dengan huruf kapital, sekalipun dengan 2 kata.
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
                full_df
            )
            for _, row in recommendations.iterrows():
                with st.expander(f"🎬 {row['title']}"):
                    st.markdown(f"""
                    Genre: {row['listed_in']}  
                    Rating: {row['rating']}  
                    Description: {row['description']}
                    """)
        else:
            st.error("Movie title not found in model title list.")

if __name__ == "__main__":
    main()
