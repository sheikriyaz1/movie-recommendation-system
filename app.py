import pandas as pd
import streamlit as st

# Page config
st.set_page_config(page_title="Movie Recommender", page_icon="🎬")

# Title
st.title("🎬 Movie Recommendation System")
st.markdown("### Get movie recommendations using AI 🎯")

# Load data
@st.cache_data
def load_data():
    column_names = ['user_id', 'item_id', 'rating', 'timestamp']
    data = pd.read_csv('ml-100k/u.data', sep='\t', names=column_names)

    movie_names = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1', header=None)
    movie_names = movie_names[[0, 1]]
    movie_names.columns = ['item_id', 'title']

    data = pd.merge(data, movie_names, on='item_id')

    movie_matrix = data.pivot_table(index='user_id', columns='title', values='rating')

    return movie_matrix

movie_matrix = load_data()

# Movie selection
movie_list = movie_matrix.columns.tolist()
selected_movie = st.selectbox("Select a movie:", movie_list)

# Recommendation button
if st.button("Recommend"):

    with st.spinner("Finding best recommendations... 🔍"):

        movie_ratings = movie_matrix[selected_movie]

        similar_movies = movie_matrix.corrwith(movie_ratings)

        corr_df = pd.DataFrame(similar_movies, columns=['correlation'])
        corr_df.dropna(inplace=True)
        corr_df = corr_df[corr_df['correlation'] < 1.0]

        recommendations = corr_df.sort_values('correlation', ascending=False).head(10)

    st.success("Top Recommendations 🎉")

    # Display results
    for i, movie in enumerate(recommendations.index, 1):
        st.write(f"{i}. {movie}")