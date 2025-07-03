import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

st.set_page_config(page_title="Movie Recommender", layout="wide")

# -------------------------
# Load data
# -------------------------
@st.cache_data
def load_data():
    ratings = pd.read_csv("ratings.csv")
    movies = pd.read_csv("movies.csv")
    return ratings, movies

ratings, movies = load_data()

# -------------------------
# Build user-item matrix
# -------------------------
def create_matrix(df):
    N = len(df['userId'].unique())
    M = len(df['movieId'].unique())
    user_mapper = dict(zip(np.unique(df["userId"]), list(range(N))))
    movie_mapper = dict(zip(np.unique(df["movieId"]), list(range(M))))
    user_inv_mapper = dict(zip(list(range(N)), np.unique(df["userId"])))
    movie_inv_mapper = dict(zip(list(range(M)), np.unique(df["movieId"])))
    user_index = [user_mapper[i] for i in df['userId']]
    movie_index = [movie_mapper[i] for i in df['movieId']]
    X = csr_matrix((df["rating"], (movie_index, user_index)), shape=(M, N))
    return X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper

X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper = create_matrix(ratings)

# -------------------------
# KNN recommender
# -------------------------
def find_similar_movies(movie_id, k=10, metric='cosine'):
    neighbour_ids = []
    if movie_id not in movie_mapper:
        return []
    movie_ind = movie_mapper[movie_id]
    movie_vec = X[movie_ind]
    k += 1
    kNN = NearestNeighbors(n_neighbors=k, algorithm="brute", metric=metric)
    kNN.fit(X)
    movie_vec = movie_vec.reshape(1, -1)
    neighbour = kNN.kneighbors(movie_vec, return_distance=False)
    for i in range(k):
        n = neighbour.item(i)
        neighbour_ids.append(movie_inv_mapper[n])
    neighbour_ids.pop(0)
    return neighbour_ids

def recommend_movies_for_user(user_id, k=10):
    df1 = ratings[ratings['userId'] == user_id]
    if df1.empty:
        return None, []
    movie_id = df1[df1['rating'] == max(df1['rating'])]['movieId'].iloc[0]
    movie_titles = dict(zip(movies['movieId'], movies['title']))
    similar_ids = find_similar_movies(movie_id, k)
    recs = [movie_titles[i] for i in similar_ids if i in movie_titles]
    return movie_titles.get(movie_id, "Unknown Movie"), recs

# -------------------------
# UI Tabs
# -------------------------
tab1, tab2 = st.tabs(["🎬 Recommendations", "📊 Data Visualizations"])

# -------------------------
# Tab 1: Recommendations
# -------------------------
with tab1:
    st.title("🎥 Movie Recommendation System")
    user_id = st.number_input("Enter your User ID:", min_value=1, step=1, value=877)
    k = st.slider("Number of recommendations:", 1, 20, 10)

    if st.button("Get Recommendations"):
        watched, recs = recommend_movies_for_user(user_id, k)
        if watched is None:
            st.warning("Sorry, no data found for this user ID.")
        else:
            st.subheader(f"Since you watched **{watched}**, you might also like:")
            for movie in recs:
                st.write(f"- {movie}")

# -------------------------
# Tab 2: Data Visualizations
# -------------------------
with tab2:
    st.title("📊 Movie Dataset Insights")

    # Genre bar chart: number of ratings
    st.subheader("Number of Ratings per Genre")
    all_genres = []
    for genres in movies['genres']:
        all_genres.extend(genres.split('|'))
    genre_counts = pd.Series(all_genres).value_counts()

    fig1, ax1 = plt.subplots()
    genre_counts.plot(kind='bar', color='skyblue', ax=ax1)
    ax1.set_ylabel("Number of Movies")
    st.pyplot(fig1)

    # Average ratings by genre
    st.subheader("Average Movie Rating by Genre")
    genre_ratings = {}
    for genre in genre_counts.index:
        genre_movie_ids = movies[movies['genres'].str.contains(genre)]['movieId']
        genre_ratings[genre] = ratings[ratings['movieId'].isin(genre_movie_ids)]['rating'].mean()
    genre_ratings_series = pd.Series(genre_ratings).sort_values(ascending=False)

    fig2, ax2 = plt.subplots()
    genre_ratings_series.plot(kind='bar', color='coral', ax=ax2)
    ax2.set_ylabel("Average Rating")
    st.pyplot(fig2)

    # Most rated movies
    st.subheader("Top Most Rated Movies")
    top_movies = ratings['movieId'].value_counts().index

    # Merge safely to get titles
    top_movies_df = pd.DataFrame({'movieId': top_movies})
    top_movies_with_titles = top_movies_df.merge(movies[['movieId', 'title']], on='movieId', how='left')
    top_movies_with_titles = top_movies_with_titles.dropna(subset=['title']).head(10)

    # Now get counts aligned
    valid_movie_ids = top_movies_with_titles['movieId']
    valid_counts = ratings['movieId'].value_counts().loc[valid_movie_ids].values

    fig3, ax3 = plt.subplots()
    ax3.barh(top_movies_with_titles['title'][::-1], valid_counts[::-1], color='limegreen')
    ax3.set_xlabel("Number of Ratings")
    st.pyplot(fig3)
    ax3.barh(top_movies_with_titles['title'][::-1], valid_counts[::-1], color='limegreen')
    ax3.set_xlabel("Number of Ratings")
    st.pyplot(fig3)


    # Ratings distribution
    st.subheader("Distribution of All Ratings")
    fig4, ax4 = plt.subplots()
    ratings['rating'].plot(kind='hist', bins=10, color='purple', edgecolor='black', ax=ax4)
    ax4.set_xlabel("Rating")
    st.pyplot(fig4)
