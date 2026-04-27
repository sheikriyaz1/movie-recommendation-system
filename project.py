import pandas as pd

# column names
column_names = ['user_id', 'item_id', 'rating', 'timestamp']

# load dataset (change path if needed)
data = pd.read_csv('ml-100k/u.data', sep='\t', names=column_names)
print(data.head())
# Create user-item matrix
movie_matrix = data.pivot_table(index='user_id', columns='item_id', values='rating')

print(movie_matrix.head())
# Choose one movie ID (example: 50)
# Choose movie
# Count number of ratings per movie
ratings_count = data.groupby('item_id').size()

# Keep only popular movies (at least 50 ratings)
popular_movies = ratings_count[ratings_count > 50].index

# Filter matrix
filtered_matrix = movie_matrix[popular_movies]

# Choose movie
target_movie = 50

# Get ratings
movie_ratings = filtered_matrix[target_movie]

# Correlation
similar_movies = filtered_matrix.corrwith(movie_ratings)

# DataFrame
corr_df = pd.DataFrame(similar_movies, columns=['correlation'])

# Clean
corr_df.dropna(inplace=True)
corr_df = corr_df[corr_df['correlation'] < 1.0]

# Sort
recommendations = corr_df.sort_values('correlation', ascending=False)

print(recommendations.head(10))
# Load movie names
movie_names = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1', header=None)

movie_names = movie_names[[0, 1]]
movie_names.columns = ['item_id', 'title']

# Merge
recommendations = recommendations.merge(movie_names, on='item_id')

# Show results
print("\nRecommended Movies:\n")
print(recommendations[['title', 'correlation']].head(10))