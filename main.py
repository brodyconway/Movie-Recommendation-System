import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sys

# Load data files from the script's directory
data_dir = Path(__file__).resolve().parent
ratings_df = pd.read_csv(data_dir / "ratings.csv")
movies_df = pd.read_csv(data_dir / "movies.csv")

# Reduce data size to prevent memory overload
# Keep only top 500 most rated movies
top_movies = ratings_df['movieId'].value_counts().head(500).index
ratings_df = ratings_df[ratings_df['movieId'].isin(top_movies)]

# Create user-movie pivot matrix with ratings (fill missing with 0)
user_movie_matrix = ratings_df.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# Cluster users using K-Means with 20 clusters (based on unique genres)
num_clusters = 20
kmeans = KMeans(n_clusters=num_clusters, random_state=1, n_init=10)
kmeans.fit(user_movie_matrix)
labels = kmeans.labels_

# Visualize clusters using PCA
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(user_movie_matrix)
plt.figure(figsize=(10, 6))
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='tab20', s=10)
plt.title("User Clusters (K-Means with K=20)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label='Cluster')
plt.tight_layout()
plt.show()

# Parse target userId from command-line
if len(sys.argv) < 2:
    print("Usage: python script.py <userId>")
    sys.exit(1)
target_user_id = int(sys.argv[1])
if target_user_id not in user_movie_matrix.index:
    print(f"User ID {target_user_id} not found in the data.")
    sys.exit(1)

# Find the cluster for the target user
user_idx = user_movie_matrix.index.get_loc(target_user_id)
user_cluster = labels[user_idx]

# Get other users in the same cluster
cluster_user_ids = user_movie_matrix.index[labels == user_cluster]
cluster_user_ids = [uid for uid in cluster_user_ids if uid != target_user_id]

# Ratings from other users in the cluster
cluster_ratings = ratings_df[ratings_df['userId'].isin(cluster_user_ids)]

# Exclude movies already rated by target user
seen_movies = set(ratings_df[ratings_df['userId'] == target_user_id]['movieId'])
cluster_ratings = cluster_ratings[~cluster_ratings['movieId'].isin(seen_movies)]

# Compute high rating counts and average ratings for each movie in cluster
high_ratings = cluster_ratings[cluster_ratings['rating'] >= 4.0]
high_counts = high_ratings.groupby('movieId')['rating'].count().reset_index(name='high_count')
avg_ratings = cluster_ratings.groupby('movieId')['rating'].mean().reset_index(name='avg_rating')
movie_stats = pd.merge(high_counts, avg_ratings, on='movieId', how='outer').fillna(0)

# Sort by high_count (desc), then avg_rating (desc)
movie_stats = movie_stats.sort_values(by=['high_count', 'avg_rating'], ascending=False)
top_10_ids = movie_stats.head(10)['movieId'].astype(int).tolist()

# Fetch movie titles for the top 10 recommendations
top_10_titles = movies_df[movies_df['movieId'].isin(top_10_ids)][['movieId', 'title']]
top_10_titles = top_10_titles.set_index('movieId').loc[top_10_ids]
for title in top_10_titles['title']:
    print(title)