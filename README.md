This is a movie recommendation system. Below is more information on the algorithm and data.

Formulation:

  Data Mining Task: Clustering

  Input: MovieID, Title, Genre & UserID, MovieID, Rating

  Output: List of top 10 recommendations from best to worst

Data Sets:

  Site: MovieLens

  Data stats: Over 200k users and over 80k movies

  Preprocessing: For efficiency purposes only recommend the top 500 rated movies and create matrix from data with each row representing users and each column representing movies.

Algorithm:

  Applied a K-means algorithm
    Takes data as a users-movies matrix
    Puts users into clusters based on similar movie ratings
    Implemented using Scikit-learn’s K-means function
    
  Calculates movies with most amount of “good” ratings (4.0 or higher)
  
  If tie or no movies with 4.0 or higher then calculates highest average ratings
  
  Returns 10 best recommendations from best to worst and screen displaying PCA Graph based on clusters of movies
