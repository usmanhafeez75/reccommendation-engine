import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds

# Load preprocessed data
data = pd.read_csv('preprocessed_data.csv')

# Create user-item matrix
user_item_matrix = data.pivot(index='user_id', columns='product_id', values='rating').fillna(0)

# Perform matrix factorization
U, sigma, Vt = svds(user_item_matrix, k=50)
sigma = np.diag(sigma)
predicted_ratings = np.dot(np.dot(U, sigma), Vt)

# Save predicted ratings
predicted_ratings_df = pd.DataFrame(predicted_ratings, columns=user_item_matrix.columns)
predicted_ratings_df.to_csv('predicted_ratings.csv', index=False)
