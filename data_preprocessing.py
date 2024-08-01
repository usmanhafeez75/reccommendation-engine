import pandas as pd

# Load the dataset
data = pd.read_csv('Reviews.csv')

# Basic preprocessing
data = data[['UserId', 'ProductId', 'Score']]
data.dropna(inplace=True)

# Rename columns for consistency
data.columns = ['user_id', 'product_id', 'rating']

# Aggregate duplicate entries by taking the mean rating for the same user-product pair
data = data.groupby(['user_id', 'product_id']).agg({'rating': 'mean'}).reset_index()

# Convert categorical data to numeric
data['user_id'] = data['user_id'].astype('category').cat.codes
data['product_id'] = data['product_id'].astype('category').cat.codes

# Save the preprocessed data
data.to_csv('preprocessed_data.csv', index=False)
