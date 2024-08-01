import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from model import NCFModel

# Load and preprocess your data
data = pd.read_csv('preprocessed_data.csv')

# Convert data types for efficiency
data['rating'] = data['rating'].astype('float32')
data['user_id'] = data['user_id'].astype('int32')
data['product_id'] = data['product_id'].astype('int32')

# Define parameters
num_users = data['user_id'].nunique()
num_items = data['product_id'].nunique()
embedding_size = 16  # Size of the user/item embeddings

# Train-test split
train, test = train_test_split(data, test_size=0.2)

# Build the NCF model
model = NCFModel(num_users, num_items, embedding_size)
model.compile(optimizer='adam', loss='mean_squared_error')

# Prepare training data
train_user_ids = train['user_id'].values
train_item_ids = train['product_id'].values
train_labels = train['rating'].values

# Train the model
model.fit([train_user_ids, train_item_ids], train_labels, epochs=10, batch_size=64)

# Save the trained model
tf.keras.models.save_model(model, "ncf_model.keras")
