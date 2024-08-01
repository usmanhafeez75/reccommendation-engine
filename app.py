from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import tensorflow as tf
from model import NCFModel

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('ncf_model.keras', custom_objects={'NCFModel': NCFModel})

# Load preprocessed data to map user_id and product_id back
data = pd.read_csv('preprocessed_data.csv')


# Get default recommendations based on the most popular products
def get_default_recommendations(n=10):
    return data['product_id'].value_counts().head(n).index.tolist()


def get_recommendations(user_id):
    user_ids = np.array([user_id] * data['product_id'].nunique())
    item_ids = np.array(data['product_id'].unique())
    predictions = model.predict([user_ids, item_ids])
    top_indices = predictions.flatten().argsort()[-10:][::-1]  # Get top 10 recommendations
    recommended_product_ids = item_ids[top_indices]
    return recommended_product_ids.tolist()


@app.route('/recommend', methods=['GET'])
def recommend():
    try:
        user_id = int(request.args.get('user_id'))

        # Check if the user_id exists in the dataset
        if user_id not in data['user_id'].unique():
            recommendations = get_default_recommendations()  # Return default recommendations
        else:
            recommendations = get_recommendations(user_id)

        return jsonify(recommendations)

    except ValueError:
        return jsonify({"error": "Invalid User ID"}), 400


if __name__ == '__main__':
    app.run(debug=True)
