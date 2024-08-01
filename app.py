from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds

app = Flask(__name__)

# Load the preprocessed data
data = pd.read_csv('preprocessed_data.csv')

# Create user-item matrix
user_item_matrix = data.pivot(index='user_id', columns='product_id', values='rating').fillna(0)
U, sigma, Vt = svds(user_item_matrix, k=50)
sigma = np.diag(sigma)
predicted_ratings = np.dot(np.dot(U, sigma), Vt)
predicted_ratings_df = pd.DataFrame(predicted_ratings, columns=user_item_matrix.columns)


def get_default_recommendations():
    top_products = data['product_id'].value_counts().head(10).index.tolist()
    return top_products


@app.route('/recommend', methods=['GET'])
def recommend():
    try:
        user_id = int(request.args.get('user_id'))
        if user_id not in predicted_ratings_df.index:
            default_recommendations = get_default_recommendations()
            return jsonify({
                "error": "User ID does not exist. Here are some default recommendations.",
                "recommendations": default_recommendations
            }), 404

        user_ratings = predicted_ratings_df.loc[user_id].sort_values(ascending=False)
        recommendations = user_ratings.head(10).index.tolist()
        return jsonify(recommendations)
    except ValueError:
        return jsonify({"error": "Invalid User ID"}), 400


if __name__ == '__main__':
    app.run(debug=True)
