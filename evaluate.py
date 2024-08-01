import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from model import NCFModel


def evaluate_model(model, test_data):
    test_user_ids = test_data['user_id'].values
    test_item_ids = test_data['product_id'].values
    true_ratings = test_data['rating'].values

    predictions = model.predict([test_user_ids, test_item_ids])

    mae = mean_absolute_error(true_ratings, predictions)
    rmse = mean_squared_error(true_ratings, predictions, squared=False)

    return mae, rmse


if __name__ == '__main__':
    # Load your data here
    data = pd.read_csv('preprocessed_data.csv')

    # Split into train and test sets
    train_data = data.sample(frac=0.8, random_state=42)
    test_data = data.drop(train_data.index)

    # Load the model
    model = tf.keras.models.load_model('ncf_model.keras', custom_objects={'NCFModel': NCFModel})

    # Evaluate the model
    mae, rmse = evaluate_model(model, test_data)

    print(f'Mean Absolute Error: {mae:.4f}')
    print(f'Root Mean Squared Error: {rmse:.4f}')
