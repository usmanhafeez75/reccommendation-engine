# Recommendation Engine

This project implements an AI-powered recommendation engine using the Amazon Product Review dataset.

## Setup Instructions

1. **Clone the repository**

```shell
git clone https://github.com/usmanhafeez75/reccommendation-engine.git
cd recommendation_engine
```
   
2. **Install the dependencies**

```shell
pip install -r requirements.txt
```

3. **Download the dataset**

Download the [Amazon Product Review dataset](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) from Kaggle and place the Reviews.csv file in the project directory.

4. **Preprocess the data**


```shell
python data_preprocessing.py
```

5. **Train the model**

```shell
python model.py
```


6. **Run the Flask application**

```shell
python app.py
```

7. **Access the recommendations**

Open your web browser and go to http://127.0.0.1:5000/recommend?user_id=<USER_ID> to get recommendations for a user.

### Default Recommendations
If the user ID does not exist, the application will return default recommendations based on the most popular products.

### Error Handling
The application will handle invalid user IDs and return appropriate error messages.
