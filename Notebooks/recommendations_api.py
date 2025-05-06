from flask import Flask, jsonify, request
from flask_jwt_extended import JWTManager, jwt_required, create_access_token
from flask_redis import FlaskRedis
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from surprise import SVD, Dataset, Reader
import pandas as pd
import json

app = Flask(__name__)

# -------------------- JWT Configuration --------------------
app.config['JWT_SECRET_KEY'] = 'bookifyy'
jwt = JWTManager(app)

# -------------------- Redis (Caching) --------------------
app.config['REDIS_URL'] = "redis://localhost:6379/0"
redis = FlaskRedis(app)

# -------------------- Rate Limiting --------------------
limiter = Limiter(get_remote_address, app=app)

# -------------------- Load and Prepare Ratings Data --------------------
ratings = pd.read_csv('D:/Database/database/cleaned_datasets/ratings_cleaned.csv')

# Rename columns
ratings.rename(columns={
    'User-ID': 'user_id',
    'ISBN': 'book_id',
    'Book-Rating': 'rating'
}, inplace=True)

# Convert types
ratings['user_id'] = ratings['user_id'].astype(str)
ratings['book_id'] = ratings['book_id'].astype(str)
ratings['rating'] = ratings['rating'].astype(int)

# -------------------- Train SVD Model --------------------
reader = Reader(rating_scale=(0, 10))
data = Dataset.load_from_df(ratings[['user_id', 'book_id', 'rating']], reader)
trainset = data.build_full_trainset()
svd = SVD()
svd.fit(trainset)

books_df = pd.read_csv('D:/Database/database/cleaned_datasets/books_cleaned.csv')
books_df.columns = books_df.columns.str.strip().str.lower().str.replace('-', '_')

# Ensure ISBN is string type
books_df['ISBN'] = books_df['isbn'].astype(str)

def get_book_from_database(book_id):
    book_id = str(book_id)
    book = books_df[books_df['isbn'] == book_id]
    
    if book.empty:
        return None
    
    book_info = book.iloc[0]
    
    return {
        "book_id": book_info['isbn'],
        "book_title": book_info.get('book_title', 'Unknown'),
        "book_author": book_info.get('book_author', 'Unknown'),
        "year_of_publication": book_info.get('year_of_publication', 'Unknown'),
        "publisher": book_info.get('publisher', 'Unknown'),
        "image_url": book_info.get('image_url_s', ''),  # or _m, _l based on what you want
        "rating": None,
        "genres": [],
        "description": "No description available"
    }

def generate_recommendations(user_id):
    user_id = str(user_id)
    top_n = 10

    # Get list of all book_ids
    all_book_ids = ratings['book_id'].unique()

    # Get books the user has already rated
    rated_books = ratings[ratings['user_id'] == user_id]['book_id'].unique()

    # Filter out books already rated
    unseen_books = [book for book in all_book_ids if book not in rated_books]

    # Predict ratings for unseen books
    predictions = [(book_id, svd.predict(user_id, book_id).est) for book_id in unseen_books]

    # Sort by estimated rating
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_predictions = predictions[:top_n]

    # Fetch book details using your function
    recommendations = []
    for book_id, _ in top_predictions:
        book = get_book_from_database(book_id)
        if book:
            recommendations.append(book)

    return recommendations

@app.route('/api/auth/login', methods=['POST'])
def login():
    username = request.json.get('username')
    password = request.json.get('password')
    
    # Add user validation (e.g., check username/password in a database)
    if username == "user@example.com" and password == "password123":
        # Create JWT token
        access_token = create_access_token(identity=username)
        return jsonify({"access_token": access_token}), 200
    
    return jsonify({"error": "Invalid credentials"}), 401

@app.route('/api/recommendations/<user_id>', methods=['GET'])
@limiter.limit("100 per hour")  # Limit to 100 requests per user per hour
def get_recommendations(user_id):
    # Check if recommendations are cached in Redis
    cached_recommendations = redis.get(f"recommendations:{user_id}")
    if cached_recommendations:
        # If cached, return the cached recommendations
        return jsonify(json.loads(cached_recommendations))
    
    # If not cached, generate recommendations
    recommendations = generate_recommendations(user_id)
    
    # Cache the recommendations for 1 hour (3600 seconds)
    redis.setex(f"recommendations:{user_id}", 3600, json.dumps(recommendations))
    
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)
