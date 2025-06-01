# src/app.py

from flask import Flask, request, jsonify
from src.movie_recommender import MovieRecommender
import os

app = Flask(__name__)

# Initialize the Movie Recommender model globally
# This initialization will only run if app.py is run directly or not in testing mode.
# When running tests, the `client` fixture in conftest.py will inject its own instance.
# We're checking os.environ.get('FLASK_ENV') which we will set in conftest.py
if os.environ.get('FLASK_ENV') != 'testing':
    try:
        print("Initializing MovieRecommender for production/development environment...")
        app.movie_recommender_instance = MovieRecommender()
    except Exception as e:
        print(f"Error initializing movie recommender model: {e}")
        app.movie_recommender_instance = None # Mark as failed to initialize
else:
    # When in testing environment, the movie_recommender_instance will be set by the pytest fixture
    # We initialize it to None here to make it clear that it's not ready yet.
    app.movie_recommender_instance = None


@app.route('/recommend_movie', methods=['POST'])
def recommend_movie():
    # Access the instance attached to the app object
    if not app.movie_recommender_instance or not app.movie_recommender_instance._is_initialized:
        return jsonify({"error": "Movie recommender not initialized. Please check server logs."}), 500

    data = request.get_json()
    user_query = data.get('query')

    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    try:
        # Call recommend_movies on the instance attached to the app object
        recommendations, _ = app.movie_recommender_instance.recommend_movies(user_query)

        if isinstance(recommendations, str): # Error message from recommender
            response_text = recommendations
            context_used = None
        else:
            response_text = "Here are some movie recommendations for you:\n\n"
            for i, rec in enumerate(recommendations):
                response_text += f"{i+1}. {rec['title']} (Genres: {rec['genres']})\n"
            context_used = recommendations # Pass the list of dicts as context

        return jsonify({
            "response": response_text,
            "context_used": context_used
        })
    except Exception as e:
        print(f"Error during movie recommendation processing: {e}")
        return jsonify({"error": "An internal error occurred while processing your request."}), 500

@app.route('/health', methods=['GET'])
def health_check():
    # Simple health check endpoint for the movie recommender
    # Check if the instance exists and is marked as initialized
    movie_model_loaded = app.movie_recommender_instance is not None and \
                         app.movie_recommender_instance._is_initialized

    if movie_model_loaded:
        return jsonify({"status": "healthy", "movie_model_loaded": True}), 200
    else:
        return jsonify({"status": "unhealthy", "movie_model_loaded": False, "message": "Movie recommender model not loaded"}), 500

if __name__ == '__main__':
    # When running directly, ensure FLASK_ENV is explicitly set for clarity
    os.environ['FLASK_ENV'] = 'development'
    app.run(debug=True, host='0.0.0.0', port=5000)