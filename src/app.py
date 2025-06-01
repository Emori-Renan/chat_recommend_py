# src/app.py

from flask import Flask, request, jsonify
from movie_recommender import MovieRecommender
import os

app = Flask(__name__)

if os.environ.get('FLASK_ENV') != 'testing':
    try:
        print("Initializing combined Movie/Anime Recommender for production/development environment...")
        # Make sure the paths here reflect your new combined models
        app.movie_recommender_instance = MovieRecommender(
            movie_faiss_index_path='models/combined_content_faiss_index.bin',
            movie_df_path='models/combined_content_df.pkl'
        )
    except Exception as e:
        print(f"Error initializing combined recommender model: {e}")
        app.movie_recommender_instance = None
else:
    app.movie_recommender_instance = None


@app.route('/recommend', methods=['POST'])
def recommend_content():
    if not app.movie_recommender_instance or not app.movie_recommender_instance._is_initialized:
        return jsonify({"error": "Content recommender not initialized. Please check server logs."}), 500

    data = request.get_json()
    user_query = data.get('query')
    top_k = data.get('top_k', 5)
    content_type = data.get('type')

    if not user_query:
        return jsonify({"error": "No query provided"}), 400
    
    if content_type is not None and content_type not in ['movie', 'anime', 'tv', 'ova', 'special', 'ona', 'music']:
        return jsonify({"error": "Invalid content type. Must be 'movie', 'anime' or omitted."}), 400


    try:
        recommendations, _ = app.movie_recommender_instance.recommend_movies(user_query, top_k=top_k, content_type=content_type)

        if isinstance(recommendations, str):
            response_text = recommendations
            context_used = None
        else:
            response_text = "Here are some recommendations for you:\n\n"
            for i, rec in enumerate(recommendations):
                response_text += f"{i+1}. {rec['title']} (Type: {rec['type'].capitalize()}, Genres: {rec['genres']})\n"
            context_used = recommendations

        return jsonify({
            "response": response_text,
            "context_used": context_used
        })
    except Exception as e:
        print(f"Error during recommendation processing: {e}")
        return jsonify({"error": "An internal error occurred while processing your request."}), 500
    
@app.route('/health', methods=['GET'])
def health_check():
    movie_model_loaded = app.movie_recommender_instance is not None and \
                         app.movie_recommender_instance._is_initialized
    if movie_model_loaded:
        return jsonify({"status": "healthy", "combined_content_model_loaded": True}), 200
    else:
        return jsonify({"status": "unhealthy", "combined_content_model_loaded": False, "message": "Combined content recommender model not loaded"}), 500

if __name__ == '__main__':
    os.environ['FLASK_ENV'] = 'development'
    app.run(debug=True, host='0.0.0.0', port=5000)