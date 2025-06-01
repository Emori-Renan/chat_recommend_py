# tests/conftest.py

import pytest
import os
import sys
import pandas as pd
# Ensure your path adjustments are correct for importing app and MovieRecommender
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from src.movie_recommender import MovieRecommender
from src.app import app # Import app *after* sys.path.insert if it's the module being tested

@pytest.fixture(scope="session")
def temp_data_dir(tmp_path_factory):
    tmpdir = tmp_path_factory.mktemp("test_data_models")
    os.makedirs(os.path.join(tmpdir, "models"), exist_ok=True)
    return tmpdir

@pytest.fixture(scope="session")
def setup_test_data(temp_data_dir):
    movies_csv_path = temp_data_dir / "movies.csv"
    dummy_movies_data = [
        "movieId,title,genres",
        "1,Test Movie A (2000),Action|Comedy",
        "2,Test Movie B (2001),Drama|Romance",
        "3,Another Test Film (1999),Sci-Fi|Thriller",
        "4,Yet Another One (2005),Comedy"
    ]
    with open(movies_csv_path, 'w', encoding='utf-8') as f:
        for line in dummy_movies_data:
            f.write(line + '\n')
    return {
        "movies_csv_path": str(movies_csv_path),
        "models_dir": str(temp_data_dir / "models")
    }

@pytest.fixture(scope="session")
def movie_recommender_instance(setup_test_data):
    recommender = MovieRecommender(
        movies_data_path=setup_test_data["movies_csv_path"],
        movie_faiss_index_path=os.path.join(setup_test_data["models_dir"], 'movie_faiss_index.bin'),
        movie_df_path=os.path.join(setup_test_data["models_dir"], 'movie_df.pkl')
    )
    # Ensure it's initialized, which happens in its __init__
    return recommender

@pytest.fixture
def client(movie_recommender_instance):
    app.config['TESTING'] = True
    # --- ADD THIS LINE ---
    os.environ['FLASK_ENV'] = 'testing' # Set environment variable for app.py to recognize testing mode
    # --- END ADD ---

    app.movie_recommender_instance = movie_recommender_instance
    with app.test_client() as client:
        yield client
    
    # --- ADD THIS LINE FOR CLEANUP ---
    del os.environ['FLASK_ENV'] # Clean up env var after test session
    # --- END ADD ---
    del app.movie_recommender_instance