# tests/test_movie_recommender.py

import pytest
import os
import pandas as pd
# Assuming MovieRecommender is correctly imported via conftest.py's sys.path.insert
# Or, if you prefer explicit import here:
# from src.movie_recommender import MovieRecommender

# Fixtures `temp_data_dir`, `setup_test_data`, `movie_recommender_instance` are provided by conftest.py

def test_movie_recommender_initialization(movie_recommender_instance):
    """Test that the movie recommender initializes correctly with dummy data."""
    assert movie_recommender_instance._is_initialized is True
    assert movie_recommender_instance.embedding_model is not None
    assert movie_recommender_instance.movie_faiss_index is not None
    assert movie_recommender_instance.movies_df is not None
    
    # We have 4 data rows + 1 header row in conftest.py dummy_movies_data.
    # With `header=0` in `pd.read_csv`, movies_df should have 4 rows (excluding the header).
    assert len(movie_recommender_instance.movies_df) == 4 # <--- FIX THIS ASSERTION
    assert 'clean_title' in movie_recommender_instance.movies_df.columns
    assert 'search_text' in movie_recommender_instance.movies_df.columns


def test_movie_recommender_recommend_known_genres(movie_recommender_instance):
    """Test that recommendations are returned for a known genre from dummy data."""
    # A query that should strongly match "Test Movie A (2000),Action|Comedy"
    query = "action comedy film"
    recommendations, _ = movie_recommender_instance.recommend_movies(query, top_k=1)
    
    assert isinstance(recommendations, list)
    assert len(recommendations) > 0
    
    # We expect "Test Movie A" to be the top result for this specific query
    assert recommendations[0]['title'] == "Test Movie A (2000)" # <--- Adjusted assertion for exact match


def test_movie_recommender_cleans_titles(movie_recommender_instance):
    """Test that movie titles are correctly cleaned (year removed)."""
    # Assuming "Test Movie A (2000)" becomes "Test Movie A" (movieId=1)
    cleaned_title = movie_recommender_instance.movies_df.loc[
        movie_recommender_instance.movies_df['movieId'] == 1, 'clean_title'
    ].iloc[0]
    assert cleaned_title == "Test Movie A"
    
    # Test another one: "Another Test Film (1999)" becomes "Another Test Film" (movieId=3)
    cleaned_title_2 = movie_recommender_instance.movies_df.loc[
        movie_recommender_instance.movies_df['movieId'] == 3, 'clean_title'
    ].iloc[0]
    assert cleaned_title_2 == "Another Test Film"