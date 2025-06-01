# tests/test_app.py

import pytest
import json
# Import app from your src directory
from src.app import app

# Fixture `client` is provided by conftest.py

def test_health_endpoint(client):
    """Test the /health endpoint."""
    response = client.get('/health')
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'healthy'
    # Ensure movie_model_loaded is True as the fixture should have initialized it
    assert data['movie_model_loaded'] is True

def test_recommend_movie_query(client):
    """Test /recommend_movie with a movie-related query, expecting dummy data results."""
    response = client.post(
        '/recommend_movie',
        data=json.dumps({"query": "Recommend a comedy movie."}),
        content_type='application/json'
    )
    assert response.status_code == 200
    data = response.get_json()
    
    # Ensure the response text indicates recommendations are provided
    assert "Here are some movie recommendations for you:" in data['response']
    
    # We expect one of the dummy comedy movies (e.g., 'Yet Another One (2005)' or 'Test Movie A (2000)')
    # to be in the response or context. Check for any of the expected dummy titles.
    recommended_titles_in_response = [rec['title'] for rec in data['context_used']]
    
    # Check if 'Yet Another One (2005)' or 'Test Movie A (2000)' is among the recommended titles
    assert 'Yet Another One (2005)' in recommended_titles_in_response or \
           'Test Movie A (2000)' in recommended_titles_in_response
    
    assert isinstance(data['context_used'], list)
    assert len(data['context_used']) > 0
    # You could add more specific checks for genre presence if desired
    # For example:
    # assert any('Comedy' in rec['genres'] for rec in data['context_used'])


def test_recommend_movie_no_query_provided(client):
    """Test /recommend_movie when no query is in the request body."""
    response = client.post(
        '/recommend_movie',
        data=json.dumps({}), # Empty body or no 'query' key
        content_type='application/json'
    )
    assert response.status_code == 400
    data = response.get_json()
    assert "No query provided" in data['error']