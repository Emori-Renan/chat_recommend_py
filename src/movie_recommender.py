import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import re # For cleaning movie titles (e.g., removing year from title)
import pickle # To save/load pandas DataFrame

class MovieRecommender:
    """
    A movie recommender model that uses semantic search based on SentenceTransformer
    embeddings and FAISS for efficient similarity search.
    It loads movie data, processes it, builds a FAISS index, and provides
    movie recommendations based on user queries.
    """
    def __init__(self, movies_data_path='data/movies.csv',
                 model_name='sentence-transformers/all-MiniLM-L6-v2',
                 movie_embeddings_path='models/movie_embeddings.pkl', # This path is not strictly used for loading, but good to have
                 movie_faiss_index_path='models/movie_faiss_index.bin',
                 movie_df_path='models/movie_df.pkl'):
        """
        Initializes the MovieRecommender.

        Args:
            movies_data_path (str): Path to the movie dataset (e.g., movies.csv).
            model_name (str): Name of the SentenceTransformer model to use for embeddings.
            movie_embeddings_path (str): Path to save/load movie embeddings (for future use, not directly loaded by FAISS).
            movie_faiss_index_path (str): Path to save/load the FAISS index for movies.
            movie_df_path (str): Path to save/load the processed movie DataFrame.
        """
        self.movies_data_path = movies_data_path
        self.model_name = model_name
        self.movie_embeddings_path = movie_embeddings_path
        self.movie_faiss_index_path = movie_faiss_index_path
        self.movie_df_path = movie_df_path

        self.embedding_model = None
        self.movie_faiss_index = None
        self.movies_df = None
        self.movie_ids_map = None # To map FAISS index to DataFrame index (original pandas index)
        self._is_initialized = False # Flag to indicate successful initialization

        # Ensure models directory exists
        os.makedirs(os.path.dirname(movie_faiss_index_path), exist_ok=True)

        self._load_or_build_model()

    def _load_or_build_model(self):
        """
        Attempts to load pre-built movie FAISS index and DataFrame.
        If files are not found, it builds them from scratch.
        """
        try:
            if os.path.exists(self.movie_faiss_index_path) and os.path.exists(self.movie_df_path):
                print(f"Loading existing movie FAISS index from {self.movie_faiss_index_path} and DataFrame from {self.movie_df_path}...")
                self.embedding_model = SentenceTransformer(self.model_name) # Ensure model is loaded even if index exists
                self.movie_faiss_index = faiss.read_index(self.movie_faiss_index_path)
                self.movies_df = pd.read_pickle(self.movie_df_path)
                self.movie_ids_map = self.movies_df.index.values # Re-establish the mapping
                print("Movie models loaded successfully.")
                self._is_initialized = True
            else:
                print("Building movie FAISS index and DataFrame from scratch...")
                self._build_model()
        except Exception as e:
            print(f"Error loading or building movie models: {e}")
            self._is_initialized = False # Mark as failed to initialize

    # Inside your MovieRecommender class, find the _build_model method

    def _build_model(self):
        """
        Builds the FAISS index and preprocesses movie data if not already present.
        """
        self.movies_df = self._load_data() # This loads your movies.csv into a DataFrame

        # --- RE-VERIFY THESE LINES CAREFULLY ---
        # 1. Clean up titles: remove year for better embedding matching with queries
        # Ensure 'title' column itself is string-like before applying regex
        self.movies_df['clean_title'] = self.movies_df['title'].astype(str).apply(lambda x: re.sub(r' \(\d{4}\)', '', x))

        # 2. Combine relevant columns for embedding: title and genres
        #    Ensure all components are strings before combining.
        #    Explicitly convert to string and strip whitespace.
        self.movies_df['search_text'] = self.movies_df.apply(
            lambda row: (
                f"Movie: {str(row['clean_title']).strip()}. "
                f"Genres: {str(row['genres']).replace('|', ', ').strip()}."
            ), axis=1
        )

        # --- Debugging print: Check types and content before encoding ---
        print("\n--- Debugging: Sample of 'search_text' column and its types ---")
        print(self.movies_df[['title', 'genres', 'clean_title', 'search_text']].head())
        print("Data types of 'search_text' column:", self.movies_df['search_text'].dtype)
        print("Number of non-string values:", self.movies_df['search_text'].apply(lambda x: not isinstance(x, str)).sum())
        print("Number of NaN values:", self.movies_df['search_text'].isna().sum())
        print("--- End Debugging ---")
        # --- END RE-VERIFY ---


        self.embedding_model = SentenceTransformer(self.model_name)

        print("Generating movie embeddings...")
        # Add .fillna('') as a final safeguard just before encoding
        embeddings = self.embedding_model.encode(
            self.movies_df['search_text'].fillna('').tolist(),
            convert_to_numpy=True
        ).astype('float32') # FAISS requires float32

        d = embeddings.shape[1]
        self.movie_faiss_index = faiss.IndexFlatL2(d)
        self.movie_faiss_index.add(embeddings)

        self.movie_ids_map = self.movies_df.index.values

        faiss.write_index(self.movie_faiss_index, self.movie_faiss_index_path)
        self.movies_df.to_pickle(self.movie_df_path)
        print("Movie models built and saved successfully.")
        self._is_initialized = True

    # Inside your MovieRecommender class

    def _load_data(self):
        """
        Loads movie data from the specified CSV file.
        Assumes movies.csv is comma-separated with a header (movieId,title,genres).
        Handles potential UnicodeDecodeError by trying common encodings.
        """
        if not os.path.exists(self.movies_data_path):
            raise FileNotFoundError(f"Movie data file not found at: {self.movies_data_path}")

        # Try common encodings
        encodings_to_try = ['utf-8', 'latin-1', 'cp1252']

        # --- IMPORTANT FIXES IN read_csv PARAMETERS ---
        # `sep=','` because your file is comma-separated
        # `header=0` (or just remove it) because your file HAS a header on the first row
        # `names` is NOT needed if header is present and names match, but good to keep if you want to ensure order
        # `engine='python'` is NOT needed for a single character separator like ','
        # --- END IMPORTANT FIXES ---

        for encoding in encodings_to_try:
            try:
                movies_df = pd.read_csv(
                    self.movies_data_path,
                    sep=',',  # <--- Change separator to comma
                    header=0,  # <--- Specify that the first row (index 0) is the header
                    # names=['movieId', 'title', 'genres'], # Optional: can remove if header names are exact
                    encoding=encoding
                    # Remove engine='python' as it's not needed for comma separator
                )
                print(f"Successfully loaded movies data with '{encoding}' encoding.")

                # --- Debugging: Check the first few rows and types right after loading ---
                print("\n--- Debugging: DataFrame head after initial load ---")
                print(movies_df.head())
                print("DataFrame columns and dtypes:")
                print(movies_df.info())
                print("--- End Debugging ---")

                return movies_df
            except UnicodeDecodeError:
                print(f"Failed to load with '{encoding}' encoding. Trying next...")
            except pd.errors.ParserError as pe:
                print(f"Parsing error with '{encoding}' encoding: {pe}. Data might not be in expected ',' format.")
            except Exception as e:
                print(f"An unexpected error occurred while loading movies data with '{encoding}': {e}")
                raise

        raise ValueError("Could not load movies data with any tried encoding (utf-8, latin-1, cp1252) or parsing failed.")

# ... (rest of the class definition) ...

    def recommend_movies(self, query, top_k=5):
        if not self._is_initialized:
            return "Movie recommender is not initialized. Please check server logs.", []

        query_embedding = self.embedding_model.encode([query]).astype('float32')

        # Search FAISS index for top_k most similar movie embeddings
        D, I = self.movie_faiss_index.search(query_embedding, top_k)

        recommendations = []
        for i in range(top_k):
            # FAISS returns -1 for indices that don't exist (e.g., if top_k > total items)
            if I[0][i] != -1:
                original_df_index = self.movie_ids_map[I[0][i]]
                movie = self.movies_df.loc[original_df_index].to_dict()

                # Exclude the 'search_text' and 'clean_title' from the final output dict
                movie_output = {k: v for k, v in movie.items() if k not in ['search_text', 'clean_title']}

                # --- IMPORTANT FIX HERE ---
                # Ensure 'genres' is a string before calling replace.
                # If it's NaN or None, convert it to an empty string or 'N/A'
                raw_genres = movie_output.get('genres', 'N/A') # Use .get() to handle potential missing key, default to 'N/A'
                movie_output['genres'] = str(raw_genres).replace('|', ', ') # Convert to string then replace
                # --- END IMPORTANT FIX ---

                recommendations.append(movie_output)

        if not recommendations:
            return "I couldn't find any movie recommendations for your query. Please try a different request.", []

        return recommendations, recommendations
# Example usage (for testing this module directly)
if __name__ == '__main__':
    # Create dummy data/models directories for local testing
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # Create a dummy movies.csv if it doesn't exist for direct testing
    dummy_movies_data_path = 'data/movies.csv'
    if not os.path.exists(dummy_movies_data_path):
        dummy_movies_data = [
            "1::Toy Story (1995)::Animation|Children|Comedy",
            "2::Jumanji (1995)::Adventure|Children|Fantasy",
            "3::Grumpier Old Men (1995)::Comedy|Romance",
            "4::Waiting to Exhale (1995)::Comedy|Drama",
            "5::Father of the Bride Part II (1995)::Comedy",
            "6::Heat (1995)::Action|Crime|Thriller",
            "7::Sabrina (1995)::Comedy|Romance",
            "8::Tom and Huck (1995)::Adventure|Children",
            "9::Sudden Death (1995)::Action",
            "10::GoldenEye (1995)::Action|Adventure|Thriller"
        ]
        with open(dummy_movies_data_path, 'w', encoding='utf-8') as f:
            for line in dummy_movies_data:
                f.write(line + '\n')
        print("Dummy movies.csv created for local testing.")

    print("\n--- Movie Recommender Test ---")
    recommender = MovieRecommender()

    if recommender._is_initialized:
        while True:
            user_query = input("You (Movie Query, type 'exit' to quit): ")
            if user_query.lower() == 'exit':
                break
            
            recs, _ = recommender.recommend_movies(user_query)
            if isinstance(recs, str): # Error message
                print(f"Recommender: {recs}")
            else:
                print("Recommender found:")
                for rec in recs:
                    print(f"- {rec['title']} (Genres: {rec['genres']})")
    else:
        print("Movie recommender failed to initialize. Please check console output for errors.")