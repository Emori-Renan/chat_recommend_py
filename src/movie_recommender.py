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
                 anime_data_path='data/anime.csv', 
                 model_name='all-MiniLM-L6-v2',
                 movie_faiss_index_path='models/combined_content_faiss_index.bin.bin',
                 movie_df_path='models/combined_content_df.pkl'):
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
        self.anime_data_path = anime_data_path
        self.movie_faiss_index_path = movie_faiss_index_path
        self.movie_df_path = movie_df_path

        self.embedding_model = None
        self.movie_faiss_index = None
        self.movies_df = None
        self._is_initialized = False # Flag to indicate successful initialization

        self._load_or_build_model()
        
    def _load_single_csv(self, file_path, content_source_type):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found at path: {file_path}")
        
        encondings_to_try = ['utf-8', 'latin-1', 'cp1252']
        for encoding in encondings_to_try:
            try:
                temp_df = pd.read_csv(file_path, sep=',', encoding=encoding)
                print(f"Successfully loaded {content_source_type} data with '{encoding}' encoding.")
                
                if content_source_type == 'movie':
                    temp_df = temp_df.rename(columns={'movieId': 'contentId'})
                    temp_df['type'] = 'movie'
                    temp_df['genres'] = temp_df['genres'].fillna('').astype(str)
                    
                elif content_source_type == 'anime':
                    temp_df = temp_df.rename(columns={'anime_id': 'contentId', 'name': 'title'})
                    temp_df['genres'] = temp_df['genre'].fillna('').astype(str).str.replace(', ', '|')
                    temp_df['type'] = temp_df['type'].fillna('').astype(str).str.lower()
                    
                    temp_df = temp_df.drop(columns='genre')
            
                temp_df['title'] = temp_df['title'].fillna('').astype(str)
                
                return temp_df[['contentId', 'title', 'genres', 'type']]
        
            except UnicodeDecodeError:
                        print(f"Failed to load {content_source_type} with '{encoding}' encoding. Trying next...")
            except pd.errors.ParserError as pe:
                print(f"Parsing error for {content_source_type} with '{encoding}' encoding: {pe}. Data might not be in expected ',' format.")
            except Exception as e:
                print(f"An unexpected error occurred while loading {content_source_type} data with '{encoding}': {e}")
                raise
                
    def _load_or_build_model(self):
        """
        Attempts to load existing combined FAISS index and DataFrame.
        If files are not found, it builds the model from scratch.
        """
        try:
            if (os.path.exists(self.movie_faiss_index_path) and
                os.path.exists(self.movie_df_path)):
                print("Loading existing combined content FAISS index and DataFrame...")
                self.embedding_model = SentenceTransformer(self.model_name)
                self.movie_faiss_index = faiss.read_index(self.movie_faiss_index_path)
                with open(self.movie_df_path, 'rb') as f:
                    self.movies_df = pickle.load(f)
                
                # Re-create the FAISS to DataFrame index map after loading
                self.faiss_to_df_index_map = {idx: df_idx for idx, df_idx in enumerate(self.movies_df.index)}

                print("Combined model loaded successfully.")
                self._is_initialized = True
            else:
                print("Combined model files not found. Building model from scratch...")
                self._build_model()
                self._is_initialized = True
        except Exception as e:
            print(f"Error loading or building combined model: {e}")
            self._is_initialized = False

    def _load_data(self):
        """
        Loads movie and anime data, combines them, and prepares for processing.
        """
        movies_df = self._load_single_csv(self.movies_data_path, 'movie')
        anime_df = self._load_single_csv(self.anime_data_path, 'anime')

        movies_df['unique_id'] = 'movie' + movies_df['contentId'].astype(str)
        anime_df['unique_id'] = 'anime_' + anime_df['contentId'].astype(str)
        
        common_cols = ['unique_id', 'title', 'genres', 'type']
        
        combined_df = pd.concat([movies_df[common_cols], anime_df[common_cols]], ignore_index=True)
        print(f"Successfully loaded and combined {len(movies_df)} movies and {len(anime_df)} anime.")
        combined_df = combined_df.set_index('unique_id')
        
        return combined_df
        
    def _build_model(self):
        """
        Builds the SentenceTransformer embedding model, processes combined content data,
        creates FAISS index, and saves them.
        """
        print("Building combined movie and anime recommendation model...")
        self.movies_df = self._load_data()
        print(f"Loading SentenceTransformer model: {self.model_name}")
        self.embedding_model = SentenceTransformer(self.model_name)
        
        self.movies_df['clean_title'] = self.movies_df['title'].apply(lambda x: re.sub(r' \(\d{4}\)', '', x)).astype(str)
      
        self.movies_df['search_text'] = self.movies_df.apply(
            lambda row: f"{row['type'].capitalize()} title: {row['clean_title']} Genres: {row['genres'].replace('|', ', ')}", axis=1
        )

        movie_embeddings = self.embedding_model.encode(
            self.movies_df['search_text'].tolist(),
            convert_to_tensor=False,
            show_progress_bar=True
        ).astype('float32')
        
        print("Building FAISS index...")
        dimension = movie_embeddings.shape[1]
        self.movie_faiss_index = faiss.IndexFlatL2(dimension)
        self.movie_faiss_index.add(movie_embeddings)
        print(f"FAISS index built with {self.movie_faiss_index.ntotal} vectors.")
        self.faiss_to_df_index_map = {idx: df_idx for idx, df_idx in enumerate(self.movies_df.index)}
        
        print("Saving FAISS index and processed DataFrame...")
        os.makedirs(os.path.dirname(self.movie_faiss_index_path), exist_ok=True)
        faiss.write_index(self.movie_faiss_index, self.movie_faiss_index_path)
        with open(self.movie_df_path, 'wb') as f:
            pickle.dump(self.movies_df, f) # Save the DataFrame with its unique_id index
        print("Model building complete.")

    def recommend_movies(self, query, top_k=5, content_type=None):
        if not self._is_initialized:
            return "Content recommender is not initialized. Please check server logs.", []

        query_embedding = self.embedding_model.encode([query]).astype('float32')

        search_k = top_k * 5
        
        if search_k == 0:
            return "No items in the content database to recommend from. Please ensure data is loaded.", []

        D, I = self.movie_faiss_index.search(query_embedding, top_k)

        raw_recommendations = []
        
        for i in range(len(I[0])): 
            if I[0][i] != -1: # Ensure FAISS returned a valid index (sometimes -1 means no valid result)
                df_unique_id = self.faiss_to_df_index_map[I[0][i]]
                content = self.movies_df.loc[df_unique_id].to_dict()
                raw_recommendations.append(content)
                
        filtered_recommendations = []
        for content_item in raw_recommendations:
            if content_type is None or content_item['type'] == content_type:
                output_item = {k: v for k, v in content_item.items() if k not in ['search_text', 'clean_title']}
                
                output_item['genres'] = output_item.get('genres', 'N/A').replace('|', ', ')
                
                filtered_recommendations.append(output_item)
            
            if len(filtered_recommendations) >= top_k:
                break

        if not filtered_recommendations:
            return "I couldn't find any recommendations for your query. Please try a different request or content type.", []

        return filtered_recommendations, filtered_recommendations
        
        
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