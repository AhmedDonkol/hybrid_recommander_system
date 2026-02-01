import numpy as np 
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder ,MinMaxScaler , StandardScaler
from category_encoders import CountEncoder
from sklearn.compose import ColumnTransformer
import joblib
from scipy.sparse import save_npz
from data_cleaning import clean_data, data_for_content_filtering    

# Cleaned Data Path
CLEANED_DATA_PATH = "data/cleaned_data.csv"

# cols to transform
frequency_enode_cols = ['year']
ohe_cols = ['artist',"time_signature","key"]
tfidf_col = 'tags'
standard_scale_cols = ["duration_ms","loudness","tempo"]
min_max_scale_cols = ["danceability","energy","speechiness","acousticness","instrumentalness","liveness","valence"]

def train_transformer(data):
    """
    Trains a ColumnTransformer on the provided data and saves the transformer to a file.
    The ColumnTransformer applies the following transformations:
    - Frequency Encoding using CountEncoder on specified columns.
    - One-Hot Encoding using OneHotEncoder on specified columns.
    - TF-IDF Vectorization using TfidfVectorizer on a specified column.
    - Standard Scaling using StandardScaler on specified columns.
    - Min-Max Scaling using MinMaxScaler on specified columns.
    Parameters:
    data (pd.DataFrame): The input data to be transformed.
    Returns:
    None
    Saves:
    transformer.joblib: The trained ColumnTransformer object.


    """
    # Initialize ColumnTransformer with different transformations
    preprocessor = ColumnTransformer(
        transformers=[
            ("frequency_encode", CountEncoder(normalize=True,return_df=True), frequency_enode_cols),
            ('ohe', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), ohe_cols),
            ('tfidf', TfidfVectorizer(), tfidf_col),
            ('std_scaler', StandardScaler(), standard_scale_cols),
            ('minmax_scaler', MinMaxScaler(), min_max_scale_cols)
        ],
        remainder='passthrough'
    )

    # Fit the data
    preprocessor.fit(data)
    # save the transformer
    joblib.dump(preprocessor, "transformer.joblib")


def transform_data(data):
    """
    Transforms the provided data using a pre-trained ColumnTransformer.
    Parameters:
    data (pd.DataFrame): The input data to be transformed.
    Returns:
    np.ndarray: The transformed data as a NumPy array.
    """
    # Load the pre-trained transformer
    preprocessor = joblib.load("transformer.joblib")
    # Transform the data
    transformed_data = preprocessor.transform(data)
    return transformed_data

def save_transformed_data(transformed_data, save_path):
    """
    Saves the transformed data to a specified file path in .npz format.
    Parameters:
    transformed_data (np.ndarray): The transformed data to be saved.
    save_path (str): The file path where the transformed data will be saved.
    Returns:
    None

    """
        # save the transformed data
    import numpy as np 
    np.save(save_path, transformed_data)


def calculate_similarity_scores(transformed_data,input_vector):
    """
    Calculates the cosine similarity between the input vector and all rows in the transformed data.
    Parameters:
    transformed_data (np.ndarray): The transformed data.
    input_vector (np.ndarray): The input vector to compare against the transformed data.
    Returns:
    np.ndarray: An array of cosine similarity scores.
    """
    # Calculate cosine similarity
    similarity_scores = cosine_similarity(input_vector, transformed_data)
    return similarity_scores

def content_recommendation(song_name,artist_name,songs_data, transformed_data, k=10):
    """
    Recommends top k songs similar to the given song based on content-based filtering.

    Parameters:
    song_name (str): The name of the song to base the recommendations on.
    artist_name (str): The name of the artist of the song.
    songs_data (DataFrame): The DataFrame containing song information.
    transformed_data (ndarray): The transformed data matrix for similarity calculations.
    k (int, optional): The number of similar songs to recommend. Default is 10.

    Returns:
    DataFrame: A DataFrame containing the top k recommended songs with their names, artists, and Spotify preview URLs.
    """
    # convert song name to lowercase
    song_name = song_name.lower()
    # convert the artist name to lowercase
    artist_name = artist_name.lower()
    # filter out the song from data
    song_row = songs_data.loc[(songs_data["name"] == song_name) & (songs_data["artist"] == artist_name)]
    # get the index of song
    song_index = song_row.index[0]
    # generate the input vector
    input_vector = transformed_data[song_index].reshape(1,-1)
    # calculate similarity scores
    similarity_scores = calculate_similarity_scores(input_vector, transformed_data)
    # get the top k songs
    top_k_songs_indexes = np.argsort(similarity_scores.ravel())[-k-1:][::-1]
    # get the top k songs names
    top_k_songs_names = songs_data.iloc[top_k_songs_indexes]
    # print the top k songs
    top_k_list = top_k_songs_names[['name','artist','spotify_preview_url']].reset_index(drop=True)
    return top_k_list


def main(data_path):
    """
    Test the recommendations for a given song using content-based filtering.

    Parameters:
    data_path (str): The path to the CSV file containing the song data.

    Returns:
    None: Prints the top k recommended songs based on content similarity.
    """
    # load the data
    data = pd.read_csv(data_path)
    # clean the data
    data_content_filtering = data_for_content_filtering(data)
    # train the transformer
    train_transformer(data_content_filtering)
    # transform the data
    transformed_data = transform_data(data_content_filtering)
    #save transformed data
    save_transformed_data(transformed_data,"data/transformed_data.npz")
    
if __name__ == "__main__":
    main(CLEANED_DATA_PATH)