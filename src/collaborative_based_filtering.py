import pandas as pd
import dask.dataframe as dd
from scipy.sparse import csr_matrix, save_npz
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List

# ===== Paths =====
TRACK_IDS_PATH = "data/track_ids.npy"
FILTERED_DATA_PATH = "data/collab_filtered_data.csv"
INTERACTION_MATRIX_PATH = "data/interaction_matrix.npz"
SONGS_DATA_PATH = "data/cleaned_data.csv"
USER_HISTORY_PATH = "data/User Listening History.csv"


def filter_songs_data(songs_data: pd.DataFrame, track_ids: List[str], save_df_path: str) -> pd.DataFrame:
    """
    Filter the songs data to keep only tracks present in user listening history.

    Args:
        songs_data (pd.DataFrame): Full songs dataset with 'track_id', 'name', 'artist', etc.
        track_ids (List[str]): List of track IDs to keep.
        save_df_path (str): Path to save the filtered songs CSV.

    Returns:
        pd.DataFrame: Filtered songs data.
    """
    filtered_data = songs_data[songs_data["track_id"].isin(track_ids)].copy()
    filtered_data.sort_values(by="track_id", inplace=True)
    filtered_data.reset_index(drop=True, inplace=True)

    filtered_data.to_csv(save_df_path, index=False)
    return filtered_data


def save_sparse_matrix(matrix: csr_matrix, file_path: str) -> None:
    """
    Save a sparse matrix to disk in .npz format.

    Args:
        matrix (csr_matrix): Sparse matrix to save.
        file_path (str): Path to save the .npz file.
    """
    save_npz(file_path, matrix)


def create_interaction_matrix(
    history_data: dd.DataFrame,
    track_ids_save_path: str,
    save_matrix_path: str
) -> csr_matrix:
    """
    Create a track x user interaction matrix from user listening history.

    Args:
        history_data (dd.DataFrame): User listening history with 'user_id', 'track_id', 'playcount'.
        track_ids_save_path (str): Path to save the track_id mapping.
        save_matrix_path (str): Path to save the sparse interaction matrix.

    Returns:
        csr_matrix: Sparse matrix of shape (num_tracks, num_users) with playcount values.
    """
    df = history_data.copy()
    df['playcount'] = df['playcount'].astype(np.float64)

    # Convert user_id and track_id to categorical codes
    df = df.categorize(columns=['user_id', 'track_id'])
    df['user_idx'] = df['user_id'].cat.codes
    df['track_idx'] = df['track_id'].cat.codes

    # Save track_id mapping
    track_ids = df['track_id'].cat.categories.values
    np.save(track_ids_save_path, track_ids, allow_pickle=True)

    # Group by track x user and sum playcounts
    interaction_df = df.groupby(['track_idx', 'user_idx'])['playcount'].sum().reset_index().compute()

    # Build sparse matrix
    matrix = csr_matrix(
        (interaction_df['playcount'], (interaction_df['track_idx'], interaction_df['user_idx'])),
        shape=(interaction_df['track_idx'].nunique(), interaction_df['user_idx'].nunique())
    )

    # Save matrix
    save_sparse_matrix(matrix, save_matrix_path)
    return matrix


def collaborative_recommendation(
    song_name: str,
    artist_name: str,
    track_ids: np.ndarray,
    songs_data: pd.DataFrame,
    interaction_matrix: csr_matrix,
    k: int = 5
) -> pd.DataFrame:
    """
    Recommend top k similar songs using collaborative filtering.

    Args:
        song_name (str): Name of the input song.
        artist_name (str): Artist of the input song.
        track_ids (np.ndarray): Array of track IDs corresponding to matrix rows.
        songs_data (pd.DataFrame): Filtered songs dataset.
        interaction_matrix (csr_matrix): Sparse user-track interaction matrix.
        k (int, optional): Number of recommendations. Defaults to 5.

    Returns:
        pd.DataFrame: Recommended songs with columns ['name', 'artist'].
    """
    # Lowercase for matching
    song_name = song_name.lower()
    artist_name = artist_name.lower()

    # Find song row
    song_row = songs_data.loc[
        (songs_data['name'].str.lower() == song_name) &
        (songs_data['artist'].str.lower() == artist_name)
    ]
    if song_row.empty:
        raise ValueError("Song not found in dataset!")

    # Get track index
    input_track_id = song_row['track_id'].values.item()
    ind = np.where(track_ids == input_track_id)[0].item()

    # Compute similarity
    input_vector = interaction_matrix[ind]
    similarity_scores = cosine_similarity(input_vector, interaction_matrix).ravel()

    # Get top k recommendations excluding the song itself
    top_indices = np.argsort(similarity_scores)[-k-1:][::-1]
    top_indices = top_indices[top_indices != ind][:k]

    top_track_ids = track_ids[top_indices]
    top_scores = similarity_scores[top_indices]

    scores_df = pd.DataFrame({
        "track_id": top_track_ids,
        "score": top_scores
    })

    top_k_songs = (
        songs_data
        .loc[songs_data["track_id"].isin(top_track_ids)]
        .merge(scores_df, on="track_id")
        .sort_values(by="score", ascending=False)
        .drop(columns=["track_id", "score"])
        .reset_index(drop=True)
    )
    return top_k_songs


def main() -> None:
    """
    Main function to process user history, create interaction matrix,
    and optionally test the recommendation function.
    """
    # Load user listening history
    user_data = dd.read_csv(USER_HISTORY_PATH)

    # Get unique track IDs
    unique_track_ids = user_data['track_id'].unique().compute().tolist()

    # Load songs dataset
    songs_data = pd.read_csv(SONGS_DATA_PATH)

    # Filter songs based on user history
    filter_songs_data(songs_data, unique_track_ids, FILTERED_DATA_PATH)

    # Create interaction matrix
    create_interaction_matrix(user_data, TRACK_IDS_PATH, INTERACTION_MATRIX_PATH)


if __name__ == "__main__":
    main()
