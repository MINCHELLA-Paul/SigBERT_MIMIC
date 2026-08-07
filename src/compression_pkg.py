import numpy as np
import torch


def pca_compression(
    df_data,
    bar_p,
    var_embd="embeddings",
    verbose=False
):
    """
    Compress embedding vectors using Principal Component Analysis (PCA).

    The compression matrix is learned from the training embeddings using
    Singular Value Decomposition (SVD). The returned projection can later
    be applied to new observations using the same centering vector and
    projection matrix.

    Parameters
    ----------
    df_data : pd.DataFrame
        DataFrame containing embedding vectors.
    bar_p : int
        Target embedding dimension after compression.
    var_embd : str, default="embeddings"
        Name of the column containing embeddings.
    verbose : bool, default=False
        Whether to print PCA diagnostics.

    Returns
    -------
    V_proj : np.ndarray
        Compressed embeddings of shape (N, bar_p).
    R_opt : np.ndarray
        PCA projection matrix of shape (bar_p, p).
    mean_embedding : np.ndarray
        Mean embedding vector used for centering.
    explained_ratio : float
        Fraction of variance explained by the first `bar_p`
        principal components.
    """
    # Extract embeddings
    V = np.stack(df_data[var_embd].values)
    N, p = V.shape

    if bar_p > p:
        raise ValueError(f"bar_p={bar_p} exceeds embedding dimension p={p}.")

    
    # Center data
    mean_embedding = V.mean(axis=0)
    V_centered = V - mean_embedding

    
    # Singular Value Decomposition
    U, S, Vh = np.linalg.svd(V_centered,full_matrices=False)

    # First bar_p principal directions
    R_opt = Vh[:bar_p]

    
    # Projection
    V_proj = V_centered @ R_opt.T

    
    # Explained variance
    total_variance = np.sum(S**2)
    explained_variance = np.sum(S[:bar_p]**2)
    explained_ratio = explained_variance / total_variance
    
    # Diagnostics

    if verbose:
        print(f"Original dimension: {p}")
        print(f"Compressed dimension: {bar_p}")
        print(f"Explained variance ratio: {100 * explained_ratio:.2f}%")

    return V_proj, R_opt, mean_embedding, explained_ratio



def apply_linear_projection(
    df_input,
    projection_matrix,
    mean_embedding,
    var_embd="embeddings"
):
    """
    Apply a previously learned PCA projection to a set of embeddings.

    The embeddings are first centered using the mean embedding vector
    estimated on the training set, then projected onto the PCA subspace.

    Parameters
    ----------
    df_input : pd.DataFrame
        DataFrame containing embedding vectors.
    projection_matrix : np.ndarray
        PCA projection matrix of shape (bar_p, p).
    mean_embedding : np.ndarray
        Mean embedding vector computed on the training set.
    var_embd : str, default="embeddings"
        Name of the embedding column.

    Returns
    -------
    df_projected : pd.DataFrame
        Copy of the input dataframe with compressed embeddings.
    """

    df_projected = df_input.copy()

    Z_original = np.vstack(df_projected[var_embd].values)

    Z_centered = Z_original - mean_embedding

    Z_compressed = Z_centered @ projection_matrix.T

    df_projected[var_embd] = [row.tolist() for row in Z_compressed]

    return df_projected