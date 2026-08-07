import numpy as np
import pandas as pd
from tqdm import tqdm

def split_dataframe_by_patient(
    df,
    patient_id_col,
    n_splits,
    output_prefix,
    n_rows=None,
    verbose=True
):
    """
    Split a large dataframe into multiple CSV files by grouping on patient identifiers.

    The function ensures that all observations from a given patient remain
    in the same split. It also provides descriptive statistics about the
    number of observations per patient.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing longitudinal patient data.
    patient_id_col : str
        Column name identifying each patient (e.g., SUBJECT_ID).
    n_splits : int
        Number of output splits (i.e., number of CSV files to generate).
    output_prefix : str
        Prefix for output CSV files. Files will be named:
        '{output_prefix}_part_1.csv', ..., '{output_prefix}_part_n.csv'.
    n_rows : int or None, default=None
        If specified, only the first `n_rows` rows of the dataframe are processed.
        If None, the full dataframe is used.
    verbose : bool, default=True
        If True, prints descriptive statistics and progress information.

    Returns
    -------
    None
        The function writes CSV files to disk and does not return anything.
    """

    # ----------------------------------------
    # Optional row restriction
    # ----------------------------------------
    if n_rows is not None:
        df = df.iloc[:n_rows].copy()

    # ----------------------------------------
    # Group by patient
    # ----------------------------------------
    patient_groups = df.groupby(patient_id_col)
    patient_ids = list(patient_groups.groups.keys())

    n_patients = len(patient_ids)

    # ----------------------------------------
    # Compute descriptive statistics
    # ----------------------------------------
    obs_per_patient = patient_groups.size().values

    mean_obs = np.mean(obs_per_patient)
    std_obs = np.std(obs_per_patient)

    if verbose:
        print(f"Number of patients: {n_patients}")
        print(f"Mean number of observations per patient: {mean_obs:.2f}")
        print(f"Standard deviation: {std_obs:.2f}")

    # ----------------------------------------
    # Shuffle patients for better balance
    # ----------------------------------------
    rng = np.random.default_rng(seed=42)
    shuffled_ids = rng.permutation(patient_ids)

    # ----------------------------------------
    # Split patient IDs
    # ----------------------------------------
    split_ids = np.array_split(shuffled_ids, n_splits)

    # ----------------------------------------
    # Create and export splits
    # ----------------------------------------
    for i, ids in enumerate(tqdm(split_ids)):

        df_split = df[df[patient_id_col].isin(ids)]

        output_path = f"{output_prefix}_part_{i+1}.csv"
        df_split.to_csv(output_path, index=False)

        if verbose:
            print(
                f"Saved split {i+1}/{n_splits} "
                f"with {len(ids)} patients and {len(df_split)} rows "
                f"→ {output_path}"
            )