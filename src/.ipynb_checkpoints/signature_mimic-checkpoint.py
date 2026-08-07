import numpy as np
import pandas as pd

# Signature Package 
import iisignature

from sklearn.preprocessing import StandardScaler


def preprocess_time(
    df,
    time_var = 'date_creation',  # Temporal variable (e.g., report date)
    patient_id_col = 'ID'        # Patient identifier column
):
    """
    Normalize the temporal variable to the [0, 1] interval for each patient.

    Each patient may have multiple reports, each associated with a timestamp.
    This function applies a per-patient min-max normalization to the specified
    temporal variable, mapping it to the [0, 1] range to represent relative time.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing patient report data.
    time_var : str, default='date_creation'
        Column name corresponding to the temporal variable (must be datetime).
    patient_id_col : str, default='ID'
        Column name identifying each patient.

    Returns
    -------
    pd.DataFrame
        DataFrame with two new columns:
        - 'timestamp_OG': original timestamp in Unix format.
        - 'timestamp': normalized timestamp in [0, 1] for each patient.
    """
    # Convert datetime column to Unix timestamp
    df.loc[:, 'timestamp_OG'] = df.loc[:, time_var].apply(lambda x: x.timestamp())

    # Container for transformed patient groups
    groups_transformed = []

    # Process each patient's group
    for patient_id, group in df.groupby(patient_id_col):
        group = group.sort_values(by='timestamp_OG')
        min_time = group['timestamp_OG'].min()
        max_time = group['timestamp_OG'].max()

        if min_time != max_time:
            group['timestamp'] = (group['timestamp_OG'] - min_time) / (max_time - min_time)
        else:
            group['timestamp'] = 0.0  # Assign zero if all timestamps are identical

        groups_transformed.append(group)

    # Concatenate all processed groups
    df_time_transfo = pd.concat(groups_transformed, ignore_index=True)

    print("Timestamps normalized for each patient to the [0, 1] range.")
    return df_time_transfo




def calculate_signature(
    path,
    order= 2,
    use_Levy = False,
    use_log = False,
    apply_lead_lag=False
):
    """
    Compute the signature of a path with optional variants.

    Supports raw, log-signature, Lévy area, and lead-lag transformations.

    Parameters
    ----------
    path : np.ndarray
        Input path as a 2D array of shape (T, d).
    order : int, default=2
        Signature order to compute.

    Returns
    -------
    np.ndarray
        Signature features extracted from the path.
    """

    prep_log = iisignature.prepare(path.shape[1], order)
    return iisignature.logsig(path, prep_log) if use_log else iisignature.sig(path, order)



def preprocess_sign(
    df,
    retire_small= False,
    patient_id_col= 'ID',
    return_id = False,
    verbose = True
):
    """
    Preprocess a DataFrame containing signature features.

    This function removes patients with missing signature values and optionally
    sets near-zero values to zero to improve numerical stability.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with signature features and patient identifiers.
    retire_small : bool, default=False
        If True, values with absolute magnitude less than 1e-15 are replaced with 0.
    patient_id_col : str, default='ID'
        Name of the column containing patient identifiers.
    return_id : bool, default=False
        If True, return the list of patient IDs after filtering.
    verbose : bool, default=True
        If True, print the percentage of rows containing missing signature values.

    Returns
    -------
    pd.DataFrame or (pd.DataFrame, np.ndarray)
        Cleaned DataFrame. If `return_id` is True, also returns the array of retained patient IDs.
    """
    # Identify patients with NaNs in any signature column
    ids_with_nan = df[df.filter(like='sig_').isna().any(axis=1)][patient_id_col].unique()

    if verbose:
        percent_with_nan = 100 * len(ids_with_nan) / len(df)
        print(f"Percentage of rows with NaNs in signature columns: {percent_with_nan:.2f}%")

    # Remove patients with NaNs in their signature features
    df = df[~df[patient_id_col].isin(ids_with_nan)]

    # Set very small values to zero if requested
    if retire_small:
        df.update(df.filter(like='sig_').applymap(lambda x: 0 if np.abs(x) < 1e-15 else x))

    id_list = df[patient_id_col].unique()

    return (df, id_list) if return_id else df




def signature_extract(
    df_OG,
    order=2,
    var_temp="timestamp",
    var_patient="ID",
    var_embd="embeddings",
    var_struct_list=None,
    interpolation_type='linear',
    use_log=False,
    verbose=True
):
    """
    Compute path-signature coefficients from longitudinal patient data.

    Automatically reinjects survival columns ('DEATH_L', 'R')
    if they exist in the input dataframe.
    """


    df = df_OG.copy()

    # --------------------------------------------------
    # Detect survival columns if present
    # --------------------------------------------------
    survival_cols = []
    if "DEATH" in df.columns:
        survival_cols.append("DEATH")
    if "DEATH_L" in df.columns:
        survival_cols.append("DEATH_L")

    if "T_days" in df.columns:
        survival_cols.append("T_days")
    if "R" in df.columns:
        survival_cols.append("R")

    # --------------------------------------------------
    # Expand embeddings if provided
    # --------------------------------------------------
    embedding_dim = 0
    emb_cols = []

    if var_embd is not None:
        embedding_dim = len(df[var_embd].iloc[0])
        emb_cols = [f"emb_{i}" for i in range(embedding_dim)]

        emb_array = np.vstack(df[var_embd].values)
        emb_df = pd.DataFrame(emb_array, columns=emb_cols, index=df.index)
        df = pd.concat([df, emb_df], axis=1)

    # --------------------------------------------------
    # Structured variables handling
    # --------------------------------------------------
    struct_cols = []

    if var_struct_list is not None:
        missing = [c for c in var_struct_list if c not in df.columns]
        if missing:
            raise ValueError(f"Missing structured columns: {missing}")

        df[var_struct_list] = df[var_struct_list].apply(
            pd.to_numeric, errors="coerce"
        )

        if interpolation_type == "linear":
            df = df.sort_values([var_patient, var_temp])
        
            def interp_one_patient(g: pd.DataFrame) -> pd.DataFrame:
                # Interpolate using the temporal index to avoid issues with irregular timestamps
                g = g.sort_values(var_temp).set_index(var_temp)
        
                # Interpolation + boundary extrapolation
                # - limit_direction="both": fills forward and backward (including boundaries)
                # - limit_area="outside": explicitly targets NaNs outside the internal data range
                out = g[var_struct_list].interpolate(
                    method="linear",
                    limit_direction="both",
                    limit_area="outside"
                )
        
                # Safety check: if a column is entirely NaN for this patient,
                # it will remain NaN after interpolation → replace with 0 (or another strategy)
                out = out.fillna(0.0)
        
                g[var_struct_list] = out
                return g.reset_index()
        
            df = (
                df.groupby(var_patient, group_keys=False)
                  .apply(interp_one_patient)
            )
        elif interpolation_type == "zeros":
            df[var_struct_list] = df[var_struct_list].fillna(0.0)
        elif interpolation_type is not None:
            raise ValueError(f"Unknown interpolation_type: {interpolation_type}")

        scaler = StandardScaler()
        df[var_struct_list] = scaler.fit_transform(df[var_struct_list])

        struct_cols = var_struct_list

    # --------------------------------------------------
    # Path dimensionality
    # --------------------------------------------------
    n_components = 1 + embedding_dim + len(struct_cols)

    if use_log:
        nbr_sig = iisignature.logsiglength(n_components, order)
    else:
        nbr_sig = iisignature.siglength(n_components, order)

    nbr_sig_order2 = iisignature.siglength(n_components, 2)

    if verbose:
        print(f"Path dimension: {n_components}")
        print(f"Signature components (order {order}): {nbr_sig}")

    # --------------------------------------------------
    # Signature computation
    # --------------------------------------------------
    results = []

    for pid, group in df.groupby(var_patient):

        group = group.sort_values(var_temp)

        path_cols = [var_temp] + emb_cols + struct_cols
        path = group[path_cols].values.astype(np.float64)

        if path.shape[1] > 256:
            raise ValueError("Path dimensionality exceeds 256.")

        # Add zero start point
        path = np.vstack([np.zeros((1, path.shape[1])), path])

        signature = calculate_signature(
            path,
            order=order,
            use_log=use_log,
        )

        row = {f"sig_{i+1}": val for i, val in enumerate(signature)}
        row[var_patient] = pid

        for col in survival_cols:
            row[col] = group[col].iloc[-1]

        results.append(row)

    signature_df = pd.DataFrame(results)

    # Ensure ID first
    ordered_cols = [var_patient] + [c for c in signature_df.columns if c != var_patient]
    signature_df = signature_df[ordered_cols]

    return signature_df, nbr_sig