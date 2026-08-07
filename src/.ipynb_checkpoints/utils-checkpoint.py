########################################################################################
#                                                                                      #
#                                                                                      #
#                                                                                      #
#                                     PACKAGES IMPORT                                  #
#                                                                                      #
#                                                                                      #
#                                                                                      #
########################################################################################


import pandas as pd
from typing import Optional
import numpy as np
# %matplotlib inline
import time


# Package COMPRESSION ###################
from compression_pkg import apply_linear_projection

# Package SIGNATURES ###################
from signature_mimic import preprocess_time, signature_extract, preprocess_sign

# Package SURVIVAL ANALYSIS ###################
from survival_analysis_sigbert import preprocess_cox, feat_event_extract, global_cox_train, skglm_datatest
from sksurv.metrics import concordance_index_censored

# Package SKLEARN
import warnings
# set_config(display="text")  # displays text representation of estimators
warnings.filterwarnings("ignore", 
                        message="invalid value encountered in divide", 
                        category=RuntimeWarning)


from sklearn.model_selection import train_test_split





import matplotlib.pyplot as plt
import seaborn as sns

def print_dataset_statistics(df, var_id, var_death):
    """
    Computes and prints basic descriptive statistics about the survival dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing at least the columns:
        - 'ID' : patient identifier
        - 'DEATH' : 1 if the patient is deceased, 0 if censored

    Prints
    ------
    - Total number of patients
    - Total number of medical reports
    - Average number of reports per patient
    - Number of deceased patients
    - Number of censored patients
    """
    # Compute statistics
    num_unique_patients = df[var_id].nunique()
    num_total_reports = len(df)
    mean_reports_per_patient = df.groupby(var_id).size().mean()
    num_deceased = df[df[var_death] == 1][var_id].nunique()
    num_censored = df[df[var_death] == 0][var_id].nunique()

    # Print results
    print(f"Total number of patients in the dataset: {num_unique_patients}")
    print(f"Total number of medical reports: {num_total_reports}")
    print(f"Average number of reports per patient: {mean_reports_per_patient:.2f}")
    print(f"Number of deceased patients: {num_deceased}")
    print(f"Number of censored patients: {num_censored}")


def plot_report_distribution_per_patient(df, var_id, export_path=None):
    """
    Plots the distribution of the number of reports per patient.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a column 'ID' indicating patient identifiers.
    export_path : str or None, default=None
        If provided, saves the plot to this path as a PNG file.
    """
    report_counts = df[var_id].value_counts()

    plt.figure(figsize=(10, 6))
    sns.histplot(report_counts, bins=20, kde=True, color='green')

    plt.xlabel("Number of reports per patient", fontsize=13)
    plt.ylabel("Frequency", fontsize=13)
    plt.title("Distribution of number of reports per patient", fontsize=15)
    plt.grid(True)

    if export_path:
        plt.savefig(export_path, dpi=300, bbox_inches='tight')

    plt.show()




########################################################################################
#                                                                                      #
#                                                                                      #
#                                                                                      #
#                                     DATA PREPROCESS                                  #
#                                                                                      #
#                                                                                      #
#                                                                                      #
########################################################################################


def convert_date_columns(data, verbose=True, date_format="%Y-%m-%d"):
    """
    Automatically convert columns that likely contain dates into pandas datetime format.

    This function scans the column names of the DataFrame and identifies any columns
    starting with 'DATE', 'Date_', or 'date_' as potential date fields. It then attempts
    to parse their values using `pd.to_datetime` with the specified date format, coercing
    invalid entries to NaT.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame potentially containing date-like columns.
    verbose : bool, default=True
        Whether to print messages when missing values (NaT) are detected after conversion.
    date_format : str, default="%Y-%m-%d"
        Expected date format for parsing (e.g. ISO format: year-month-day).

    Returns
    -------
    pd.DataFrame
        The original DataFrame with specified columns converted to datetime format.
    """
    date_columns = [col for col in data.columns if col.startswith('DATE')
                    or col.startswith('Date_') or col.startswith('date_')]

    for col in date_columns:
        data[col] = pd.to_datetime(data[col], format=date_format, errors='coerce')
        if data[col].isna().any() and verbose:
            print(f"Warning: Column '{col}' contains NaT values after conversion with format '{date_format}'.")

    return data



def convert_embedding_strings(
    df,
    var_embd="embeddings"
):
    """
    Convert string representations of embeddings into NumPy arrays.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    var_embd : str, default="embeddings"
        Name of the embedding column.

    Returns
    -------
    pd.DataFrame
        DataFrame with embeddings stored as NumPy arrays.
    """

    df = df.copy()

    df[var_embd] = df[var_embd].apply(
        lambda x: np.fromstring(
            x.strip("[]"),
            sep=" "
        )
        if isinstance(x, str)
        else np.asarray(x)
    )

    return df





def plot_survival_time_distribution(
    df,
    patient_id_col="SUBJECT_ID",
    event_col="delta_i",
    duration_col="survival_time",
    bins=50,
    figsize=(10, 6),
    alpha=0.6,
    color_event="#B21D61",
    color_censored="#1f77b4",
    verbose=True
):
    """
    Plot the distribution of survival times at the patient level.

    Each patient contributes exactly one observation.
    The first row of each patient is retained.

    Parameters
    ----------
    df : pd.DataFrame
        Longitudinal dataframe.
    patient_id_col : str, default="SUBJECT_ID"
        Patient identifier.
    event_col : str, default="delta_i"
        Event indicator (1=death, 0=censored).
    duration_col : str, default="survival_time"
        Survival duration in days.

    Returns
    -------
    df_patient : pd.DataFrame
        One-row-per-patient dataframe.
    stats_deceased : pd.Series
        Descriptive statistics for deceased patients.
    stats_censored : pd.Series
        Descriptive statistics for censored patients.
    """

    # --------------------------------------------------
    # One row per patient
    # --------------------------------------------------
    df_patient = (
        df.groupby(patient_id_col, as_index=False)
          .first()
    )

    deceased = df_patient[df_patient[event_col] == 1]
    censored = df_patient[df_patient[event_col] == 0]

    # --------------------------------------------------
    # Histogram
    # --------------------------------------------------
    plt.figure(figsize=figsize)

    sns.histplot(
        deceased[duration_col],
        bins=bins,
        color=color_event,
        alpha=alpha,
        label="Death"
    )

    sns.histplot(
        censored[duration_col],
        bins=bins,
        color=color_censored,
        alpha=alpha,
        label="Censored"
    )

    plt.axvline(
        deceased[duration_col].median(),
        color=color_event,
        linestyle="--",
        linewidth=2,
        label="Median (death)"
    )

    plt.axvline(
        censored[duration_col].median(),
        color=color_censored,
        linestyle="--",
        linewidth=2,
        label="Median (censored)"
    )

    plt.title(
        "Distribution of survival times",
        fontsize=16
    )

    plt.xlabel(
        "Survival time (days)",
        fontsize=14
    )

    plt.ylabel(
        "Number of patients",
        fontsize=14
    )

    plt.legend()
    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.5
    )

    plt.show()

    # --------------------------------------------------
    # Statistics
    # --------------------------------------------------
    stats_deceased = deceased[duration_col].describe()
    stats_censored = censored[duration_col].describe()

    if verbose:

        print(
            f"Number of unique patients: "
            f"{df_patient[patient_id_col].nunique()}"
        )

        print(
            f"Number of deceased patients: "
            f"{len(deceased)}"
        )

        print(
            f"Number of censored patients: "
            f"{len(censored)}"
        )

        print("\nDeceased patients")
        print("-----------------")
        print(stats_deceased)

        print("\nCensored patients")
        print("-----------------")
        print(stats_censored)

    return (
        df_patient,
        stats_deceased,
        stats_censored
    )





def prep_import(
    df_longitudinal,
    signature_order=2,
    use_log_signature=False,
    use_levy_area=False,
    print_progress=False,
    patient_id_col="SUBJECT_ID",
    embedding_col="embeddings",
    event_col="delta_i",
    duration_col="survival_time",
    survival_time_col="survival_time",
    survival_event_col="event",
    verbose=True
):
    """
    Prepare longitudinal clinical data for survival analysis using
    path-signature features extracted from sequential embeddings.

    This function performs:
    - temporal normalization,
    - path signature extraction,
    - signature preprocessing,
    - Cox-compatible feature extraction,
    - survival outcome formatting.

    Parameters
    ----------
    df_longitudinal : pd.DataFrame
        Longitudinal dataset containing sequential clinical reports.
    signature_order : int, default=3
        Truncation order of the path signature.
    use_log_signature : bool, default=False
        Whether to compute log-signatures instead of classical signatures.
    use_levy_area : bool, default=False
        Whether to include Lévy area features.
    print_progress : bool, default=False
        Whether to print intermediate progress messages.
    patient_id_col : str, default="ID"
        Column identifying patients.
    embedding_col : str, default="embeddings"
        Column containing embedding vectors.
    event_col : str, default="DEATH"
        Binary event indicator column.
    duration_col : str, default="duration"
        Survival duration column.
    survival_time_col : str, default="time"
        Name of the output survival time column.
    survival_event_col : str, default="event"
        Name of the output survival event column.
    verbose : bool, default=True
        Whether to enable verbose mode during signature extraction.

    Returns
    -------
    Xt : np.ndarray
        Signature feature matrix.
    y : np.ndarray
        Structured survival outcome array.
    feature_names : list
        Names of signature features.
    nbr_signature_features : int
        Number of signature coefficients.
    patient_ids : np.ndarray
        Patient identifiers retained in the final dataset.
    df_study : pd.DataFrame
        Final dataframe combining features and survival outcomes.
    """

    start_time = time.time()

    df = df_longitudinal.copy()

    # --------------------------------------------------
    # Normalize timestamps between 0 and 1
    # --------------------------------------------------
    df_time_normalized = preprocess_time(
        df,
        patient_id_col=patient_id_col
    )

    if print_progress:
        print("*" * 30)
        print("Time normalization completed.")

    # --------------------------------------------------
    # Signature extraction
    # --------------------------------------------------
    df_signature, nbr_signature_features = (
        signature_extract(
            df_time_normalized,
            order=signature_order,
            var_patient=patient_id_col,
            var_embd=embedding_col,
            verbose=verbose
        )
    )

    if print_progress:
        print("*" * 30)
        print("Signature extraction completed.")

    # --------------------------------------------------
    # Signature preprocessing
    # --------------------------------------------------
    df_signature = preprocess_sign(
        df_signature,
        retire_small=False,
        return_id=False
    )

    if print_progress:
        print("*" * 30)
        print("Signature preprocessing completed.")


    # df_signature = df_signature.rename(columns={"delta_i": "event"})
    # df_signature = df_signature.rename(columns={"survival_time": "time"})
    
    # --------------------------------------------------
    # Cox-compatible preprocessing
    # --------------------------------------------------
    df_signature_filtered, feature_names, patient_ids = (
        preprocess_cox(
            df_signature,
            var_death=event_col,
            var_duration=duration_col,
            var_id=patient_id_col,
            return_id=True
        )
    )

    # --------------------------------------------------
    # Final feature extraction
    # --------------------------------------------------
    Xt, y, patient_ids = feat_event_extract(
        df_signature_filtered,
        features=feature_names,
        var_id=patient_id_col,
        var_DEATH=event_col,
        var_duree=duration_col
    )

    if print_progress:
        print("Survival preprocessing completed.")
        print(
            f"Total preprocessing time: "
            f"{time.time() - start_time:.2f} seconds"
        )
        print("-" * 70)

    # --------------------------------------------------
    # Final study dataframe
    # --------------------------------------------------
    df_study = pd.DataFrame(Xt,columns=feature_names)

    df_study.insert(0,patient_id_col,patient_ids)

    df_study[survival_event_col] = y["event"].astype(bool)
    df_study[survival_time_col] = y["time"].astype(float)


    if print_progress:
        print(f"Final study dataframe shape: {df_study.shape}")

    return (
        Xt,
        y,
        feature_names,
        nbr_signature_features,
        patient_ids,
        df_study
    )







def make_train_test(
    df_OG,
    var_id="SUBJECT_ID",
    var_date="note_time",
    min_date="1990-01-01",
    n_group=10,
    random_state=177,
    size_test=0.5,
    verbose=True
):
    """
    Split a longitudinal dataset into one training set and several test sets.

    Splitting is performed at the patient level to avoid information leakage.
    All observations from a given patient remain in the same subset.

    Parameters
    ----------
    df_OG : pd.DataFrame
        Longitudinal dataframe.
    var_id : str, default="SUBJECT_ID"
        Patient identifier column.
    var_date : str, default="note_time"
        Report datetime column.
    min_date : str or None, default="1990-01-01"
        Optional minimum first-report date.
    n_group : int, default=10
        Number of test subsets.
    random_state : int, default=177
        Random seed.
    size_test : float, default=0.5
        Proportion of patients allocated to the test set.
    verbose : bool, default=True
        Whether to print dataset statistics.

    Returns
    -------
    df_train : pd.DataFrame
        Training dataframe.
    test_groups : list
        List of test dataframes.
    """

    df = df_OG.copy()

    if var_date in df.columns:
        df[var_date] = pd.to_datetime(df[var_date],errors="coerce")

    if verbose:
        print(f"Initial rows: {len(df)}\nInitial patients: {df[var_id].nunique()}")

    # Optional date filtering
    if min_date is not None:
        min_date = pd.to_datetime(min_date)
        first_dates = df.groupby(var_id)[var_date].transform("min")
        df = df[first_dates >= min_date]

        if verbose:
            print(f"Patients after date filtering: {df[var_id].nunique()}")

    # Unique patients indexes
    unique_ids = df[var_id].unique()

    if len(unique_ids) == 0:
        raise ValueError(
            "No patients remain after filtering. "
            "Check `min_date` and the datetime column."
        )

    # Train/test split
    train_ids, test_ids = train_test_split(unique_ids,test_size=size_test,random_state=random_state)

    df_train = df[df[var_id].isin(train_ids)]
    df_test = df[df[var_id].isin(test_ids)]


    # Split test patients into groups
    test_id_groups = np.array_split(test_ids,n_group)
    test_groups = [df_test[df_test[var_id].isin(ids)] for ids in test_id_groups]
    
    # Verbose summary
    if verbose:
        print(f"Training patients: {df_train[var_id].nunique()}")
        for i, group in enumerate(test_groups,start=1):
            print(f"Test group {i}: {group[var_id].nunique()} patients")

    return df_train, test_groups

    

def make_df_conform0(
    df: pd.DataFrame,
    var_id: str = 'ID',
    var_start: str = 'date_start',
    var_end: str = 'date_end',
    var_death: str = 'DEATH',
    var_death_date: str = 'date_death',
    var_T: str = 'Ti',
    var_known: Optional[str] = None,     # e.g. 'duration_known'
    var_gap: str = 'death_know_gap',
    limite_gap: Optional[int] = None,    # ← FIXED here
    verbose: bool = True
):
    """
    Conform dataset for survival analysis using patient-level T_days.
    Returns (df_filtered, df_last_obs_one_row_per_patient, id_list).
    """
    # Ensure datetime
    for col in [var_start, var_end, var_death_date]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # If T_days missing, compute once from last row per patient
    if var_T not in df.columns:
        if verbose:
            print(f"[make_df_conform] '{var_T}' not found — computing from dates.")
        df_last_obs_tmp = (
            df.sort_values(by=var_end)
              .groupby(var_id, as_index=False)
              .last()
              .loc[:, [var_id, var_death, var_start, var_end, var_death_date]]
              .assign(**{
                  var_T: lambda x: np.where(
                      x[var_death] == 1,
                      (x[var_death_date] - x[var_start]).dt.days,
                      (x[var_end] - x[var_start]).dt.days
                  )
              })
        )
        df = df.merge(df_last_obs_tmp[[var_id, var_T]], on=var_id, how='left')

    # Build one-line-per-patient view
    keep_cols = [c for c in [var_id, var_death, var_start, var_end, var_death_date, var_T, var_known] if c]
    df_last_obs = (
        df.sort_values(by=var_end)
          .groupby(var_id, as_index=False)
          .last()[keep_cols]
          .copy()
    )

    # Compute death gap only for deceased
    if var_known:
        if var_known not in df_last_obs.columns:
            raise ValueError(f"'{var_known}' not found in DataFrame.")
        df_last_obs[var_gap] = np.where(
            df_last_obs[var_death] == 1,
            (df_last_obs[var_death_date] - (df_last_obs[var_start] + pd.to_timedelta(df_last_obs[var_known], unit='d'))).dt.days,
            np.nan
        )

    # Filter invalid cases
    df_last_obs = (
        df_last_obs
        .dropna(subset=[var_start, var_end, var_T])
        .loc[df_last_obs[var_T] > 0]
        .reset_index(drop=True)
    )
    id_list = df_last_obs[var_id].values

    # Filter df accordingly
    initial_lines = len(df)
    initial_ids = df[var_id].nunique()
    df = df[df[var_id].isin(id_list)].copy()
    final_lines = len(df)
    final_ids = df[var_id].nunique()

    if verbose:
        print(f"Removed rows: {initial_lines - final_lines}")
        print(f"Removed patient IDs: {initial_ids - final_ids}")
        kept_prop = (final_ids / initial_ids) if initial_ids else 0.0
        print(f"Kept patient IDs: {final_ids}/{initial_ids} ({kept_prop:.1%})")

    # Optional: filter by death gap threshold
    if limite_gap is not None and var_known:
        pre_gap_n = df_last_obs.shape[0]
        df_last_obs = df_last_obs[df_last_obs[var_gap] <= limite_gap].copy()
        removed_gap = pre_gap_n - df_last_obs.shape[0]
        id_list = df_last_obs[var_id].values
        df = df[df[var_id].isin(id_list)].copy()
        if verbose:
            print(f"Removed by {var_gap} ≤ {limite_gap}: {removed_gap} patients")

    return df, df_last_obs, id_list




def make_df_conform(
    df,
    var_id="SUBJECT_ID",
    var_time="note_time",
    var_death="delta_i",
    var_T="Ti",
    var_death_time="death_time",
    verbose=True
):
    """
    Prepare a longitudinal dataset for survival analysis.

    The function:
    - ensures datetime consistency,
    - constructs a one-row-per-patient dataframe using the last report,
    - removes invalid patients,
    - filters the longitudinal dataframe accordingly.

    Parameters
    ----------
    df : pd.DataFrame
        Longitudinal dataframe.
    var_id : str, default="SUBJECT_ID"
        Patient identifier column.
    var_time : str, default="note_time"
        Report timestamp column.
    var_death : str, default="delta_i"
        Event indicator column.
    var_T : str, default="Ti"
        Survival duration column.
    var_death_time : str, default="death_time"
        Death timestamp column.
    verbose : bool, default=True
        Whether to print filtering statistics.

    Returns
    -------
    df : pd.DataFrame
        Filtered longitudinal dataframe.
    df_last_obs : pd.DataFrame
        One-row-per-patient dataframe.
    id_list : np.ndarray
        Retained patient identifiers.
    """

    df = df.copy()

    if var_time in df.columns:
        df[var_time] = pd.to_datetime(
            df[var_time],
            errors="coerce"
        )

    if var_death_time in df.columns:
        df[var_death_time] = pd.to_datetime(
            df[var_death_time],
            errors="coerce"
        )

    keep_cols = [
        c for c in [
            var_id,
            var_time,
            var_death,
            var_T,
            var_death_time
        ]
        if c in df.columns
    ]

    df_last_obs = (
        df.sort_values(var_time)
          .groupby(var_id, as_index=False)
          .last()[keep_cols]
          .copy()
    )

    df_last_obs = (
        df_last_obs
        .dropna(subset=[var_T])
        .loc[df_last_obs[var_T] > 0]
        .reset_index(drop=True)
    )

    id_list = df_last_obs[var_id].values

    initial_rows = len(df)
    initial_patients = df[var_id].nunique()

    df = df[
        df[var_id].isin(id_list)
    ].copy()

    final_rows = len(df)
    final_patients = df[var_id].nunique()

    if verbose:

        print(
            f"Removed rows: "
            f"{initial_rows - final_rows}"
        )

        print(
            f"Removed patient IDs: "
            f"{initial_patients - final_patients}"
        )

        kept_prop = (
            final_patients / initial_patients
            if initial_patients > 0
            else 0.0
        )

        print(
            f"Kept patient IDs: "
            f"{final_patients}/{initial_patients} "
            f"({kept_prop:.1%})"
        )

    return df, df_last_obs, id_list




def global_sigbert_mimic_pipeline(
    df_full,
    df_train,
    test_groups,
    projection_matrix,
    mean_embedding,
    lambda_l1=0.7,
    signature_order=2,
    use_levy_area=False,
    print_progress=False,
    patient_id_col="SUBJECT_ID",
    embedding_col="embeddings",
    timestamp_col="note_time",
    event_col="delta_i",
    duration_col="survival_time",
    var_T="Ti",
    cox_model_type="sk_cox",
    use_standard_scaling=False
):
    """
    End-to-end SigBERT pipeline for longitudinal MIMIC data.

    The Cox model is trained once on `df_train`, then evaluated
    independently on every dataframe contained in `test_groups`.

    Returns
    -------
    tuple
        df_results
            Global summary of the pipeline.

        cph
            Trained Cox model.

        df_survival
            Concatenation of the training survival dataframe and all
            test survival dataframes.

        w_cox
            Cox regression coefficients.

        scores
            Training risk scores.

        X
            Concatenation of X_train and every X_test matrix.

        y_train
            Training survival outcomes.

        y_cox
            Cox-formatted training outcomes.

        c_index_train
            Training C-index.

        c_index_test_list
            Test C-index for each test group.

        c_index_test_mean
            Mean test C-index.

        c_index_test_std
            Standard deviation of test C-indexes.

        df_survival_test_list
            List containing one survival dataframe per test group.

        df_test_metrics
            Test metrics by group.

        X_test_list
            List containing one design matrix per test group.

        y_test_list
            List containing one survival outcome array per test group.
    """

    from sklearn.preprocessing import StandardScaler

    print("\n### Running SigBERT pipeline on MIMIC dataset ###")

    time_start = time.time()

    # ==================================================
    # 1. Input validation
    # ==================================================
    if not isinstance(test_groups, (list, tuple)):
        raise TypeError(
            "`test_groups` must be a list or tuple of dataframes."
        )

    if len(test_groups) == 0:
        raise ValueError("`test_groups` is empty.")

    train_raw_ids = set(
        df_train[patient_id_col]
        .dropna()
        .unique()
    )

    seen_test_ids = set()

    for i, df_test_i in enumerate(test_groups, start=1):

        if not isinstance(df_test_i, pd.DataFrame):
            raise TypeError(
                f"Test group {i} is not a pandas DataFrame."
            )

        if df_test_i.empty:
            raise ValueError(
                f"Test group {i} is empty. "
                "Reduce `n_group` in `make_train_test`."
            )

        current_ids = set(
            df_test_i[patient_id_col]
            .dropna()
            .unique()
        )

        train_overlap = train_raw_ids.intersection(current_ids)

        if train_overlap:
            raise ValueError(
                f"Patient leakage between train and test group {i}: "
                f"{len(train_overlap)} overlapping patients."
            )

        previous_test_overlap = seen_test_ids.intersection(
            current_ids
        )

        if previous_test_overlap:
            raise ValueError(
                f"Patient overlap between test groups before group {i}: "
                f"{len(previous_test_overlap)} patients."
            )

        seen_test_ids.update(current_ids)

    print(
        f"Number of test groups received: "
        f"{len(test_groups)}"
    )

    # ==================================================
    # 2. Train formatting
    # ==================================================
    (
        df_train_processed,
        df_train_last_obs,
        _
    ) = make_df_conform(
        df_train,
        var_id=patient_id_col,
        var_time=timestamp_col,
        var_death=event_col,
        var_T=var_T,
        verbose=False
    )

    total_train_patients = (
        df_train_processed[patient_id_col].nunique()
    )

    print(
        f"Total number of patients in train set: "
        f"{total_train_patients}"
    )

    # ==================================================
    # 3. Independent formatting of every test group
    # ==================================================
    processed_test_groups = []
    test_last_obs_groups = []

    for i, df_test_i in enumerate(test_groups, start=1):

        (
            df_test_i_processed,
            df_test_i_last_obs,
            _
        ) = make_df_conform(
            df_test_i,
            var_id=patient_id_col,
            var_time=timestamp_col,
            var_death=event_col,
            var_T=var_T,
            verbose=False
        )

        if df_test_i_processed.empty:
            raise ValueError(
                f"Test group {i} is empty after "
                "`make_df_conform`."
            )

        processed_test_groups.append(
            df_test_i_processed
        )

        test_last_obs_groups.append(
            df_test_i_last_obs
        )

        print(
            f"Test group {i}: "
            f"{df_test_i_processed[patient_id_col].nunique()} "
            f"patients, {len(df_test_i_processed)} reports"
        )

    df_test_all = pd.concat(
        processed_test_groups,
        axis=0,
        ignore_index=True
    )

    df_test_last_obs_all = pd.concat(
        test_last_obs_groups,
        axis=0,
        ignore_index=True
    )

    total_test_patients = (
        df_test_all[patient_id_col].nunique()
    )

    print(
        "Total number of patients across all test groups: "
        f"{total_test_patients}"
    )

    # ==================================================
    # 4. Global descriptive statistics
    # ==================================================
    df_all_processed = pd.concat(
        [
            df_train_processed,
            df_test_all
        ],
        axis=0,
        ignore_index=True
    )

    reports_per_patient = (
        df_all_processed
        .groupby(patient_id_col)
        .size()
    )

    total_unique_patients = (
        df_all_processed[patient_id_col].nunique()
    )

    total_reports = len(df_all_processed)

    mean_reports_per_patient = (
        reports_per_patient.mean()
    )

    std_reports_per_patient = (
        reports_per_patient.std()
    )

    total_deceased_patients = (
        df_all_processed.loc[
            df_all_processed[event_col] == 1,
            patient_id_col
        ]
        .nunique()
    )

    total_censored_patients = (
        df_all_processed.loc[
            df_all_processed[event_col] == 0,
            patient_id_col
        ]
        .nunique()
    )

    # ==================================================
    # 5. Study-duration statistics
    # ==================================================
    df_all_last_obs = pd.concat(
        [
            df_train_last_obs,
            df_test_last_obs_all
        ],
        axis=0,
        ignore_index=True
    )

    study_times = pd.to_numeric(
        df_all_last_obs[var_T],
        errors="coerce"
    )

    mean_study_time = study_times.mean()
    std_study_time = study_times.std(ddof=0)

    # ==================================================
    # 6. Train projection and signature extraction
    # ==================================================
    df_train_proj = apply_linear_projection(
        df_train_processed,
        projection_matrix=projection_matrix,
        mean_embedding=mean_embedding
    )

    (
        Xt_train,
        y_train,
        feature_names,
        nbr_sig,
        train_ids,
        df_study_train
    ) = prep_import(
        df_train_proj,
        signature_order=signature_order,
        use_levy_area=use_levy_area,
        print_progress=print_progress,
        patient_id_col=patient_id_col,
        embedding_col=embedding_col,
        event_col=event_col,
        duration_col=duration_col,
        verbose=print_progress
    )

    # ==================================================
    # 7. Optional scaling fitted only on train
    # ==================================================
    scaler = None

    if use_standard_scaling:
        scaler = StandardScaler()

        Xt_train = scaler.fit_transform(
            Xt_train
        )

    # ==================================================
    # 8. Cox model training
    # ==================================================
    print(
        "\n--------------- Cox training ---------------"
    )

    (
        cph,
        df_survival_train,
        w_cox,
        scores,
        X_train,
        y_cox,
        c_index_train,
        log_likelihood,
        _
    ) = global_cox_train(
        Xt_train,
        y_train,
        id_list_train=train_ids,
        learning_cox_map=cox_model_type,
        lambda_l1_CV=lambda_l1
    )

    print(
        "--------------------------------------------"
    )

    # Identify the training rows in the final dataframe
    df_survival_train = df_survival_train.copy()
    df_survival_train["dataset_split"] = "train"
    df_survival_train["test_group"] = pd.NA

    # ==================================================
    # 9. Independent evaluation of every test group
    # ==================================================
    X_test_list = []
    y_test_list = []

    df_survival_test_list = []
    c_index_test_list = []
    test_metrics_rows = []

    for i, df_test_i_processed in enumerate(
        processed_test_groups,
        start=1
    ):
        current_group_ids = set(
            df_test_i_processed[patient_id_col]
            .dropna()
            .unique()
        )

        print(
            f"\n{'=' * 60}\n"
            f"Evaluation on test group "
            f"{i}/{len(processed_test_groups)}\n"
            f"Patients before signature: "
            f"{len(current_group_ids)}\n"
            f"Reports: {len(df_test_i_processed)}\n"
            f"{'=' * 60}"
        )

        df_test_i_proj = apply_linear_projection(
            df_test_i_processed,
            projection_matrix=projection_matrix,
            mean_embedding=mean_embedding
        )

        (
            Xt_test,
            y_test,
            _,
            _,
            test_ids,
            df_study_test
        ) = prep_import(
            df_test_i_proj,
            signature_order=signature_order,
            use_levy_area=use_levy_area,
            print_progress=print_progress,
            patient_id_col=patient_id_col,
            embedding_col=embedding_col,
            event_col=event_col,
            duration_col=duration_col,
            verbose=print_progress
        )

        evaluated_ids = set(
            np.asarray(test_ids).tolist()
        )

        if not evaluated_ids.issubset(
            current_group_ids
        ):
            raise RuntimeError(
                f"Test group {i} contains unexpected "
                "patient IDs after signature preprocessing."
            )

        if use_standard_scaling:
            Xt_test = scaler.transform(
                Xt_test
            )

        (
            df_survival_test,
            c_index_test,
            X_test,
            y_test_final
        ) = skglm_datatest(
            Xt_test,
            y_test,
            w_cox,
            cph,
            test_ids,
            print_all=False
        )
        
        X_test_list.append(
            np.asarray(X_test).copy()
        )
        
        y_test_list.append(
            y_test.copy()
        )
        
        df_survival_test = df_survival_test.copy()
        df_survival_test["dataset_split"] = "test"
        df_survival_test["test_group"] = i
        
        df_survival_test_list.append(
            df_survival_test
        )
        
        n_events = int(
            np.asarray(
                y_test["event"],
                dtype=int
            ).sum()
        )
        
        n_censored = len(y_test) - n_events
        
        c_index_test_list.append(
            c_index_test
        )


        test_metrics_rows.append({
            "Test group": i,
            "Number of reports": len(
                df_test_i_processed
            ),
            "Patients before signature": len(
                current_group_ids
            ),
            "Patients evaluated": len(
                test_ids
            ),
            "Events": n_events,
            "Censored": n_censored,
            "C-index": c_index_test
        })

    df_test_metrics = pd.DataFrame(
        test_metrics_rows
    )

    # ==================================================
    # 10. Mean and standard deviation across test groups
    # ==================================================
    c_index_array = np.asarray(
        c_index_test_list,
        dtype=float
    )

    valid_c_indexes = c_index_array[
        np.isfinite(c_index_array)
    ]

    if len(valid_c_indexes) == 0:
        c_index_test_mean = np.nan
        c_index_test_std = np.nan

    else:
        c_index_test_mean = np.mean(
            valid_c_indexes
        )

        c_index_test_std = (
            np.std(
                valid_c_indexes,
                ddof=1
            )
            if len(valid_c_indexes) >= 2
            else np.nan
        )

    total_runtime = (
        time.time() - time_start
    )

    print("\nTest-group results")
    print("------------------")
    print(
        df_test_metrics.to_string(
            index=False
        )
    )

    print(
        f"\nMean test C-index: "
        f"{c_index_test_mean:.4f}"
    )

    if np.isfinite(c_index_test_std):
        print(
            f"Std test C-index: "
            f"{c_index_test_std:.4f}"
        )
    else:
        print(
            "Std test C-index: NaN "
            "(fewer than two valid test scores)"
        )

    # ==================================================
    # 11. Global summary
    # ==================================================
    df_results = pd.DataFrame([{
        "Mean C-index": c_index_test_mean,
        "Std C-index": c_index_test_std,
        "Number of Test Groups": len(
            processed_test_groups
        ),
        "Number of Valid Test Scores": len(
            valid_c_indexes
        ),
        "Total Deceased Patients": (
            total_deceased_patients
        ),
        "Total Censored Patients": (
            total_censored_patients
        ),
        "Total Unique Patients": (
            total_unique_patients
        ),
        "Total Number of Reports": total_reports,
        "Mean Study Time (days)": round(
            mean_study_time,
            3
        ),
        "Std Study Time (days)": round(
            std_study_time,
            3
        ),
        "Execution Time (s)": round(
            total_runtime,
            2
        ),
        "Mean Reports per Patient": round(
            mean_reports_per_patient,
            3
        ),
        "Std Reports per Patient": round(
            std_reports_per_patient,
            3
        )
    }])

    # ==================================================
    # 12. Concatenate all train and test outputs
    # ==================================================
    df_survival = pd.concat(
        [
            df_survival_train,
            *df_survival_test_list
        ],
        axis=0,
        ignore_index=True
    )

    X = np.vstack(
        [
            np.asarray(X_train),
            *[
                np.asarray(X_test)
                for X_test in X_test_list
            ]
        ]
    )

    return (
        df_results,
        cph,
        df_survival,
        w_cox,
        scores,
        X,
        y_train,
        y_cox,
        c_index_train,
        c_index_test_list,
        c_index_test_mean,
        c_index_test_std,
        df_survival_test_list,
        df_test_metrics,
        X_test_list,
        y_test_list
    )
########################################################################################
#                                                                                      #
#                                                                                      #
#                                                                                      #
#                                   LANDMARK APPROACH                                  #
#                                                                                      #
#                                                                                      #
#                                                                                      #
########################################################################################



def compute_survival_time(df: pd.DataFrame,
                          var_id: str = 'ID',
                          var_death: str = 'DEATH',
                          var_start: str = 'date_start',
                          var_end: str = 'date_end',
                          var_death_date: str = 'date_death',
                          new_var_time: str = 'T_days',
                          verbose: bool = True):
    """
    Compute survival time Ti = (date_death - date_start) if death occurred,
    else (date_end - date_start). The result is added to df for each patient's
    observations and also returned as a patient-level summary.
    """
    # Ensure datetime format
    for col in [var_start, var_end, var_death_date]:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    # Compute Ti for each patient (one row per ID)
    df_surv = (
        df.sort_values(by=var_end)
          .groupby(var_id, as_index=False)
          .last()
          .loc[:, [var_id, var_death, var_start, var_end, var_death_date]]
          .assign(
              **{
                  new_var_time: lambda x: np.where(
                      x[var_death] == 1,
                      (x[var_death_date] - x[var_start]).dt.days,
                      (x[var_end] - x[var_start]).dt.days
                  )
              }
          )
    )

    # Merge survival times back to full df
    df = df.merge(df_surv[[var_id, new_var_time]], on=var_id, how='left')

    if verbose:
        print(f"[compute_survival_time] {df_surv.shape[0]} patients processed.")
        print(f"  Mean T: {df_surv[new_var_time].mean():.1f} days "
              f"({df_surv[new_var_time].std():.1f} SD)")
        print(f"  Deaths: {df_surv[var_death].sum()} / {df_surv.shape[0]}")

    return df, df_surv



def define_landmark_cohort(df: pd.DataFrame,
                           landmark_months: int,
                           var_time: str = 'note_time',
                           var_id: str = 'SUBJECT_ID',
                           var_T: str = 'survival_time',
                           var_since_start: str = 'time_since_first_note',
                           var_death :str = 'delta_i',
                           window_months: int = 6,
                           verbose: bool = True):
    """
    Build the landmark cohort for a given relative time (in months) since baseline.
    Patients with T_days >= L_days are kept, reports restricted to [L-w, L].
    Returns:
        - df_L : sequential data restricted to [L-w, L]
        - patients_in_study : list of eligible IDs
        - df_gamma : one row per patient with gamma_i(L)
    """

    # Ensure datetime types
    for col in [var_time]:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    # Convert months to days
    L_days = int(landmark_months * 30)
    w_days = int(window_months * 30)

    # Patients still in study at the landmark: T_days >= L_days
    patients_in_study = df.loc[df[var_T] >= L_days, var_id].unique().tolist()

    # Subset: keep only patients at risk
    df_sub = df[df[var_id].isin(patients_in_study)].copy()

    # Apply time window [L-w, L]
    total_obs_before = len(df_sub)
    df_L = df_sub[(df_sub[var_since_start] >= (L_days - w_days)) &
                  (df_sub[var_since_start] <= L_days)].copy()
    total_obs_after = len(df_L)
    obs_removed = total_obs_before - total_obs_after
    prop_removed = obs_removed / total_obs_before if total_obs_before > 0 else 0

    # ---------------------------------------------------------------
    # Compute gamma_i(L), but DO NOT MERGE into sequential dataframe
    # ---------------------------------------------------------------
    first_obs = (
        df_L.groupby(var_id)[var_since_start]
        .min()
        .reset_index()
        .rename(columns={var_since_start: 'first_obs_days'})
    )
    first_obs['gamma'] = ((L_days - first_obs['first_obs_days']) < w_days).astype(int)

    # Prepare gamma output: only ID and gamma
    df_gamma = first_obs[[var_id, 'gamma']].copy()
    # ---------------------------------------------------------------

    # ===============================================================
    # === IMPORTANT LANDMARK FIX: UPDATE RESIDUAL SURVIVAL TIME R ===
    # ===============================================================
    df_L["R"] = df_L[var_T] - L_days
    # ===============================================================

    # ====================================================================
    # === SECOND IMPORTANT FIX: UPDATE EVENT INDICATOR FOR LANDMARKING ===
    # === δ_i(L) = 1 if patient dies after L; 0 otherwise                ===
    # ====================================================================
    df_L["DEATH_L"] = (
        (df_L[var_T] > L_days) &
        (df_L[var_death] == 1)
    ).astype(int)
    # ====================================================================

    # --- Verbose summary ---
    if verbose:
        n_total = df[var_id].nunique()
        n_kept = len(patients_in_study)
        n_gamma = df_gamma['gamma'].sum()
        prop_gamma = n_gamma / n_kept if n_kept > 0 else 0

        print(f"[Landmark {landmark_months} mo] Patients kept: {n_kept}/{n_total} "
              f"({n_total - n_kept} excluded)")
        print(f"  → Short-history γ=1: {n_gamma}/{n_kept} = {prop_gamma:.1%}")
        print(f"  → Observations kept: {total_obs_after}/{total_obs_before} "
              f"({100 * (1 - prop_removed):.1f}% retained, {100 * prop_removed:.1f}% removed)")
        print(f"  → Residual survival R_i computed (min={df_L['R'].min()} days).")
        print(f"  → Landmark events computed: DEATH_L sum = {df_L['DEATH_L'].sum()}.")

    return df_L, patients_in_study, df_gamma






# UPDATE 2026




def prep_signature_cox(
    df_OG,
    order_sign=2,
    use_log=False,
    print_progress=False,
    var_id="ID",
    var_embd="embeddings",
    var_start="date_start",
    var_death="DEATH_L",
    var_duration="R",
    var_time="time",
    var_event="event",
    interpolation_type=None,
    var_struct_seq_list_OG=None,
    verbose=True
):
    """
    Prepare Cox-ready data using path-signature features extracted
    from sequential embeddings (landmark setting).
    """

    import time
    start = time.time()

    df = df_OG.copy()

    # ------------------------------------------------------------------
    # 1. Time normalization
    # ------------------------------------------------------------------
    df_time = preprocess_time(df)

    if print_progress:
        print("Time normalization completed.")

    # ------------------------------------------------------------------
    # 2. Signature extraction (landmark version)
    # ------------------------------------------------------------------
    df_sign, nbr_sig, nbr_levy = signature_extract(
        df_time,
        order=order_sign,
        var_embd=var_embd,
        interpolation_type=interpolation_type,
        var_struct_list=var_struct_seq_list_OG,
        verbose=verbose
    )

    if print_progress:
        print("Signature extraction completed.")

    # ------------------------------------------------------------------
    # 3. Signature preprocessing
    # ------------------------------------------------------------------
    df_sign = preprocess_sign(df_sign, retire_small=False, return_id=False, verbose=False)

    # ------------------------------------------------------------------
    # 4. Prepare Cox-compatible dataset
    # ------------------------------------------------------------------
    df_cox, features_name, id_list = preprocess_cox(
        df_sign,
        debut_etude=var_start,
        return_id=True,
        compute_duree = False
    )

    # ------------------------------------------------------------------
    # 5. Extract survival matrix
    # ------------------------------------------------------------------
    Xt, y, id_list = feat_event_extract(
        df_cox,
        features=features_name,
        var_id=var_id,
        var_DEATH=var_death,
        var_duree=var_duration
    )

    # ------------------------------------------------------------------
    # 6. Construct df_study
    # ------------------------------------------------------------------
    df_study = pd.DataFrame(Xt, columns=features_name)
    df_study.insert(0, var_id, id_list)
    df_study[var_event] = y[var_event].astype(bool)
    df_study[var_time] = y[var_time].astype(float)

    if print_progress:
        print(f"Preprocessing completed in {time.time() - start:.2f} seconds.")
        print(f"Final dataset shape: {df_study.shape}")

    return Xt, y, features_name, nbr_sig, nbr_levy, id_list, df_study






########################################################################################
#                                                                                      #
#                                                                                      #
#                                                                                      #
#                                   POST LASSO REFIT                                   #
#                                                                                      #
#                                                                                      #
#                                                                                      #
########################################################################################








def evaluate_post_lasso_on_test_groups(
    post_lasso_coefficients,
    X_test_list,
    y_test_list,
    selected_mask,
    lasso_c_index_list=None,
    group_names=None,
    plot=True
):
    """
    Evaluate post-LASSO Cox coefficients on several test groups.
    """

    selected_mask = np.asarray(
        selected_mask,
        dtype=bool
    )

    post_lasso_coefficients = np.asarray(
        post_lasso_coefficients,
        dtype=np.float64
    ).ravel()

    if len(X_test_list) != len(y_test_list):
        raise ValueError(
            "`X_test_list` and `y_test_list` must have "
            "the same length."
        )

    n_groups = len(X_test_list)

    if group_names is None:
        group_names = [
            f"Test group {i + 1}"
            for i in range(n_groups)
        ]

    if len(group_names) != n_groups:
        raise ValueError(
            "`group_names` must contain one name per test group."
        )

    if (
        lasso_c_index_list is not None
        and len(lasso_c_index_list) != n_groups
    ):
        raise ValueError(
            "`lasso_c_index_list` must contain one value "
            "per test group."
        )

    n_selected = int(selected_mask.sum())

    if len(post_lasso_coefficients) != n_selected:
        raise ValueError(
            "The number of post-LASSO coefficients must match "
            f"the number of selected covariates: "
            f"{len(post_lasso_coefficients)} != {n_selected}."
        )

    metrics = []
    post_lasso_c_index_list = []

    for group_index, (X_test, y_test) in enumerate(
        zip(X_test_list, y_test_list)
    ):
        X_test = np.asarray(
            X_test,
            dtype=np.float64
        )

        if X_test.ndim != 2:
            raise ValueError(
                f"Test matrix {group_index + 1} must be "
                "two-dimensional."
            )

        if X_test.shape[1] != len(selected_mask):
            raise ValueError(
                f"Test group {group_index + 1}: "
                f"{X_test.shape[1]} covariates found, but "
                f"the selection mask contains "
                f"{len(selected_mask)} entries."
            )

        X_test_selected = np.ascontiguousarray(
            X_test[:, selected_mask],
            dtype=np.float64
        )

        # Linear post-LASSO risk score
        risk_scores = (X_test_selected @ post_lasso_coefficients).ravel()

        events = np.asarray(
            y_test["event"],
            dtype=bool
        )

        durations = np.asarray(
            y_test["time"],
            dtype=np.float64
        )

        if len(risk_scores) != len(y_test):
            raise ValueError(
                f"Test group {group_index + 1}: "
                "risk scores and survival outcomes have "
                "different lengths."
            )

        c_index_post_lasso = (
            concordance_index_censored(
                events,
                durations,
                risk_scores
            )[0]
        )

        post_lasso_c_index_list.append(
            c_index_post_lasso
        )

        row = {
            "Test Group": group_names[group_index],
            "N Patients": len(y_test),
            "N Events": int(events.sum()),
            "Post-LASSO C-index": c_index_post_lasso
        }

        if lasso_c_index_list is not None:
            c_index_lasso = float(
                lasso_c_index_list[group_index]
            )

            row["LASSO C-index"] = c_index_lasso
            row["C-index Difference"] = (
                c_index_post_lasso
                - c_index_lasso
            )

        metrics.append(row)

    df_metrics = pd.DataFrame(metrics)

    if plot:
        plot_columns = []

        if lasso_c_index_list is not None:
            plot_columns.append(
                "LASSO C-index"
            )

        plot_columns.append(
            "Post-LASSO C-index"
        )

        ax = (
            df_metrics
            .set_index("Test Group")[plot_columns]
            .plot(
                kind="bar",
                figsize=(9, 5)
            )
        )

        ax.axhline(
            y=0.5,
            linestyle="--",
            linewidth=1,
            label="Random concordance"
        )

        ax.set_ylabel("C-index")
        ax.set_xlabel("")
        ax.set_title(
            "LASSO versus post-LASSO test performance"
        )
        ax.set_ylim(0.0, 1.0)
        ax.tick_params(
            axis="x",
            rotation=0
        )
        ax.legend()

        plt.tight_layout()
        plt.show()

    return (
        df_metrics,
        post_lasso_c_index_list
    )



def post_lasso_cox_refit(
    X_train,
    y_train,
    lasso_coefficients,
    df_survival=None,
    patient_ids=None,
    patient_id_col="SUBJECT_ID",
    learning_cox_map="sk_cox",
    coefficient_tolerance=1e-12,
    refit_lambda=0.0,
    verbose=True
):
    """
    Refit a Cox model using only the covariates selected by an
    L1-penalized Cox model.

    Covariates whose LASSO coefficients have an absolute value below
    `coefficient_tolerance` are removed. The Cox model is then refitted
    on the selected covariates, without penalization by default.

    Parameters
    ----------
    X_train : np.ndarray or pd.DataFrame
        Training covariate matrix with shape
        (n_patients, n_features).

    y_train : np.ndarray
        Structured survival outcome array used by
        `global_cox_train`.

    lasso_coefficients : np.ndarray
        Coefficients estimated by the initial LASSO Cox model.

    df_survival : pd.DataFrame or None, default=None
        Training survival dataframe, optionally used to recover
        patient identifiers.

    patient_ids : array-like or None, default=None
        Patient identifiers ordered consistently with the rows
        of `X_train`.

    patient_id_col : str, default="SUBJECT_ID"
        Patient identifier column in `df_survival`.

    learning_cox_map : str, default="sk_cox"
        Cox implementation passed to `global_cox_train`.

    coefficient_tolerance : float, default=1e-12
        Threshold below which a coefficient is considered zero.

    refit_lambda : float, default=0.0
        Penalization parameter used for the refit. A value of zero
        corresponds to an unpenalized post-LASSO refit.

    verbose : bool, default=True
        Whether to print information about selected covariates and
        training performance.

    Returns
    -------
    dict
        Dictionary containing the refitted model, selected covariates,
        refitted coefficients, training C-index, and other Cox outputs.
    """

    # --------------------------------------------------------------
    # 1. Convert the training matrix
    # --------------------------------------------------------------
    if isinstance(X_train, pd.DataFrame):
        feature_names = X_train.columns.to_numpy()

        X_array = X_train.to_numpy(
            dtype=np.float64
        )
    else:
        X_array = np.asarray(
            X_train,
            dtype=np.float64
        )

        if X_array.ndim != 2:
            raise ValueError(
                "`X_train` must be a two-dimensional matrix."
            )

        feature_names = np.array([
            f"feature_{j}"
            for j in range(X_array.shape[1])
        ])

    if X_array.ndim != 2:
        raise ValueError(
            "`X_train` must be a two-dimensional matrix."
        )

    if not np.isfinite(X_array).all():
        raise ValueError(
            "`X_train` contains NaN or infinite values."
        )

    # --------------------------------------------------------------
    # 2. Validate outcomes and LASSO coefficients
    # --------------------------------------------------------------
    if len(y_train) != X_array.shape[0]:
        raise ValueError(
            "The number of rows in `X_train` must match the "
            "number of observations in `y_train`."
        )

    lasso_coefficients = np.asarray(
        lasso_coefficients,
        dtype=np.float64
    ).ravel()

    if X_array.shape[1] != len(lasso_coefficients):
        raise ValueError(
            "The number of columns in `X_train` must match "
            "the number of LASSO coefficients."
        )

    if not np.isfinite(lasso_coefficients).all():
        raise ValueError(
            "`lasso_coefficients` contains NaN or infinite values."
        )

    # --------------------------------------------------------------
    # 3. Select covariates retained by the LASSO
    # --------------------------------------------------------------
    selected_mask = (
        np.abs(lasso_coefficients)
        > coefficient_tolerance
    )

    selected_indices = np.flatnonzero(
        selected_mask
    )

    if len(selected_indices) == 0:
        raise ValueError(
            "No covariate was selected by the LASSO Cox model."
        )

    X_selected = np.ascontiguousarray(
        X_array[:, selected_mask],
        dtype=np.float64
    )

    selected_feature_names = (
        feature_names[selected_mask]
    )

    # --------------------------------------------------------------
    # 4. Recover patient identifiers
    # --------------------------------------------------------------
    if patient_ids is None:
        if (
            df_survival is not None
            and patient_id_col in df_survival.columns
        ):
            patient_ids = (
                df_survival[patient_id_col]
                .to_numpy()
            )
        else:
            patient_ids = np.arange(
                X_selected.shape[0]
            )

    patient_ids = np.asarray(
        patient_ids
    )

    if len(patient_ids) != X_selected.shape[0]:
        raise ValueError(
            "The number of patient IDs must match the number "
            "of rows in `X_train`."
        )

    # --------------------------------------------------------------
    # 5. Post-LASSO Cox refit
    # --------------------------------------------------------------
    refit_outputs = global_cox_train(
        X_selected,
        y_train,
        id_list_train=patient_ids,
        learning_cox_map=learning_cox_map,
        lambda_l1_CV=refit_lambda
    )

    if len(refit_outputs) < 8:
        raise ValueError(
            "`global_cox_train` returned fewer than eight outputs."
        )

    (
        cph_refit,
        df_survival_refit,
        w_sk_refit,
        scores_refit,
        X_refit,
        y_cox_refit,
        c_index_train_refit,
        log_likelihood_refit
    ) = refit_outputs[:8]

    extra_refit_outputs = refit_outputs[8:]

    coefficients_refit = np.asarray(
        w_sk_refit,
        dtype=np.float64
    ).ravel()

    if len(coefficients_refit) != len(selected_indices):
        raise ValueError(
            "The number of refitted coefficients does not match "
            "the number of selected covariates."
        )

    # Reconstruct coefficients in the original feature space
    full_coefficients_refit = np.zeros_like(
        lasso_coefficients,
        dtype=np.float64
    )

    full_coefficients_refit[
        selected_mask
    ] = coefficients_refit

    # --------------------------------------------------------------
    # 6. Optional summary
    # --------------------------------------------------------------
    if verbose:
        print(
            "Selected covariates: "
            f"{len(selected_indices)}/{X_array.shape[1]}"
        )

        print(
            "Post-LASSO training C-index: "
            f"{c_index_train_refit:.4f}"
        )

    return {
        "model": cph_refit,
        "df_survival": df_survival_refit,
        "selected_mask": selected_mask,
        "selected_indices": selected_indices,
        "selected_feature_names": selected_feature_names,
        "X_selected": X_selected,
        "selected_coefficients": coefficients_refit,
        "full_coefficients": full_coefficients_refit,
        "scores": scores_refit,
        "X_refit": X_refit,
        "y_cox": y_cox_refit,
        "c_index_train": c_index_train_refit,
        "log_likelihood": log_likelihood_refit,
        "extra_outputs": extra_refit_outputs
    }




def jackknife_confidence_interval(scores, alpha=0.05):
    """
    Compute a confidence interval using the Jackknife resampling method.

    The Jackknife method estimates the variance of a statistic (e.g., C-index)
    by systematically leaving out one observation at a time. This function
    provides a bias-corrected point estimate and computes the corresponding
    confidence interval under normal approximation.

    Parameters
    ----------
    scores : list or np.ndarray
        List of evaluation scores (e.g., C-index values from multiple test folds).
    alpha : float, default=0.05
        Significance level for the confidence interval (e.g., 0.05 for 95% CI).

    Returns
    -------
    tuple
        (lower_bound, upper_bound) of the estimated confidence interval.

    Notes
    -----
    - The normal quantile is fixed to z = 1.96 for 95% CI (useful for small sample size).
    - The result includes a bias correction using the Jackknife estimate.
    - Assumes the score distribution is approximately symmetric.
    """
    n = len(scores)
    scores = np.array(scores)

    # Compute the full-sample mean
    mean_score = np.mean(scores)

    # Compute leave-one-out means
    jackknife_means = np.array([
        (np.sum(scores) - scores[i]) / (n - 1)
        for i in range(n)
    ])

    # Bias correction
    jackknife_mean = np.mean(jackknife_means)
    bias_corrected_mean = n * mean_score - (n - 1) * jackknife_mean

    # Jackknife variance and standard deviation
    jackknife_var = (n - 1) / n * np.sum((jackknife_means - jackknife_mean) ** 2)
    jackknife_std = np.sqrt(jackknife_var)

    # Confidence interval using normal approximation
    z = 1.96  # For 95% CI
    lower_bound = bias_corrected_mean - z * jackknife_std
    upper_bound = bias_corrected_mean + z * jackknife_std

    return lower_bound, upper_bound




