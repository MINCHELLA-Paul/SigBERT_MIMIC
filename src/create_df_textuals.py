import pandas as pd
import numpy as np


def load_patients(patients_path: str) -> pd.DataFrame:
    """
    Load the PATIENTS table and construct survival information.

    Parameters
    ----------
    patients_path : str
        Path to PATIENTS_sorted.csv.

    Returns
    -------
    pd.DataFrame
        DataFrame containing:
        - SUBJECT_ID
        - event (1 if dead, 0 otherwise)
        - death_time (datetime)
    """

    patients = pd.read_csv(patients_path)

    # Convert date columns
    patients["DOD"] = pd.to_datetime(patients["DOD"], errors="coerce")

    # Event indicator
    patients["event"] = (~patients["DOD"].isna()).astype(int)

    patients_surv = patients[["SUBJECT_ID", "event", "DOD"]].rename(
        columns={"DOD": "death_time"}
    )

    print(f"Loaded {len(patients_surv)} patients")

    return patients_surv


def load_notes(notes_path: str) -> pd.DataFrame:
    """
    Load clinical notes and keep relevant columns.

    Parameters
    ----------
    notes_path : str
        Path to NOTEEVENTS_sorted.csv.

    Returns
    -------
    pd.DataFrame
        DataFrame containing:
        - SUBJECT_ID
        - note_time
        - TEXT
    """

    notes = pd.read_csv(notes_path)

    # Convert time column
    notes["CHARTDATE"] = pd.to_datetime(notes["CHARTDATE"], errors="coerce")

    notes_clean = notes[["SUBJECT_ID", "CHARTDATE", "TEXT"]].rename(
        columns={"CHARTDATE": "note_time"}
    )

    notes_clean = notes_clean.dropna(subset=["note_time", "TEXT"])

    print(f"Loaded {len(notes_clean)} notes")

    return notes_clean


def merge_notes_patients(notes: pd.DataFrame, patients: pd.DataFrame) -> pd.DataFrame:
    """
    Merge clinical notes with survival outcomes.

    Parameters
    ----------
    notes : pd.DataFrame
        Notes DataFrame.

    patients : pd.DataFrame
        Patients survival DataFrame.

    Returns
    -------
    pd.DataFrame
        Longitudinal dataset for survival modeling.
    """

    df = notes.merge(patients, on="SUBJECT_ID", how="inner")

    print(f"Merged dataset size: {len(df)} rows")

    return df


def compute_followup_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute follow-up time relative to the first note of each patient.

    Parameters
    ----------
    df : pd.DataFrame
        Merged dataset.

    Returns
    -------
    pd.DataFrame
        Dataset with:
        - note_time
        - followup_time
        - death_time
    """

    # First note per patient
    first_note = df.groupby("SUBJECT_ID")["note_time"].transform("min")

    df["time_since_first_note"] = (df["note_time"] - first_note).dt.days

    # Survival time
    df["survival_time"] = (df["death_time"] - first_note).dt.days

    return df


def sort_notes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort notes chronologically for each patient.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """

    df = df.sort_values(["SUBJECT_ID", "note_time"]).reset_index(drop=True)

    return df


def build_survival_notes_dataset(
    patients_path: str,
    notes_path: str
) -> pd.DataFrame:
    """
    Complete pipeline constructing the survival dataset from MIMIC.

    Parameters
    ----------
    patients_path : str
    notes_path : str

    Returns
    -------
    pd.DataFrame
        Final dataset used for survival modeling.
    """

    print("Loading patients...")
    patients = load_patients(patients_path)

    print("Loading notes...")
    notes = load_notes(notes_path)

    print("Merging datasets...")
    df = merge_notes_patients(notes, patients)

    print("Computing follow-up time...")
    df = compute_followup_time(df)

    print("Sorting notes...")
    df = sort_notes(df)

    print("Dataset construction completed")

    return df


def compute_followup_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute follow-up time and survival target Y = min(T, C).

    Time origin is defined as the first note date for each patient.

    Parameters
    ----------
    df : pd.DataFrame
        Merged dataset containing note timestamps and patient survival data.

    Returns
    -------
    pd.DataFrame
        Dataset with:
        - time_since_first_note
        - survival_time (Y = min(T, C))
        - event indicator
    """

    # First note per patient (time origin)
    first_note = df.groupby("SUBJECT_ID")["note_time"].transform("min")

    # Last note per patient (censoring time)
    last_note = df.groupby("SUBJECT_ID")["note_time"].transform("max")

    # Time since origin for each note
    df["time_since_first_note"] = (df["note_time"] - first_note).dt.days

    # Observed time Y = min(T, C)
    observed_time = np.where(
        df["event"] == 1,
        (df["death_time"] - first_note).dt.days,
        (last_note - first_note).dt.days
    )

    df["survival_time"] = observed_time

    return df


patients_path = "PATIENTS_sorted.csv"
notes_path = "NOTEEVENTS_sorted.csv"

df_model = build_survival_notes_dataset(
    patients_path,
    notes_path
)

print(df_model.head())

df_model.to_csv("mimic_survival_notes_dataset.csv", index=False); print("Saved dataset to mimic_survival_notes_dataset.csv")