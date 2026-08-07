# -----------------------------
# 1. Load CSV files
# -----------------------------
patients = pd.read_csv("./PATIENTS_sorted.csv")
notes = pd.read_csv("./NOTEEVENTS_sorted.csv")
diagnoses = pd.read_csv("./DIAGNOSES_ICD_sorted.csv")
# Load ICD-9 dictionary (mapping pour ID des diagnoses avec un nom associé)
icd_dict = pd.read_csv("./D_ICD_DIAGNOSES.csv")


# -----------------------------
# 2. Keep only discharge summaries (clinical notes)
# -----------------------------
notes_ds = notes[notes["CATEGORY"] == "Discharge summary"]

# -----------------------------
# 3. Keep only required columns
# -----------------------------
notes_ds = notes_ds[["SUBJECT_ID", "HADM_ID", "TEXT"]]
diagnoses = diagnoses[["SUBJECT_ID", "HADM_ID", "ICD9_CODE"]]
patients = patients[["SUBJECT_ID"]]

# -----------------------------
# 4. Join notes with diagnoses (per admission)
# -----------------------------
df = notes_ds.merge(
    diagnoses,
    on=["SUBJECT_ID", "HADM_ID"],
    how="inner"
)

# -----------------------------
# 5. (Optional) Ensure patient-level consistency
# -----------------------------
df = df.merge(
    patients,
    on="SUBJECT_ID",
    how="inner"
)

# -----------------------------
# 6. Final dataframe
# -----------------------------
df_final = df.rename(columns={
    "SUBJECT_ID": "patient_id",
    "TEXT": "clinical_notes",
    "ICD9_CODE": "diagnosis"
})

# Keep useful columns
icd_dict = icd_dict[["ICD9_CODE", "SHORT_TITLE", "LONG_TITLE"]]

# Merge with main dataframe
df_final = df_final.merge(
    icd_dict,
    left_on="diagnosis",
    right_on="ICD9_CODE",
    how="left"
)

# Optional cleanup
df_final = df_final.drop(columns=["ICD9_CODE"])

print("Final dataframe shape:", df_final.shape)

df = df_final[['patient_id', 'clinical_notes', 'SHORT_TITLE']]



###### VERIFIER LE MAPPING non deterministe

mapping_check = (
    df
    .groupby('clinical_notes')['SHORT_TITLE']
    .nunique()
    .value_counts()
    .sort_index()
)

print("=== Nombre de diagnostics distincts par note clinique ===")
for n_labels, count in mapping_check.items():
    print(f"Notes associées à {n_labels} diagnostic(s) : {count}")