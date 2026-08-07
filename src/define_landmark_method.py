def define_landmark_cohort(df: pd.DataFrame,
                           landmark_months: int,
                           var_time: str = 'date_creation',
                           var_id: str = 'ID',
                           var_T: str = 'T_days',
                           var_since_start: str = 'time_since_first_note',
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
        (df_L["DEATH"] == 1)
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
