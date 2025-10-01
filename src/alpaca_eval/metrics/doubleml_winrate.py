from __future__ import annotations

from typing import Union, Sequence, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from alpaca_eval import utils
from .winrate import get_winrate

from sentence_transformers import SentenceTransformer
from doubleml import DoubleMLIRM, DoubleMLData

# --- RF defaults + merge with user params
rf_reg_defaults = dict(
    n_estimators=600,
    max_depth=None,
    min_samples_leaf=5,
    n_jobs=-1,
    random_state=42,
)
rf_clf_defaults = dict(
    n_estimators=600,
    max_depth=None,
    min_samples_leaf=5,
    # class_weight can help if treatment classes are imbalanced;
    # DoubleML tolerates either setting; enable if needed:
    # class_weight="balanced",
    n_jobs=-1,
    random_state=42,
)


def get_doubleml_length_position_controlled_winrate(
        annotations: Union[pd.DataFrame, Sequence[dict]],
        rf_reg_params: dict = rf_reg_defaults,
        rf_clf_params: dict = rf_clf_defaults,
        n_folds: int = 5,
        n_rep: int = 1,
):
    """
    Compute length+position controlled winrate via DoubleML (IRM) with RandomForest models.

    Steps:
      1) Convert input annotations to a DataFrame.
      2) If the model under test equals the baseline model, use a constant 0.5 prediction.
      3) Else: featurize -> build DoubleMLData -> fit DoubleMLIRM with RF(g) and RF(m).
      4) Construct the adjusted/orthogonalized target: y_tilde = y - g_hat(X).
      5) Aggregate metrics via _add_length_controlled_metrics(...).

    Parameters
    ----------
    annotations : pd.DataFrame | Sequence[dict]
        Raw annotations (can be a list of records). Must contain fields required by
        _get_featurized_data to produce at least: 'preference' (target) and 'model' (treatment).
    rf_reg_params : dict | None
        Extra params for RandomForestRegressor (outcome model g). Merged with sane defaults.
    rf_clf_params : dict | None
        Extra params for RandomForestClassifier (propensity model m). Merged with sane defaults.
    n_folds : int
        Number of cross-fitting folds.
    n_rep : int
        Number of cross-fitting repetitions.

    Returns
    -------
    Any
        Whatever _add_length_controlled_metrics returns in your project
        (e.g., a metrics dict/Series/DataFrame).
    """
    # Normalize input to DataFrame
    df = utils.convert_to_dataframe(annotations)

    # If this “run” is a baseline-vs-baseline comparison, use the trivial 0.5 predictor
    if _is_model_baseline(df):
        predicted_preferences = pd.Series(0.5, index=df.index)
        return _add_length_controlled_metrics(df, predicted_preferences)

    # --- Featurization & sanity checks
    featured_df = _get_featurized_data(df)

    print(featured_df)

    # required_cols = {"preference", "model"}
    # missing = required_cols - set(featured_df.columns)
    # if missing:
    #     raise KeyError(f"Featurized DataFrame is missing required columns: {sorted(missing)}")
    #
    # # Build DoubleMLData (X = all except treatment/target)
    # dml_data = _get_dml_data(
    #     featured_df,
    #     treatment_variable_name="model",
    #     target_variable_name="preference",
    # )
    #
    # ml_g = RandomForestRegressor(**rf_reg_params)
    # ml_m = RandomForestClassifier(**rf_clf_params)
    #
    # # --- Fit DoubleML IRM with cross-fitting
    # dml = DoubleMLIRM(
    #     dml_data,
    #     ml_g=ml_g,
    #     ml_m=ml_m,
    #     n_folds=n_folds,
    #     n_rep=n_rep,
    #     trimming_threshold=1e-3,
    # )
    # dml.fit(store_predictions=True)
    #
    # # Adjusted (orthogonalized) target: y_tilde = y - g_hat(X)
    # g_hat = np.asarray(dml.predictions["ml_g"]).reshape(-1)  # (n,)
    # y = featured_df["preference"].to_numpy(dtype=float)
    # y_tilde = y - g_hat
    #
    # # Keep as a Series aligned to the original DataFrame index
    # predicted_preferences = pd.Series(y_tilde, index=featured_df.index)

    # Aggregate into final metrics (your project-level helper)
    return get_winrate(annotations)


def _get_dml_data(
        featured_df: pd.DataFrame,
        treatment_variable_name: str,
        target_variable_name: str
) -> DoubleMLData:
    """
    Build a DoubleMLData object from a featured dataframe.

    Parameters
    ----------
    featured_df : pd.DataFrame
        Input DataFrame with features, treatment, and target.
    treatment_variable_name : str
        Column name of the treatment variable.
    target_variable_name : str
        Column name of the target variable.

    Returns
    -------
    DoubleMLData
        Object ready for DoubleML estimation.
    """
    if treatment_variable_name not in featured_df.columns:
        raise KeyError(f"Treatment column '{treatment_variable_name}' not found in DataFrame.")

    if target_variable_name not in featured_df.columns:
        raise KeyError(f"Target column '{target_variable_name}' not found in DataFrame.")

    treatment = featured_df[target_variable_name]
    target = featured_df[target_variable_name]

    print(featured_df[treatment_variable_name].unique())

    # take all columns except treatment and target as confounders (X)
    confounders = featured_df.drop(columns=[treatment_variable_name, target_variable_name])

    # wrap into DoubleMLData
    dml_data = DoubleMLData.from_arrays(
        x=confounders,
        y=target,
        d=treatment
    )

    return dml_data


def _add_length_controlled_metrics(annotations: pd.Union[pd.DataFrame, Sequence[dict]],
                                   predicted_preferences: pd.Series) -> dict:
    """
    Add length-controlled winrate and standard error to the metrics dictionary,
    using df_annotations to construct the predicted preferences.

    Parameters
    ----------
    annotations : Union[pd.DataFrame, Sequence[dict]]
            AlpacaEval input
    predicted_preferences : pd.Series
        Results of estimation from the DoubleML model

    Returns
    -------
    dict
        Updated dictionary with new keys:
        - "length_controlled_winrate"
        - "lc_standard_error"
    """
    metrics = dict(get_winrate(annotations))  # get the non-length controlled winrate + copy to avoid mutating input

    metrics["length_controlled_winrate"] = predicted_preferences.mean() * 100
    metrics["lc_standard_error"] = predicted_preferences.sem() * 100

    return metrics


def _is_model_baseline(df: pd.DataFrame) -> bool:
    """
    Check whether the model in column 'generator_2' is the same as
    the baseline model in column 'generator_1'.

    Assumes df["generator_2"] has only one unique value.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with columns "generator_1" and "generator_2".

    Returns
    -------
    bool
        True if the model in generator_2 is the same as the baseline model
        in generator_1, False otherwise.
    """

    uniques_gen2 = df["generator_2"].unique()

    assert len(uniques_gen2) == 1, "generator_2 must contain exactly one unique model"

    model_name = uniques_gen2[0]
    baseline_name = df["generator_1"].unique()[0]

    return model_name == baseline_name


def _get_featurized_data(df_annotations: pd.DataFrame):
    """Featurizes train set

        Parameters
        ----------
        df_annotations : pd.DataFrame
            The input dataframe, should have columns "preference", "output_1", "output_2", "index", "generator_1",
            "generator_2".
    """
    df = df_annotations.reset_index()

    # Len extraction
    len_1 = df["output_1"].str.len()
    len_2 = df["output_2"].str.len()

    std_delta_len = len_1 - len_2

    # Reinitialization
    df = df[["preference", "instruction", "generator_2"]].copy().rename(columns={"generator_2": "model"})

    # New features
    df["std_delta_len"] = np.tanh(std_delta_len / std_delta_len.std())
    df["position_component"] = _extract_position_component(df_annotations)

    # Add target variable
    df["preference"] = df["preference"].astype(float).replace({0.0: 1.5}) - 1  # easier to work with in [0,1]

    # Embed 'instruction' feature
    df = _embed_feature_to_df(df, instr_col="instruction")

    return df


def _embed_feature_to_df(
        df_annotations: pd.DataFrame,
        instr_col: str,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        prefix: str | None = None,
        drop_original: bool = True,
) -> pd.DataFrame:
    """
    Encode df[instr_col] into embeddings and append them as new columns to df.
    Returns a new DataFrame with preserved index.

    Parameters
    ----------
    df_annotations : DataFrame
        Input DataFrame.
    instr_col : str
        Name of the text column to encode.
    model_name : str
        SentenceTransformer model name (default MiniLM 384-d).
    prefix : str | None
        Prefix for new embedding column names. If None, use instr_col.
    drop_original : bool
        If True, remove the original text column after adding embeddings.

    Returns
    -------
    DataFrame
        Copy of df with additional embedding columns.
    """
    if instr_col not in df_annotations.columns:
        raise KeyError(f"Column '{instr_col}' not found in df.")

    model = SentenceTransformer(model_name)
    texts = df_annotations[instr_col].astype(str).tolist()

    emb = model.encode(texts, batch_size=512, show_progress_bar=False, normalize_embeddings=False)
    emb = np.asarray(emb)

    base = prefix or instr_col

    emb_cols = [f"{base}_emb_{i}" for i in range(emb.shape[1])]
    emb_df = pd.DataFrame(emb, index=df_annotations.index, columns=emb_cols)

    out = pd.concat([df_annotations.copy(), emb_df], axis=1)

    if drop_original:
        out = out.drop(columns=[instr_col])

    return out


def _extract_position_component(
        df_annotations: pd.DataFrame,
        mask_lower: str = "m",
        positive_case: int = 1,
        negative_case: int = 0,
        threshold: float = 1.5,
) -> np.ndarray:
    """
    Compute a ±1 position component per row based on the first generated token and the raw preference score.

    Rules (mirrors original behavior):
      - If the first token equals `mask_lower`:  preference >= threshold -> +1 else -> 0
      - Otherwise (token != mask_lower):        preference  < threshold -> +1 else -> 0

    Parameters
    ----------
    df_annotations : pd.DataFrame
        Must contain columns:
          - 'preference' (numeric, raw preference before mapping)
          - 'raw_completion' (dict-like with path ['logprobs']['content'][0]['token'])
            If absent, an array of zeros is returned.
    mask_lower : str, default "m"
        Token to check against the first generated token.
    positive_case : int, default 1
        Value used for the positive outcome.
    negative_case : int, default 0
        Value used for the negative outcome.
    threshold : float, default 1.5
        Raw preference threshold.

    Returns
    -------
    np.ndarray
        Array of shape (len(df_annotations),) with values in {positive_case, negative_case},
        or zeros if 'raw_completion' is missing.
    """

    n = len(df_annotations)
    if n == 0:
        return np.array([], dtype=int)

    # If raw completions are missing, return zeros (more usable than a scalar 0.0)
    if "raw_completion" not in df_annotations.columns or "preference" not in df_annotations.columns:
        return np.zeros(n, dtype=int)

    def _first_token(raw) -> Optional[str]:
        try:
            return raw["logprobs"]["content"][0]["token"]
        except Exception:
            return None

    # Extract first tokens (object dtype Series of str/None)
    tokens = df_annotations["raw_completion"].apply(_first_token)

    # Ensure numeric preference
    pref_raw = pd.to_numeric(df_annotations["preference"], errors="coerce")

    # Booleans for the two branches
    is_mask_lower = tokens.eq(mask_lower)
    meets_threshold = pref_raw.ge(threshold)

    # Branch logic:
    #   when is_mask_lower:   meets_threshold -> positive, else negative
    #   else:                 (~meets_threshold) -> positive, else negative
    result = np.where(
        is_mask_lower,
        np.where(meets_threshold, positive_case, negative_case),
        np.where(~meets_threshold, positive_case, negative_case),
    )

    return result.astype(int)
