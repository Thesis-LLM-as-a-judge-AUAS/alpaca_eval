from __future__ import annotations

import logging
import warnings
from typing import Union, Sequence, Optional

import numpy as np
import pandas as pd
from doubleml import DoubleMLAPOS, DoubleMLData
from huggingface_hub import hf_hub_download
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import make_scorer, log_loss, mean_squared_error
from sklearn.model_selection import GridSearchCV

from alpaca_eval import utils, constants
from .winrate import get_winrate

# Suppress sklearn deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", message=".*force_all_finite.*", category=FutureWarning)

# Suppress DoubleML warnings
warnings.filterwarnings("ignore", message=".*Propensity predictions.*are close to zero or one.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*The proportion of observations with treatment level.*is less than 5%.*", category=UserWarning)

# Default parameter grids for cross-validation
gb_reg_param_grid = {
    'n_estimators': [100, 150, 200],  # Reduced from 4 to 3
    'learning_rate': [0.05, 0.1, 0.15],  # Reduced from 5 to 3
    'max_depth': [3, 4, 5],  # Reduced from 5 to 3
    'min_samples_split': [10, 20],  # Reduced from 4 to 2
    'min_samples_leaf': [5, 10],  # Reduced from 4 to 2
    'subsample': [0.8, 0.9, 1.0]  # Reduced from 4 to 3
}

gb_reg_base_params = dict(
    loss='squared_error',
    random_state=42
)

logreg_cv_params = dict(
    penalty="l2",
    solver="lbfgs",
    max_iter=1000,
    n_jobs=None,
    random_state=42
)


def get_doubleml_length_position_controlled_winrate(
        annotations: Union[pd.DataFrame, Sequence[dict]],
        rf_reg_params=None,
        rf_clf_params=None,
        n_folds: int = 5,
        n_rep: int = 1,
        cv_folds: int = 5,
):
    """
    Compute length+position controlled winrate via DoubleML (IRM/APOS) with strong models.
    
    This function uses Double Machine Learning to estimate average potential outcomes,
    controlling for length and position bias in preference annotations.
    
    The function uses cross-validation for hyperparameter tuning:
    - GradientBoostingRegressor: GridSearchCV with MSE (mean squared error) metric
    - LogisticRegression: LogisticRegressionCV with log_loss metric
    
    Steps:
    1. Convert input annotations to a DataFrame
    2. Featurize the data (extract length, position, and instruction difficulty features)
    3. Initialize ML models with cross-validation (GridSearchCV for regression, LogisticRegressionCV for classification)
    4. Setup and fit DoubleMLAPOS model
    5. Extract nuisance function predictions
    6. Compute orthogonalized target: y_tilde = y - g_hat(X)
    7. Handle special cases where generator_1 == model (set to 0.5)
    8. Aggregate final metrics
    
    Parameters
    ----------
    annotations : Union[pd.DataFrame, Sequence[dict]]
        Input annotations containing preference data, outputs, and metadata.
    rf_reg_params : dict, optional
        Parameters for the regression model (outcome function g).
        If None, uses GridSearchCV with default parameter grid.
        If dict provided, used as base parameters or to customize the search grid.
    rf_clf_params : dict, optional
        Parameters for the classification model (propensity function m).
        If None, uses default LogisticRegressionCV parameters with log_loss metric.
    n_folds : int, default 5
        Number of folds for cross-fitting in DoubleML.
    n_rep : int, default 1
        Number of repetitions for sample splitting in DoubleML.
    cv_folds : int, default 5
        Number of folds for cross-validation hyperparameter tuning.
    
    Returns
    -------
    list[dict]
        List of dictionaries containing winrate metrics for each model including:
        - length_controlled_winrate: The main metric of interest
        - lc_standard_error: Standard error of the length-controlled winrate
        - win_rate: Standard winrate
        - standard_error: Standard error of winrate
        - model: Model name
    
    Raises
    ------
    KeyError
        If required columns are missing after featurization.
    AttributeError
        If predictions cannot be extracted from the DoubleML model.
    ValueError
        If predictions are invalid or None.
    """
    logging.info("Starting DoubleML length+position controlled winrate computation (GradientBoosting + LogisticRegression).")
    logging.debug(f"Input type: {type(annotations)}, n_folds={n_folds}, n_rep={n_rep}")

    # Convert input to DataFrame
    try:
        logging.info("Converting annotations to DataFrame.")
        df = utils.convert_to_dataframe(annotations)
        logging.debug(f"Converted DataFrame shape: {df.shape}")
    except Exception:
        logging.exception("Failed to convert annotations to DataFrame.")
        raise

    # Feature extraction
    try:
        logging.info("Featurizing data.")
        featured_df = _get_featurized_data(df)
    except Exception:
        logging.exception("Feature extraction failed.")
        raise

    # Initialize ML models
    try:
        ml_g, ml_m = _initialize_ml_models(rf_reg_params, rf_clf_params, cv_folds)
    except Exception:
        logging.exception("Failed to initialize ML models.")
        raise

    # Setup and fit DoubleML model
    try:
        dml = _setup_and_fit_dml_model(featured_df, ml_g, ml_m, n_folds, n_rep)
    except Exception:
        logging.exception("DoubleML model training failed.")
        raise

    # Extract predictions
    try:
        g_hat = _extract_predictions_from_dml(dml, featured_df)
    except Exception:
        logging.exception("Failed to extract predictions from DoubleML model.")
        raise

    # Compute orthogonalized target
    try:
        predicted_preferences = _compute_orthogonalized_target(featured_df, g_hat)
    except Exception:
        logging.exception("Failed during orthogonalization or preference adjustment.")
        raise

    # Aggregate final metrics
    try:
        logging.info("Aggregating final metrics with _add_length_controlled_metrics.")
        metrics_list = _add_length_controlled_metrics(annotations, predicted_preferences)
        logging.info(f"Metrics computation completed successfully. Found {len(metrics_list)} models.")
        return metrics_list
    except Exception:
        logging.exception("Failed to compute final metrics.")
        raise

def _initialize_ml_models(rf_reg_params=None, rf_clf_params=None, cv_folds=5):
    """
    Initialize machine learning models for DoubleML estimation with cross-validation.
    
    Parameters
    ----------
    rf_reg_params : dict, optional
        Parameters for the regression model (outcome function g). 
        If None, uses GridSearchCV with default parameter grid.
        If dict provided, used as base parameters or to customize the search grid.
    rf_clf_params : dict, optional
        Parameters for the classification model (propensity function m).
        If None, uses default LogisticRegressionCV parameters.
    cv_folds : int, default 5
        Number of folds for cross-validation.
    
    Returns
    -------
    tuple
        A tuple containing (ml_g, ml_m) where:
        - ml_g: GridSearchCV instance for outcome function
        - ml_m: LogisticRegressionCV instance for propensity function
    """
    # Initialize LogisticRegressionCV with log_loss metric
    if rf_clf_params is None:
        rf_clf_params = logreg_cv_params.copy()
    
    # Create scorer for log_loss
    log_loss_scorer = make_scorer(
        log_loss, 
        greater_is_better=False, 
        response_method='predict_proba'
    )
    
    # Add CV parameters
    rf_clf_params_cv = rf_clf_params.copy()
    rf_clf_params_cv['cv'] = cv_folds
    rf_clf_params_cv['scoring'] = log_loss_scorer
    
    ml_m = LogisticRegressionCV(**rf_clf_params_cv)
    logging.debug(f"Using LogisticRegressionCV with {cv_folds} folds and log_loss metric.")
    
    # Initialize GradientBoostingRegressor with GridSearchCV and MSE metric
    # Always use default parameter grid
    param_grid = gb_reg_param_grid.copy()
    base_params = gb_reg_base_params.copy()
    
    # If rf_reg_params provided, add parameters not in param_grid to base_params
    if rf_reg_params is not None:
        for key, value in rf_reg_params.items():
            if key not in gb_reg_param_grid.keys():
                base_params[key] = value
    
    # Create base model
    base_model = GradientBoostingRegressor(**base_params)
    
    # Create MSE scorer
    mse_scorer = make_scorer(
        mean_squared_error,
        greater_is_better=False
    )
    
    # Create GridSearchCV with verbose output
    # verbose=1: prints progress for each parameter combination
    # verbose=2: prints detailed info including score for each combination
    ml_g = GridSearchCV(
        base_model,
        param_grid,
        cv=cv_folds,
        scoring=mse_scorer,
        n_jobs=-1,
        verbose=2  # Enable detailed verbose output to see training progress and scores
    )
    # Calculate total number of parameter combinations
    total_combinations = 1
    for param_values in param_grid.values():
        total_combinations *= len(param_values)
    total_fits = total_combinations * cv_folds
    
    logging.info(f"GridSearchCV configuration:")
    logging.info(f"  - Parameter combinations: {total_combinations}")
    logging.info(f"  - CV folds: {cv_folds}")
    logging.info(f"  - Total fits per GridSearchCV: {total_fits}")
    logging.info(f"  - Verbose=2 mode enabled - you will see detailed training progress with scores for each parameter combination.")
    
    logging.info("Initialized ML models: GridSearchCV (g) with MSE and LogisticRegressionCV (m) with log_loss")
    return ml_g, ml_m


def _setup_and_fit_dml_model(featured_df, ml_g, ml_m, n_folds=5, n_rep=1):
    """
    Setup and fit a DoubleMLAPOS model with the provided ML models.
    
    Parameters
    ----------
    featured_df : pd.DataFrame
        DataFrame with featurized data containing treatment, target, and confounders.
    ml_g : sklearn.base.BaseEstimator or GridSearchCV
        Machine learning model for the outcome function g(X).
        Can be GradientBoostingRegressor or GridSearchCV wrapper.
    ml_m : LogisticRegressionCV
        Machine learning model for the propensity function m(X).
        Uses LogisticRegressionCV with cross-validation.
    n_folds : int, default 5
        Number of folds for cross-fitting.
    n_rep : int, default 1
        Number of repetitions for sample splitting.
    
    Returns
    -------
    DoubleMLAPOS
        Fitted DoubleMLAPOS model.
    """
    logging.info("Building DoubleMLData object.")
    dml_data = _get_dml_data(
        featured_df,
        treatment_variable_name="model",
        target_variable_name="preference",
    )
    
    treatment_levels = np.unique(dml_data.d).tolist()
    logging.debug(f"Detected treatment levels: {treatment_levels}")
    
    logging.info("Initializing DoubleMLAPOS estimator.")
    dml = DoubleMLAPOS(
        obj_dml_data=dml_data,
        ml_g=ml_g,
        ml_m=ml_m,
        treatment_levels=treatment_levels,
        n_folds=n_folds,
        n_rep=n_rep,
        score='APO',
        normalize_ipw=True,
        trimming_rule='truncate',
        trimming_threshold=0.01,  # Increased threshold for better stability
        draw_sample_splitting=True
    )
    
    logging.info("Fitting DoubleML model.")
    logging.info("=" * 80)
    logging.info("Training will show sklearn verbose output:")
    logging.info("  - GridSearchCV: progress for each parameter combination")
    logging.info("  - This will be repeated for each fold in cross-fitting (n_folds={}, n_rep={})".format(n_folds, n_rep))
    logging.info("=" * 80)
    
    # Redirect sklearn verbose output to logging
    # sklearn's verbose prints to stdout, so we capture it and redirect to logging
    from contextlib import redirect_stdout
    
    # Create a simple stream that redirects to logging
    class SklearnLogStream:
        def __init__(self):
            self.buffer = []
        
        def write(self, s):
            if s and s.strip():  # Only log non-empty content
                # Remove trailing newlines and log
                for line in s.rstrip().split('\n'):
                    if line.strip():
                        logging.info(f"[sklearn] {line}")
            return len(s)
        
        def flush(self):
            pass
    
    # Fit with verbose output captured and logged
    sklearn_log_stream = SklearnLogStream()
    with redirect_stdout(sklearn_log_stream):
        dml.fit(store_predictions=True)
    
    logging.info("=" * 80)
    logging.info("DoubleML fitting completed successfully.")
    
    return dml


def _extract_predictions_from_dml(dml, featured_df):
    """
    Extract nuisance function predictions from a fitted DoubleML model.
    
    This function tries multiple approaches to access predictions from the DoubleML model,
    handling different versions and storage mechanisms.
    
    Parameters
    ----------
    dml : DoubleMLAPOS
        Fitted DoubleMLAPOS model.
    featured_df : pd.DataFrame
        DataFrame with featurized data for making predictions if needed.
    
    Returns
    -------
    np.ndarray
        Predictions from the outcome function g(X).
    
    Raises
    ------
    AttributeError
        If predictions cannot be found in any expected location.
    ValueError
        If predictions are None or invalid.
    """
    logging.info("Extracting predictions from DoubleML model.")
    
    g_hat = None
    
    # Try to access predictions from the standard predictions attribute
    if hasattr(dml, 'predictions'):
        predictions = dml.predictions
        logging.debug(f"Found predictions on dml object: {type(predictions)}")
        g_hat = np.asarray(predictions["ml_g"]).reshape(-1)
    
    # Try to access predictions from the framework attribute
    elif hasattr(dml, 'framework') and hasattr(dml.framework, 'predictions'):
        predictions = dml.framework.predictions
        logging.debug(f"Found predictions on framework object: {type(predictions)}")
        g_hat = np.asarray(predictions["ml_g"]).reshape(-1)
    
    # Try to access predictions through modellist (manual prediction)
    elif hasattr(dml, 'modellist') and len(dml.modellist) > 0:
        logging.debug("Attempting to access predictions through modellist")
        logging.debug(f"Modellist length: {len(dml.modellist)}")
        
        # Get the confounders (X) from the original data
        confounders = featured_df.drop(columns=["preference", "model", "generator_1"])
        logging.debug(f"Confounders shape: {confounders.shape}")

        # Use stored predictions from modellist[0] directly
        predictions = dml.modellist[0].predictions
        logging.debug(f"Predictions type: {type(predictions)}")
        
        if isinstance(predictions, dict):
            # For DoubleMLAPOS, we need to combine predictions from all treatment levels
            # ml_g_d_lvl0 and ml_g_d_lvl1 are predictions for different treatment levels
            g_hat_combined = []
            for key in predictions.keys():
                if key.startswith('ml_g_d_lvl'):
                    g_hat_combined.append(predictions[key])
            
            if g_hat_combined:
                # Average across treatment levels to get overall outcome predictions
                g_hat = np.mean(g_hat_combined, axis=0).reshape(-1)
                logging.debug(f"Successfully got combined predictions from {len(g_hat_combined)} treatment levels: shape {g_hat.shape}")
            else:
                raise KeyError("No ml_g_d_lvl* keys found in predictions")
        else:
            g_hat = np.asarray(predictions).reshape(-1)
            logging.debug(f"Successfully got predictions from modellist[0].predictions: shape {g_hat.shape}")
    
    else:
        raise AttributeError("Cannot find predictions attribute in DoubleMLAPOS object, its framework, or modellist")
    
    if g_hat is None:
        raise ValueError("Failed to obtain g_hat predictions")
    
    logging.info(f"Successfully extracted predictions with shape: {g_hat.shape}")
    return g_hat


def _compute_orthogonalized_target(featured_df, g_hat):
    """
    Compute the orthogonalized target y_tilde = y - g_hat(X).
    
    This function also handles special cases where generator_1 == model by setting
    the predicted preference to 0.5.
    
    Parameters
    ----------
    featured_df : pd.DataFrame
        DataFrame with featurized data containing the target variable.
    g_hat : np.ndarray
        Predictions from the outcome function g(X).
    
    Returns
    -------
    pd.Series
        Orthogonalized target values with same-generator cases set to 0.5.
    """
    logging.info("Computing orthogonalized target (y_tilde).")
    
    y = featured_df["preference"].to_numpy(dtype=float)
    y_tilde = y - g_hat
    
    predicted_preferences = pd.Series(y_tilde, index=featured_df.index)
    
    # Handle special case where generator_1 == model
    same_generator_mask = featured_df["generator_1"] == featured_df["model"]
    predicted_preferences[same_generator_mask] = 0.5
    
    logging.debug(f"Number of same-generator cases set to 0.5: {same_generator_mask.sum()}")
    logging.info(f"Computed orthogonalized target with {len(predicted_preferences)} values")
    
    return predicted_preferences





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

    treatment = _encode_treatment(featured_df[treatment_variable_name])
    target = featured_df[target_variable_name]

    # take all columns except treatment, target, and generator_1 as confounders (X)
    confounders = featured_df.drop(columns=[treatment_variable_name, target_variable_name, "generator_1"])

    # wrap into DoubleMLData
    dml_data = DoubleMLData.from_arrays(
        x=confounders,
        y=target,
        d=treatment
    )

    return dml_data


def _encode_treatment(treatment_series: pd.Series):
    """
    Encodes the treatment column into numeric values if it contains strings.

    Parameters
    ----------
    treatment_series : pd.Series
        A pandas Series representing the treatment variable.
        It may contain string or numeric values.

    Returns
    -------
    pd.Series or np.ndarray
        If the column is of type 'object' (string), categorical codes are returned.
        If the column is already numeric, it is returned unchanged.
    """
    if treatment_series.dtype == 'object':
        return pd.Categorical(treatment_series).codes

    return treatment_series


def _add_length_controlled_metrics(annotations: pd.Union[pd.DataFrame, Sequence[dict]],
                                   predicted_preferences: pd.Series) -> list[dict]:
    """
    Add length-controlled winrate and standard error to the metrics dictionary,
    using df_annotations to construct the predicted preferences.
    
    This function now returns a list of metrics for each model separately.

    Parameters
    ----------
    annotations : Union[pd.DataFrame, Sequence[dict]]
            AlpacaEval input
    predicted_preferences : pd.Series
        Results of estimation from the DoubleML model

    Returns
    -------
    list[dict]
        List of dictionaries with metrics for each model:
        - "length_controlled_winrate"
        - "lc_standard_error"
        - "win_rate" 
        - "standard_error"
    """
    df = utils.convert_to_dataframe(annotations)
    
    # Get unique models
    models = df['generator_2'].unique()
    
    metrics_list = []
    
    for model in models:
        # Filter annotations for this model
        model_mask = df['generator_2'] == model
        model_annotations = df[model_mask]
        model_preferences = predicted_preferences[model_mask]
        
        # Get basic winrate metrics for this model
        model_metrics = dict(get_winrate(model_annotations))
        
        # Add length-controlled metrics
        model_metrics["length_controlled_winrate"] = model_preferences.mean() * 100
        model_metrics["lc_standard_error"] = model_preferences.sem() * 100
        
        # Add model name
        model_metrics["model"] = model
        
        metrics_list.append(model_metrics)
    
    return metrics_list


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
    df = df[["preference", "generator_1", "generator_2", "index"]].copy().rename(columns={"generator_2": "model"})

    # New features
    df["std_delta_len"] = np.tanh(std_delta_len / std_delta_len.std())
    df["position_component"] = _extract_position_component(df_annotations)

    # Add target variable
    df["preference"] = df["preference"].astype(float).replace({0.0: 1.5}) - 1  # easier to work with in [0,1]

    # Embed 'instruction' feature
    df["instruction_difficulty"] = _get_instruction_difficulty(df)

    return df.dropna()


def _get_instruction_difficulty(df_annotations):
    out = hf_hub_download(
        repo_id="tatsu-lab/alpaca_eval",
        filename="df_gamed.csv",
        repo_type="dataset",
        token=constants.DATASETS_TOKEN,
        force_download=constants.DATASETS_FORCE_DOWNLOAD,
        cache_dir=constants.DEFAULT_CACHE_DIR,
    )

    df_gamed_out = pd.read_csv(out)

    instruction_difficulty = df_gamed_out.drop(columns=["model"]).drop_duplicates("index")["instruction_difficulty"]

    return df_annotations["index"].transform(lambda g: instruction_difficulty[g % len(instruction_difficulty)])


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
