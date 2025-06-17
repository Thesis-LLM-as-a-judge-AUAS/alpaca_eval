import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import make_scorer
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
sklearn.set_config(enable_metadata_routing=True)


def logloss(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    all_logloss = y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
    return -np.mean(all_logloss)



class LengthControlledAlpacaEval:
    """
    Implementation of Length-Controlled AlpacaEval as described in the paper.

    This class implements the regression-based approach to debias AlpacaEval
    by controlling for length effects while maintaining interpretability.
    """

    def __init__(self, l2_reg=0.01, cv_folds=5):
        """
        Initialize the Length-Controlled AlpacaEval evaluator.

        Args:
            l2_reg: L2 regularization strength for the length coefficient
            cv_folds: Number of cross-validation folds
        """
        self.l2_reg = l2_reg
        self.cv_folds = cv_folds
        self.instruction_difficulties = {}
        self.model = None
        self.length_scaler = StandardScaler()
        self.fitted = False

    def _prepare_test_data(self, data):
        data = data.copy()
        data['length_feature'] = np.zeros(len(data))

        return data


    def _prepare_data(self, data):
        """
        Prepare the data for regression analysis.

        Args:
            data: DataFrame with columns ['instruction', 'model_response', 'baseline_response',
                  'model_name', 'baseline_name', 'preference']

        Returns:
            Prepared features and labels
        """
        # Calculate response lengths
        data = data.copy()
        data['model_length'] = data['model_response'].str.len()
        data['baseline_length'] = data['baseline_response'].str.len()
        data['length_diff'] = data['model_length'] - data['baseline_length']

        # Standardize length differences
        length_diff_scaled = self.length_scaler.fit_transform(
            data['length_diff'].values.reshape(-1, 1)
        ).flatten()

        # Apply tanh transformation as per paper
        data['length_feature'] = np.tanh(length_diff_scaled)

        return data

    def _fit_instruction_difficulties(self, data):
        """
        Step 1: Fit a joint regression to estimate instruction difficulties.

        This step estimates γ_x (instruction difficulty) by fitting a model
        with ψ_m - ψ_b fixed to 1 for all models.

        Args:
            data: Prepared data DataFrame
        """
        print("Step 1: Estimating instruction difficulties...")

        # Create design matrix for instruction difficulties
        instructions = data['instruction'].unique()
        n_instructions = len(instructions)
        n_samples = len(data)

        # Create instruction dummy variables
        instruction_matrix = np.zeros((n_samples, n_instructions))
        instruction_to_idx = {inst: idx for idx, inst in enumerate(instructions)}

        for i, inst in enumerate(data['instruction']):
            instruction_matrix[i, instruction_to_idx[inst]] = 1

        # Combine features: model effects + instruction effects
        X = np.hstack([instruction_matrix])
        y = data['preference'].values

        # Fit logistic regression
        lr = LogisticRegression(fit_intercept=False, max_iter=1000, C=1 / self.l2_reg)
        lr.fit(X, y)

        # Extract instruction difficulties (last n_instructions coefficients)
        instruction_coefs = lr.coef_[0][-n_instructions:]

        # Store instruction difficulties
        for inst, coef in zip(instructions, instruction_coefs):
            self.instruction_difficulties[inst] = coef

        print(f"Estimated difficulties for {len(instructions)} instructions")

    def _extract_features(self, model_data):
        # Prepare features for this model
        n_samples = len(model_data)

        # Feature 1: Model identity (always 1 for evaluated model vs baseline)
        model_feature = np.ones(n_samples)

        # Feature 2: Length feature (tanh of normalized length difference)
        length_feature = model_data['length_feature'].values

        # Feature 3: Instruction difficulty * model instruction coefficient
        # Using pre-computed instruction difficulties
        instruction_feature = np.array([
            self.instruction_difficulties.get(inst, 0)
            for inst in model_data['instruction']
        ])

        return np.column_stack([model_feature, length_feature, instruction_feature])

    def _fit_model_coefficients(self, data):
        """
        Step 2: Fit individual model coefficients (θ_m, φ_m,b, ψ_m).

        For each model, fit coefficients while using pre-computed instruction
        difficulties from step 1.

        Args:
            data: Prepared data DataFrame
        """
        print("Step 2: Fitting individual model coefficients...")

        models = data['model_name'].unique()

        for model in models:
            model_data = data[data['model_name'] == model].copy()

            if len(model_data) == 0:
                continue

            # Combine features
            X = self._extract_features(model_data)
            y = model_data['preference'].values
            # Apply cross-validation to avoid overfitting
            custom_scorer = make_scorer(
                logloss,
                response_method="predict_proba",
                greater_is_better=False
            )

            if self.cv_folds > 1:
                cv = KFold(n_splits=self.cv_folds)

                self.model = LogisticRegressionCV(
                    fit_intercept=False,
                    cv=cv,
                    scoring=custom_scorer,
                    max_iter=1000,
                    solver='liblinear',
                )
            else:
                self.model = LogisticRegression(
                    fit_intercept=False,
                    max_iter=1000,
                    solver='liblinear',
                )

            print(y)
            self.model.fit(X, y)

        print(f"Fitted models for {len(models)} models")

    def _test_model_coefficients(self, data):
        duplicated_data = data.copy()

        X = self._extract_features(duplicated_data)
        results = list(map(lambda x: x[0], self.model.predict_proba(X)))

        duplicated_data['predicted_result'] = results

        return duplicated_data

    def fit(self, data):
        """
        Fit the Length-Controlled AlpacaEval model.

        Args:
            data: DataFrame with columns:
                - 'instruction': The instruction/prompt
                - 'model_response': Response from the evaluated model
                - 'baseline_response': Response from the baseline model
                - 'model_name': Name of the evaluated model
                - 'baseline_name': Name of the baseline model (usually consistent)
                - 'preference': Binary preference (1 if model preferred, 0 if baseline preferred)
        """
        # Prepare data
        data = self._prepare_data(data)

        # Step 1: Estimate instruction difficulties
        self._fit_instruction_difficulties(data)

        # Step 2: Fit individual model coefficients
        self._fit_model_coefficients(data)

        self.fitted = True
        print("Length-Controlled AlpacaEval model fitted successfully!")

    def test(self, data, name):
        """
            Test the Length-Controlled AlpacaEval model.

            Args:
                data: DataFrame with columns:
                    - 'instruction': The instruction/prompt
                    - 'model_response': Response from the evaluated model
                    - 'baseline_response': Response from the baseline model
                    - 'model_name': Name of the evaluated model
                    - 'baseline_name': Name of the baseline model (usually consistent)
                    - 'preference': Binary preference (1 if model preferred, 0 if baseline preferred)
        """
        data = self._prepare_test_data(data)
        result = self._test_model_coefficients(data)
        result.to_json(f'./fair-eval-test/alpaca-result-{name}.json', orient='records')

        print(f"Results for {name} are saved!")


# Demonstration of the implementation
if __name__ == "__main__":
    print("Download GPT-3.5 + Auto-J dataset...")
    gpt_3_5_data = pd.read_json("./fair-eval-test/alpaca-auto-j.json")

    print("\nInitializing Length-Controlled AlpacaEval with GPT-3.5 + Auto-J...")
    lc_eval = LengthControlledAlpacaEval(l2_reg=0.01, cv_folds=5)

    print("\nFitting Length-Controlled AlpacaEval model with GPT-3.5 + Auto-J...")
    lc_eval.fit(gpt_3_5_data)

    print("\nTesting Length-Controlled AlpacaEval model with GPT-3.5 + Auto-J...")
    lc_eval.test(gpt_3_5_data, name="gpt-3.5")

    print("Download GPT-4 + Auto-J dataset...")
    gpt_4_data = pd.read_json("./fair-eval-test/alpaca-auto-j-gpt4.json")

    print("\nInitializing Length-Controlled AlpacaEval with GPT-4 + Auto-J...")
    lc_eval_gpt4 = LengthControlledAlpacaEval(l2_reg=0.01, cv_folds=5)

    print("\nFitting Length-Controlled AlpacaEval model with GPT-4 + Auto-J...")
    lc_eval_gpt4.fit(gpt_4_data)

    print("\nTesting Length-Controlled AlpacaEval model with GPT-4 + Auto-J...")
    lc_eval_gpt4.test(gpt_4_data, name="gpt-4")

    print("Download GPT-3.5 + Judgelm dataset...")
    gpt_35_judgelm_data = pd.read_json("./fair-eval-test/alpaca-judgelm.json")

    print("\nInitializing Length-Controlled AlpacaEval with GPT-3.5 + JudgeLM...")
    lc_eval_gpt35_judgelm = LengthControlledAlpacaEval(l2_reg=0.01, cv_folds=5)

    print("\nFitting Length-Controlled AlpacaEval model with GPT-3.5 + JudgeLM...")
    lc_eval_gpt35_judgelm.fit(gpt_35_judgelm_data)

    print("\nTesting Length-Controlled AlpacaEval model with GPT-3.5 + JudgeLM...")
    lc_eval_gpt35_judgelm.test(gpt_35_judgelm_data, name="gpt-3-5-judgelm")

    print("Download GPT-4 + Judgelm dataset...")
    gpt_4_judgelm_data = pd.read_json("./fair-eval-test/alpaca-judgelm-gpt4.json")

    print("\nInitializing Length-Controlled AlpacaEval with GPT-3.5 + JudgeLM...")
    lc_eval_gpt4_judgelm = LengthControlledAlpacaEval(l2_reg=0.01, cv_folds=5)

    print("\nFitting Length-Controlled AlpacaEval model with GPT-3.5 + JudgeLM...")
    lc_eval_gpt4_judgelm.fit(gpt_4_judgelm_data)

    print("\nTesting Length-Controlled AlpacaEval model with GPT-3.5 + JudgeLM...")
    lc_eval_gpt4_judgelm.test(gpt_4_judgelm_data, name="gpt-4-judgelm")

    print("Download Verbosity dataset...")
    verbosity_data = pd.read_json("./fair-eval-test/final_verbs.json")
    balanced = pd.read_json("./fair-eval-test/balanced.json")

    print("\nInitializing Length-Controlled AlpacaEval with Verbosity...")
    lc_eval_verbosity_data = LengthControlledAlpacaEval(l2_reg=0.01, cv_folds=1)

    print("\nFitting Length-Controlled AlpacaEval model with Verbosity...")
    lc_eval_verbosity_data.fit(balanced)

    print("\nTesting Length-Controlled AlpacaEval model with Verbosity...")
    lc_eval_verbosity_data.test(balanced, name="verbosity")

    print("\nLength-controlled win rates successfully computed!")
