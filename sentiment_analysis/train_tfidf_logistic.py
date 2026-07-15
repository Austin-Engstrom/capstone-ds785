"""
Train and evaluate a supervised sentiment classifier using TF-IDF features
and logistic regression.

This script:

1. Loads manually labeled Pinkbike review articles.
2. Creates a binary target: Positive versus Not Positive.
3. Creates a stratified train/test split.
4. Tunes TF-IDF and logistic-regression parameters with GridSearchCV.
5. Uses stratified five-fold cross-validation and macro F1 scoring.
6. Evaluates the selected model on a held-out test set.
7. Extracts interpretable feature coefficients.
8. Saves predictions, misclassifications, metrics, CV results, tables,
   figures, and the fitted model pipeline.

AI Use:
AI tools were used to assist with code design, documentation, and workflow
planning.
"""

from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    learning_curve,
    train_test_split,
)
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "modeling"
    / "manual_sentiment_labels.csv"
)

MODELING_OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "modeling"
)

TABLE_OUTPUT_DIR = (
    BASE_DIR
    / "sentiment_analysis"
    / "outputs"
    / "tables"
)

FIGURE_OUTPUT_DIR = (
    BASE_DIR
    / "sentiment_analysis"
    / "outputs"
    / "figures"
)

MODEL_OUTPUT_DIR = (
    BASE_DIR
    / "sentiment_analysis"
    / "outputs"
    / "models"
)

MODEL_FILE = (
    MODEL_OUTPUT_DIR
    / "tfidf_logistic_pipeline.joblib"
)

PREDICTIONS_FILE = (
    MODELING_OUTPUT_DIR
    / "tfidf_logistic_identifier_removed_predictions.csv"
)

MISCLASSIFICATIONS_FILE = (
    MODELING_OUTPUT_DIR
    / "tfidf_logistic_identifier_removed_misclassified_articles.csv"
)

METRICS_FILE = (
    TABLE_OUTPUT_DIR
    / "tfidf_logistic_identifier_removed_metrics.csv"
)

CLASSIFICATION_REPORT_FILE = (
    TABLE_OUTPUT_DIR
    / "tfidf_logistic_identifier_removed_classification_report.csv"
)

CV_RESULTS_FILE = (
    TABLE_OUTPUT_DIR
    / "tfidf_logistic_identifier_removed_cv_results.csv"
)

BEST_PARAMETERS_FILE = (
    TABLE_OUTPUT_DIR
    / "tfidf_logistic_identifier_removed_best_parameters.csv"
)

CONFUSION_MATRIX_FIGURE = (
    FIGURE_OUTPUT_DIR
    / "tfidf_logistic_confusion_matrix.png"
)

ROC_CURVE_FIGURE = (
    FIGURE_OUTPUT_DIR
    / "tfidf_logistic_roc_curve.png"
)

PRECISION_RECALL_FIGURE = (
    FIGURE_OUTPUT_DIR
    / "tfidf_logistic_precision_recall_curve.png"
)

POSITIVE_FEATURES_FIGURE = (
    FIGURE_OUTPUT_DIR
    / "tfidf_logistic_top_positive_features.png"
)

NOT_POSITIVE_FEATURES_FIGURE = (
    FIGURE_OUTPUT_DIR
    / "tfidf_logistic_top_not_positive_features.png"
)

LEARNING_CURVE_FIGURE = (
    FIGURE_OUTPUT_DIR
    / "tfidf_logistic_learning_curve.png"
)

# ---------------------------------------------------------------------------
# Modeling settings
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
TOP_FEATURE_COUNT = 20

TARGET_LABELS = [
    "Not Positive",
    "Positive",
]

POSITIVE_CLASS = "Positive"

VALID_MANUAL_LABELS = {
    "Positive",
    "Mixed",
    "Negative",
}


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def load_labeled_reviews() -> pd.DataFrame:
    """
    Load the manually labeled article dataset.

    Returns
    -------
    pd.DataFrame
        One row per labeled review article.
    """

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Manual-labeling file was not found: {INPUT_FILE}"
        )

    reviews = pd.read_csv(
        INPUT_FILE,
        keep_default_na=False,
    )

    if reviews.empty:
        raise ValueError(
            "The manual-labeling file contains no records."
        )

    return reviews


def validate_input_data(
    reviews: pd.DataFrame,
) -> None:
    """
    Validate required article fields and sentiment labels.
    """

    required_columns = [
        "article_id",
        "source_url",
        "title",
        "manual_sentiment",
        "article_text",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in reviews.columns
    ]

    if missing_columns:
        raise ValueError(
            "The following required columns are missing: "
            + ", ".join(missing_columns)
        )

    duplicate_article_ids = (
        reviews["article_id"]
        .astype(str)
        .duplicated()
        .sum()
    )

    if duplicate_article_ids > 0:
        raise ValueError(
            f"{duplicate_article_ids} duplicated article IDs were found."
        )

    blank_text_count = (
        reviews["article_text"]
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    if blank_text_count > 0:
        raise ValueError(
            f"{blank_text_count} articles contain blank text."
        )

    observed_labels = set(
        reviews["manual_sentiment"]
        .astype(str)
        .str.strip()
        .str.title()
        .unique()
    )

    invalid_labels = (
        observed_labels
        - VALID_MANUAL_LABELS
    )

    if invalid_labels:
        raise ValueError(
            "Unexpected manual sentiment labels were found: "
            + ", ".join(sorted(invalid_labels))
        )

def remove_product_identifiers(
    row: pd.Series,
) -> str:
    """
    Remove known brand and product-name identifiers from article text.

    This reduces the risk that the classifier learns product-specific
    shortcuts instead of language associated with sentiment.

    Parameters
    ----------
    row : pd.Series
        Article record containing article_text, brand, and product_name.

    Returns
    -------
    str
        Article text with known product identifiers removed.
    """

    cleaned_text = str(
        row["article_text"]
    )

    identifiers = []

    for column in [
        "brand",
        "product_name",
    ]:
        if column not in row.index:
            continue

        value = str(row[column]).strip()

        if (
            value
            and value.lower() != "nan"
        ):
            identifiers.append(value)

    # Remove longer identifiers first so a full product name is removed
    # before a shorter brand name contained within it.
    identifiers = sorted(
        set(identifiers),
        key=len,
        reverse=True,
    )

    for identifier in identifiers:
        cleaned_text = re.sub(
            pattern=re.escape(identifier),
            repl=" ",
            string=cleaned_text,
            flags=re.IGNORECASE,
        )

    cleaned_text = re.sub(
        r"\s+",
        " ",
        cleaned_text,
    )

    return cleaned_text.strip()

def prepare_modeling_data(
    reviews: pd.DataFrame,
) -> pd.DataFrame:
    """
    Standardize labels and create the binary supervised target.
    """

    prepared = reviews.copy()

    prepared["manual_sentiment"] = (
        prepared["manual_sentiment"]
        .astype(str)
        .str.strip()
        .str.title()
    )

    prepared["modeling_sentiment"] = np.where(
        prepared["manual_sentiment"].eq("Positive"),
        "Positive",
        "Not Positive",
    )

    prepared["is_positive"] = (
        prepared["modeling_sentiment"]
        .eq("Positive")
        .astype(int)
    )

    prepared["article_text"] = (
        prepared["article_text"]
        .astype(str)
        .str.replace(
            r"\s+",
            " ",
            regex=True,
        )
        .str.strip()
    )

    prepared["modeling_text"] = prepared.apply(
        remove_product_identifiers,
        axis=1,
    )
    return prepared


# ---------------------------------------------------------------------------
# Model construction and tuning
# ---------------------------------------------------------------------------

def create_pipeline() -> Pipeline:
    """
    Create the TF-IDF and logistic-regression pipeline.
    """

    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "logistic",
                LogisticRegression(
                    penalty="l2",
                    solver="liblinear",
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def create_parameter_grid() -> dict[str, list[Any]]:
    """
    Create the GridSearchCV parameter grid.
    """

    return {
        "tfidf__ngram_range": [
            (1, 1),
            (1, 2),
        ],
        "tfidf__min_df": [
            2,
            3,
        ],
        "tfidf__max_df": [
            0.85,
            0.90,
            0.95,
        ],
        "logistic__C": [
            0.01,
            0.1,
            1.0,
            10.0,
            100.0,
        ],
        "logistic__class_weight": [
            None,
            "balanced",
        ],
    }


def create_cross_validator() -> StratifiedKFold:
    """
    Create the reproducible five-fold validation strategy.
    """

    return StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )


def perform_grid_search(
    pipeline: Pipeline,
    parameter_grid: dict[str, list[Any]],
    cross_validator: StratifiedKFold,
    x_train: pd.Series,
    y_train: pd.Series,
) -> GridSearchCV:
    """
    Tune the full TF-IDF and logistic-regression pipeline.
    """

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring="f1_macro",
        cv=cross_validator,
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
        refit=True,
    )

    grid_search.fit(
        x_train,
        y_train,
    )

    return grid_search


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def calculate_test_metrics(
    y_test: pd.Series,
    predictions: np.ndarray,
    positive_probabilities: np.ndarray,
) -> dict[str, Any]:
    """
    Calculate held-out test metrics.
    """

    numeric_actual = (
        y_test
        .eq(POSITIVE_CLASS)
        .astype(int)
    )

    return {
        "model": (
            "Supervised Logistic Regression "
            "with TF-IDF"
        ),
        "evaluation_type": "Binary Test Set",
        "accuracy": accuracy_score(
            y_test,
            predictions,
        ),
        "balanced_accuracy":
            balanced_accuracy_score(
                y_test,
                predictions,
            ),
        "macro_precision":
            precision_score(
                y_test,
                predictions,
                labels=TARGET_LABELS,
                average="macro",
                zero_division=0,
            ),
        "macro_recall":
            recall_score(
                y_test,
                predictions,
                labels=TARGET_LABELS,
                average="macro",
                zero_division=0,
            ),
        "macro_f1":
            f1_score(
                y_test,
                predictions,
                labels=TARGET_LABELS,
                average="macro",
                zero_division=0,
            ),
        "weighted_precision":
            precision_score(
                y_test,
                predictions,
                labels=TARGET_LABELS,
                average="weighted",
                zero_division=0,
            ),
        "weighted_recall":
            recall_score(
                y_test,
                predictions,
                labels=TARGET_LABELS,
                average="weighted",
                zero_division=0,
            ),
        "weighted_f1":
            f1_score(
                y_test,
                predictions,
                labels=TARGET_LABELS,
                average="weighted",
                zero_division=0,
            ),
        "cohen_kappa":
            cohen_kappa_score(
                y_test,
                predictions,
                labels=TARGET_LABELS,
            ),
        "roc_auc":
            roc_auc_score(
                numeric_actual,
                positive_probabilities,
            ),
        "test_article_count": len(y_test),
    }


def get_positive_probability_index(
    fitted_pipeline: Pipeline,
) -> int:
    """
    Find the probability-column index for the Positive class.
    """

    logistic_model = (
        fitted_pipeline
        .named_steps["logistic"]
    )

    classes = list(
        logistic_model.classes_
    )

    if POSITIVE_CLASS not in classes:
        raise ValueError(
            "The fitted model does not contain the Positive class."
        )

    return classes.index(POSITIVE_CLASS)


def create_prediction_output(
    test_data: pd.DataFrame,
    predictions: np.ndarray,
    positive_probabilities: np.ndarray,
) -> pd.DataFrame:
    """
    Create article-level held-out prediction results.
    """

    output_columns = [
        "article_id",
        "source_url",
        "title",
        "author",
        "publish_date",
        "brand",
        "product_name",
        "product_category",
        "product_subcategory",
        "review_type",
        "manual_sentiment",
        "modeling_sentiment",
        "label_confidence",
        "article_text",
    ]

    available_columns = [
        column
        for column in output_columns
        if column in test_data.columns
    ]

    output = (
        test_data[available_columns]
        .copy()
        .reset_index(drop=True)
    )

    output[
        "tfidf_logistic_prediction"
    ] = predictions

    output[
        "tfidf_positive_probability"
    ] = positive_probabilities

    output[
        "tfidf_not_positive_probability"
    ] = (
        1.0
        - output["tfidf_positive_probability"]
    )

    output["prediction_correct"] = (
        output["modeling_sentiment"]
        .eq(
            output[
                "tfidf_logistic_prediction"
            ]
        )
    )

    output["prediction_confidence"] = np.where(
        output[
            "tfidf_logistic_prediction"
        ].eq(POSITIVE_CLASS),
        output["tfidf_positive_probability"],
        output[
            "tfidf_not_positive_probability"
        ],
    )

    output["error_type"] = np.select(
        condlist=[
            (
                output["modeling_sentiment"]
                .eq("Not Positive")
                & output[
                    "tfidf_logistic_prediction"
                ].eq("Positive")
            ),
            (
                output["modeling_sentiment"]
                .eq("Positive")
                & output[
                    "tfidf_logistic_prediction"
                ].eq("Not Positive")
            ),
        ],
        choicelist=[
            "False Positive",
            "False Negative",
        ],
        default="Correct",
    )

    return output


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def extract_feature_importance(
    fitted_pipeline: Pipeline,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract highest positive and most negative logistic coefficients.
    """

    vectorizer = (
        fitted_pipeline
        .named_steps["tfidf"]
    )

    logistic_model = (
        fitted_pipeline
        .named_steps["logistic"]
    )

    feature_names = (
        vectorizer
        .get_feature_names_out()
    )

    coefficients = (
        logistic_model
        .coef_[0]
    )

    coefficient_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
            "absolute_coefficient": np.abs(
                coefficients
            ),
        }
    )

    positive_features = (
        coefficient_df
        .sort_values(
            "coefficient",
            ascending=False,
        )
        .head(TOP_FEATURE_COUNT)
        .reset_index(drop=True)
    )

    not_positive_features = (
        coefficient_df
        .sort_values(
            "coefficient",
            ascending=True,
        )
        .head(TOP_FEATURE_COUNT)
        .reset_index(drop=True)
    )

    positive_features[
        "associated_class"
    ] = "Positive"

    not_positive_features[
        "associated_class"
    ] = "Not Positive"

    return (
        positive_features,
        not_positive_features,
    )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def save_confusion_matrix_outputs(
    y_test: pd.Series,
    predictions: np.ndarray,
) -> pd.DataFrame:
    """
    Save a binary confusion matrix as CSV and PNG.
    """

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=TARGET_LABELS,
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=[
            f"actual_{label}"
            for label in TARGET_LABELS
        ],
        columns=[
            f"predicted_{label}"
            for label in TARGET_LABELS
        ],
    )

    matrix_df.to_csv(
        CONFUSION_MATRIX_FIGURE,
    )

    figure, axis = plt.subplots(
        figsize=(7, 6)
    )

    image = axis.imshow(matrix)

    axis.set_title(
        "TF-IDF Logistic Regression Confusion Matrix"
    )

    axis.set_xlabel("Predicted sentiment")
    axis.set_ylabel("Actual sentiment")

    axis.set_xticks(
        range(len(TARGET_LABELS)),
        labels=TARGET_LABELS,
        rotation=25,
        ha="right",
    )

    axis.set_yticks(
        range(len(TARGET_LABELS)),
        labels=TARGET_LABELS,
    )

    for row_index in range(matrix.shape[0]):
        for column_index in range(
            matrix.shape[1]
        ):
            axis.text(
                column_index,
                row_index,
                str(
                    matrix[
                        row_index,
                        column_index,
                    ]
                ),
                ha="center",
                va="center",
            )

    figure.colorbar(
        image,
        ax=axis,
        label="Article count",
    )

    figure.tight_layout()

    figure.savefig(
        CONFUSION_MATRIX_FIGURE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    return matrix_df


def save_roc_curve(
    y_test: pd.Series,
    positive_probabilities: np.ndarray,
) -> None:
    """
    Save the held-out ROC curve.
    """

    numeric_actual = (
        y_test
        .eq(POSITIVE_CLASS)
        .astype(int)
    )

    false_positive_rate, true_positive_rate, _ = (
        roc_curve(
            numeric_actual,
            positive_probabilities,
        )
    )

    auc_value = roc_auc_score(
        numeric_actual,
        positive_probabilities,
    )

    figure, axis = plt.subplots(
        figsize=(7, 6)
    )

    axis.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"ROC AUC = {auc_value:.3f}",
    )

    axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random classifier",
    )

    axis.set_title(
        "TF-IDF Logistic Regression ROC Curve"
    )

    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.legend(loc="lower right")

    figure.tight_layout()

    figure.savefig(
        ROC_CURVE_FIGURE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_precision_recall_curve(
    y_test: pd.Series,
    positive_probabilities: np.ndarray,
) -> None:
    """
    Save the held-out precision-recall curve.
    """

    numeric_actual = (
        y_test
        .eq(POSITIVE_CLASS)
        .astype(int)
    )

    precision, recall, _ = (
        precision_recall_curve(
            numeric_actual,
            positive_probabilities,
        )
    )

    baseline = numeric_actual.mean()

    figure, axis = plt.subplots(
        figsize=(7, 6)
    )

    axis.plot(
        recall,
        precision,
        label="TF-IDF Logistic Regression",
    )

    axis.axhline(
        baseline,
        linestyle="--",
        label=(
            "Positive-class prevalence "
            f"= {baseline:.3f}"
        ),
    )

    axis.set_title(
        "TF-IDF Logistic Regression "
        "Precision-Recall Curve"
    )

    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.legend(loc="lower left")

    figure.tight_layout()

    figure.savefig(
        PRECISION_RECALL_FIGURE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_feature_figure(
    features: pd.DataFrame,
    title: str,
    output_file: Path,
) -> None:
    """
    Save a horizontal feature-coefficient chart.
    """

    chart_data = (
        features
        .sort_values(
            "coefficient",
            ascending=True,
        )
    )

    figure, axis = plt.subplots(
        figsize=(9, 7)
    )

    axis.barh(
        chart_data["feature"],
        chart_data["coefficient"],
    )

    axis.set_title(title)
    axis.set_xlabel(
        "Logistic regression coefficient"
    )
    axis.set_ylabel("TF-IDF feature")

    figure.tight_layout()

    figure.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_learning_curve(
    fitted_pipeline: Pipeline,
    x_train: pd.Series,
    y_train: pd.Series,
    cross_validator: StratifiedKFold,
) -> None:
    """
    Generate a macro-F1 learning curve.
    """

    train_sizes, train_scores, validation_scores = (
        learning_curve(
            estimator=fitted_pipeline,
            X=x_train,
            y=y_train,
            cv=cross_validator,
            scoring="f1_macro",
            train_sizes=np.linspace(
                0.30,
                1.00,
                5,
            ),
            n_jobs=-1,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
    )

    train_means = train_scores.mean(axis=1)
    train_standard_deviations = (
        train_scores.std(axis=1)
    )

    validation_means = (
        validation_scores.mean(axis=1)
    )

    validation_standard_deviations = (
        validation_scores.std(axis=1)
    )

    figure, axis = plt.subplots(
        figsize=(8, 6)
    )

    axis.plot(
        train_sizes,
        train_means,
        marker="o",
        label="Training macro F1",
    )

    axis.plot(
        train_sizes,
        validation_means,
        marker="o",
        label="Validation macro F1",
    )

    axis.fill_between(
        train_sizes,
        train_means
        - train_standard_deviations,
        train_means
        + train_standard_deviations,
        alpha=0.15,
    )

    axis.fill_between(
        train_sizes,
        validation_means
        - validation_standard_deviations,
        validation_means
        + validation_standard_deviations,
        alpha=0.15,
    )

    axis.set_title(
        "TF-IDF Logistic Regression Learning Curve"
    )

    axis.set_xlabel(
        "Training articles per fold"
    )

    axis.set_ylabel("Macro F1")
    axis.legend()

    figure.tight_layout()

    figure.savefig(
        LEARNING_CURVE_FIGURE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


# ---------------------------------------------------------------------------
# Output handling
# ---------------------------------------------------------------------------

def create_output_directories() -> None:
    """
    Create all output folders.
    """

    for output_directory in [
        MODELING_OUTPUT_DIR,
        TABLE_OUTPUT_DIR,
        FIGURE_OUTPUT_DIR,
        MODEL_OUTPUT_DIR,
    ]:
        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )


def save_grid_search_outputs(
    grid_search: GridSearchCV,
) -> None:
    """
    Save cross-validation results and best parameters.
    """

    cv_results = pd.DataFrame(
        grid_search.cv_results_
    )

    selected_columns = [
        "rank_test_score",
        "mean_test_score",
        "std_test_score",
        "mean_train_score",
        "std_train_score",
        "mean_fit_time",
        "param_tfidf__ngram_range",
        "param_tfidf__min_df",
        "param_tfidf__max_df",
        "param_logistic__C",
        "param_logistic__class_weight",
    ]

    available_columns = [
        column
        for column in selected_columns
        if column in cv_results.columns
    ]

    cv_results = (
        cv_results[available_columns]
        .sort_values(
            [
                "rank_test_score",
                "mean_test_score",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    cv_results.to_csv(
        CV_RESULTS_FILE,
        index=False,
    )

    best_parameters = pd.DataFrame(
        [
            {
                "parameter": parameter,
                "value": str(value),
            }
            for parameter, value
            in grid_search.best_params_.items()
        ]
    )

    best_parameters.loc[
        len(best_parameters)
    ] = {
        "parameter": (
            "best_cross_validation_macro_f1"
        ),
        "value": (
            f"{grid_search.best_score_:.6f}"
        ),
    }

    best_parameters.to_csv(
        BEST_PARAMETERS_FILE,
        index=False,
    )


def save_classification_report(
    y_test: pd.Series,
    predictions: np.ndarray,
) -> pd.DataFrame:
    """
    Save the binary test classification report.
    """

    report = classification_report(
        y_test,
        predictions,
        labels=TARGET_LABELS,
        output_dict=True,
        zero_division=0,
    )

    report_df = (
        pd.DataFrame(report)
        .transpose()
        .reset_index()
        .rename(columns={"index": "class"})
    )

    report_df.to_csv(
        CLASSIFICATION_REPORT_FILE,
        index=False,
    )

    return report_df


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Run the complete supervised text-classification workflow.
    """

    create_output_directories()

    reviews = load_labeled_reviews()

    validate_input_data(reviews)

    reviews = prepare_modeling_data(reviews)

    train_data, test_data = train_test_split(
        reviews,
        test_size=TEST_SIZE,
        stratify=reviews[
            "modeling_sentiment"
        ],
        random_state=RANDOM_STATE,
    )

    train_data = train_data.reset_index(
        drop=True
    )

    test_data = test_data.reset_index(
        drop=True
    )

    x_train = train_data["article_text"]
    y_train = train_data[
        "modeling_sentiment"
    ]

    x_test = test_data["article_text"]
    y_test = test_data[
        "modeling_sentiment"
    ]

    print(
        "Training supervised TF-IDF "
        "logistic-regression model."
    )

    print(f"Total articles: {len(reviews)}")
    print(f"Training articles: {len(train_data)}")
    print(f"Test articles: {len(test_data)}")

    print()
    print("Training class distribution:")
    print(
        y_train
        .value_counts()
        .to_string()
    )

    print()
    print("Test class distribution:")
    print(
        y_test
        .value_counts()
        .to_string()
    )

    pipeline = create_pipeline()

    parameter_grid = (
        create_parameter_grid()
    )

    cross_validator = (
        create_cross_validator()
    )

    grid_search = perform_grid_search(
        pipeline=pipeline,
        parameter_grid=parameter_grid,
        cross_validator=cross_validator,
        x_train=x_train,
        y_train=y_train,
    )

    best_pipeline = (
        grid_search.best_estimator_
    )

    predictions = best_pipeline.predict(
        x_test
    )

    positive_probability_index = (
        get_positive_probability_index(
            best_pipeline
        )
    )

    positive_probabilities = (
        best_pipeline
        .predict_proba(x_test)[
            :,
            positive_probability_index,
        ]
    )

    test_metrics = calculate_test_metrics(
        y_test=y_test,
        predictions=predictions,
        positive_probabilities=positive_probabilities,
    )

    metrics_df = pd.DataFrame(
        [test_metrics]
    )

    metrics_df[
        "best_cv_macro_f1"
    ] = grid_search.best_score_

    metrics_df[
        "train_article_count"
    ] = len(train_data)

    metrics_df.to_csv(
        METRICS_FILE,
        index=False,
    )

    prediction_output = (
        create_prediction_output(
            test_data=test_data,
            predictions=predictions,
            positive_probabilities=positive_probabilities,
        )
    )

    prediction_output.to_csv(
        PREDICTIONS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    misclassifications = (
        prediction_output[
            ~prediction_output[
                "prediction_correct"
            ]
        ]
        .sort_values(
            "prediction_confidence",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    misclassifications.to_csv(
        MISCLASSIFICATIONS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    save_grid_search_outputs(
        grid_search
    )

    save_classification_report(
        y_test=y_test,
        predictions=predictions,
    )

    save_confusion_matrix_outputs(
        y_test=y_test,
        predictions=predictions,
    )

    save_roc_curve(
        y_test=y_test,
        positive_probabilities=positive_probabilities,
    )

    save_precision_recall_curve(
        y_test=y_test,
        positive_probabilities=positive_probabilities,
    )

    (
        positive_features,
        not_positive_features,
    ) = extract_feature_importance(
        best_pipeline
    )

    positive_features.to_csv(
        POSITIVE_FEATURES_FIGURE,
        index=False,
    )

    not_positive_features.to_csv(
        NOT_POSITIVE_FEATURES_FIGURE,
        index=False,
    )

    save_feature_figure(
        features=positive_features,
        title=(
            "Top TF-IDF Features Associated "
            "with Positive Reviews"
        ),
        output_file=POSITIVE_FEATURES_FIGURE,
    )

    save_feature_figure(
        features=not_positive_features,
        title=(
            "Top TF-IDF Features Associated "
            "with Not-Positive Reviews"
        ),
        output_file=NOT_POSITIVE_FEATURES_FIGURE,
    )

    save_learning_curve(
        fitted_pipeline=best_pipeline,
        x_train=x_train,
        y_train=y_train,
        cross_validator=cross_validator,
    )

    joblib.dump(
        best_pipeline,
        MODEL_FILE,
    )

    print()
    print(
        "TF-IDF logistic-regression training "
        "completed successfully."
    )

    print()
    print("Best parameters:")

    for parameter, value in (
        grid_search.best_params_.items()
    ):
        print(f"- {parameter}: {value}")

    print(
        "- Cross-validation macro F1: "
        f"{grid_search.best_score_:.4f}"
    )

    print()
    print("Held-out test evaluation:")

    print(
        f"Accuracy: "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"Balanced accuracy: "
        f"{test_metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"Macro F1: "
        f"{test_metrics['macro_f1']:.4f}"
    )

    print(
        f"Weighted F1: "
        f"{test_metrics['weighted_f1']:.4f}"
    )

    print(
        f"ROC AUC: "
        f"{test_metrics['roc_auc']:.4f}"
    )

    print(
        f"Misclassified test articles: "
        f"{len(misclassifications)}"
    )

    print()
    print("Saved outputs:")

    print(f"- Predictions: {PREDICTIONS_FILE}")
    print(
        f"- Misclassifications: "
        f"{MISCLASSIFICATIONS_FILE}"
    )
    print(f"- Metrics: {METRICS_FILE}")
    print(f"- CV results: {CV_RESULTS_FILE}")
    print(
        f"- Best parameters: "
        f"{BEST_PARAMETERS_FILE}"
    )
    print(
        f"- Classification report: "
        f"{CLASSIFICATION_REPORT_FILE}"
    )
    print(
        f"- Confusion matrix: "
        f"{CONFUSION_MATRIX_FIGURE}"
    )
    print(
        f"- Positive features: "
        f"{POSITIVE_FEATURES_FIGURE}"
    )
    print(
        f"- Not-positive features: "
        f"{NOT_POSITIVE_FEATURES_FIGURE}"
    )
    print(f"- Fitted model: {MODEL_FILE}")


if __name__ == "__main__":
    main()