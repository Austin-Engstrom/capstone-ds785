"""
Run VADER sentiment analysis and evaluate it against manual sentiment labels.

This script:

1. Loads the manually labeled Pinkbike review dataset.
2. Applies VADER sentiment analysis to each full article.
3. Creates three-class and binary VADER predictions.
4. Evaluates predictions against the manually assigned labels.
5. Saves article-level predictions, performance metrics, confusion matrices,
   classification reports, and confusion-matrix visualizations.

Three-class labels:
- Positive
- Mixed
- Negative

Binary modeling labels:
- Positive
- Not Positive

AI Use:
AI tools were used to assist with code design, documentation, and workflow
planning.
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


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

OUTPUT_DIR = BASE_DIR / "data" / "modeling"

FIGURE_DIR = (
    BASE_DIR
    / "sentiment_analysis"
    / "outputs"
    / "figures"
)

TABLE_DIR = (
    BASE_DIR
    / "sentiment_analysis"
    / "outputs"
    / "tables"
)

PREDICTIONS_FILE = OUTPUT_DIR / "vader_predictions.csv"

METRICS_FILE = TABLE_DIR / "vader_metrics.csv"

THREE_CLASS_REPORT_FILE = (
    TABLE_DIR
    / "vader_three_class_classification_report.csv"
)

BINARY_REPORT_FILE = (
    TABLE_DIR
    / "vader_binary_classification_report.csv"
)

THREE_CLASS_CONFUSION_FILE = (
    TABLE_DIR
    / "vader_three_class_confusion_matrix.csv"
)

BINARY_CONFUSION_FILE = (
    TABLE_DIR
    / "vader_binary_confusion_matrix.csv"
)

THREE_CLASS_CONFUSION_FIGURE = (
    FIGURE_DIR
    / "vader_three_class_confusion_matrix.png"
)

BINARY_CONFUSION_FIGURE = (
    FIGURE_DIR
    / "vader_binary_confusion_matrix.png"
)


# ---------------------------------------------------------------------------
# Model settings
# ---------------------------------------------------------------------------

POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05

THREE_CLASS_LABELS = [
    "Negative",
    "Mixed",
    "Positive",
]

BINARY_LABELS = [
    "Not Positive",
    "Positive",
]


def load_labeled_reviews() -> pd.DataFrame:
    """
    Load the manually labeled review dataset.

    Returns
    -------
    pd.DataFrame
        Review-level dataset containing manual sentiment labels.

    Raises
    ------
    FileNotFoundError
        If the manual-labeling file does not exist.
    ValueError
        If the file is empty.
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


def validate_input_data(reviews: pd.DataFrame) -> None:
    """
    Validate fields required for VADER evaluation.

    Parameters
    ----------
    reviews : pd.DataFrame
        Manual sentiment-labeling dataset.

    Raises
    ------
    ValueError
        If required columns or valid labels are missing.
    """

    required_columns = [
        "article_id",
        "source_url",
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

    blank_text_count = (
        reviews["article_text"]
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    if blank_text_count > 0:
        raise ValueError(
            f"{blank_text_count} article records contain blank text."
        )

    blank_label_count = (
        reviews["manual_sentiment"]
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    if blank_label_count > 0:
        raise ValueError(
            f"{blank_label_count} articles do not have manual labels."
        )

    valid_labels = set(THREE_CLASS_LABELS)

    observed_labels = set(
        reviews["manual_sentiment"]
        .astype(str)
        .str.strip()
        .unique()
    )

    invalid_labels = observed_labels - valid_labels

    if invalid_labels:
        raise ValueError(
            "Unexpected manual sentiment labels were found: "
            + ", ".join(sorted(invalid_labels))
        )


def prepare_modeling_targets(
    reviews: pd.DataFrame,
) -> pd.DataFrame:
    """
    Standardize labels and create binary modeling targets.

    The original three-class label remains unchanged. Mixed and Negative
    reviews are grouped into Not Positive for binary evaluation.

    Parameters
    ----------
    reviews : pd.DataFrame
        Manual sentiment-labeling dataset.

    Returns
    -------
    pd.DataFrame
        Dataset with modeling_sentiment and is_positive columns.
    """

    prepared = reviews.copy()

    prepared["manual_sentiment"] = (
        prepared["manual_sentiment"]
        .astype(str)
        .str.strip()
        .str.title()
    )

    prepared["modeling_sentiment"] = (
        prepared["manual_sentiment"]
        .map(
            {
                "Positive": "Positive",
                "Mixed": "Not Positive",
                "Negative": "Not Positive",
            }
        )
    )

    prepared["is_positive"] = (
        prepared["manual_sentiment"]
        .eq("Positive")
        .astype(int)
    )

    return prepared


def classify_vader_compound(
    compound_score: float,
) -> str:
    """
    Convert a VADER compound score into a three-class sentiment label.

    Parameters
    ----------
    compound_score : float
        VADER compound sentiment score.

    Returns
    -------
    str
        Positive, Mixed, or Negative.
    """

    if compound_score >= POSITIVE_THRESHOLD:
        return "Positive"

    if compound_score <= NEGATIVE_THRESHOLD:
        return "Negative"

    return "Mixed"


def convert_to_binary_sentiment(
    sentiment: str,
) -> str:
    """
    Convert a three-class sentiment prediction to the binary target.

    Parameters
    ----------
    sentiment : str
        Three-class VADER prediction.

    Returns
    -------
    str
        Positive or Not Positive.
    """

    if sentiment == "Positive":
        return "Positive"

    return "Not Positive"


def analyze_articles_with_vader(
    reviews: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run VADER sentiment analysis on each article.

    Parameters
    ----------
    reviews : pd.DataFrame
        Prepared review dataset.

    Returns
    -------
    pd.DataFrame
        Review dataset with VADER scores and predictions.
    """

    analyzer = SentimentIntensityAnalyzer()

    results: list[dict[str, float]] = []

    total_articles = len(reviews)

    for article_number, article_text in enumerate(
        reviews["article_text"],
        start=1,
    ):
        sentiment_scores = analyzer.polarity_scores(
            str(article_text)
        )

        results.append(sentiment_scores)

        if (
            article_number % 25 == 0
            or article_number == total_articles
        ):
            print(
                "Processed "
                f"{article_number} of {total_articles} articles."
            )

    vader_scores = pd.DataFrame(results)

    vader_scores = vader_scores.rename(
        columns={
            "neg": "vader_negative",
            "neu": "vader_neutral",
            "pos": "vader_positive",
            "compound": "vader_compound",
        }
    )

    predictions = pd.concat(
        [
            reviews.reset_index(drop=True),
            vader_scores.reset_index(drop=True),
        ],
        axis=1,
    )

    predictions["vader_sentiment"] = (
        predictions["vader_compound"]
        .apply(classify_vader_compound)
    )

    predictions["vader_binary"] = (
        predictions["vader_sentiment"]
        .apply(convert_to_binary_sentiment)
    )

    predictions["vader_is_positive"] = (
        predictions["vader_binary"]
        .eq("Positive")
        .astype(int)
    )

    predictions["three_class_correct"] = (
        predictions["manual_sentiment"]
        .eq(predictions["vader_sentiment"])
    )

    predictions["binary_correct"] = (
        predictions["modeling_sentiment"]
        .eq(predictions["vader_binary"])
    )

    return predictions


def calculate_three_class_metrics(
    predictions: pd.DataFrame,
) -> dict[str, Any]:
    """
    Calculate three-class VADER evaluation metrics.

    Parameters
    ----------
    predictions : pd.DataFrame
        Dataset containing manual and VADER labels.

    Returns
    -------
    dict[str, Any]
        Three-class evaluation metrics.
    """

    actual = predictions["manual_sentiment"]
    predicted = predictions["vader_sentiment"]

    return {
        "evaluation_type": "Three Class",
        "accuracy": accuracy_score(actual, predicted),
        "balanced_accuracy": balanced_accuracy_score(
            actual,
            predicted,
        ),
        "macro_precision": precision_score(
            actual,
            predicted,
            labels=THREE_CLASS_LABELS,
            average="macro",
            zero_division=0,
        ),
        "macro_recall": recall_score(
            actual,
            predicted,
            labels=THREE_CLASS_LABELS,
            average="macro",
            zero_division=0,
        ),
        "macro_f1": f1_score(
            actual,
            predicted,
            labels=THREE_CLASS_LABELS,
            average="macro",
            zero_division=0,
        ),
        "weighted_precision": precision_score(
            actual,
            predicted,
            labels=THREE_CLASS_LABELS,
            average="weighted",
            zero_division=0,
        ),
        "weighted_recall": recall_score(
            actual,
            predicted,
            labels=THREE_CLASS_LABELS,
            average="weighted",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            actual,
            predicted,
            labels=THREE_CLASS_LABELS,
            average="weighted",
            zero_division=0,
        ),
        "cohen_kappa": cohen_kappa_score(
            actual,
            predicted,
            labels=THREE_CLASS_LABELS,
        ),
        "roc_auc": None,
        "article_count": len(predictions),
    }


def calculate_binary_metrics(
    predictions: pd.DataFrame,
) -> dict[str, Any]:
    """
    Calculate binary VADER evaluation metrics.

    Positive is treated as the positive class.

    Parameters
    ----------
    predictions : pd.DataFrame
        Dataset containing binary manual and VADER labels.

    Returns
    -------
    dict[str, Any]
        Binary evaluation metrics.
    """

    actual = predictions["modeling_sentiment"]
    predicted = predictions["vader_binary"]

    actual_numeric = predictions["is_positive"]

    return {
        "evaluation_type": "Binary",
        "accuracy": accuracy_score(actual, predicted),
        "balanced_accuracy": balanced_accuracy_score(
            actual,
            predicted,
        ),
        "macro_precision": precision_score(
            actual,
            predicted,
            labels=BINARY_LABELS,
            average="macro",
            zero_division=0,
        ),
        "macro_recall": recall_score(
            actual,
            predicted,
            labels=BINARY_LABELS,
            average="macro",
            zero_division=0,
        ),
        "macro_f1": f1_score(
            actual,
            predicted,
            labels=BINARY_LABELS,
            average="macro",
            zero_division=0,
        ),
        "weighted_precision": precision_score(
            actual,
            predicted,
            labels=BINARY_LABELS,
            average="weighted",
            zero_division=0,
        ),
        "weighted_recall": recall_score(
            actual,
            predicted,
            labels=BINARY_LABELS,
            average="weighted",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            actual,
            predicted,
            labels=BINARY_LABELS,
            average="weighted",
            zero_division=0,
        ),
        "cohen_kappa": cohen_kappa_score(
            actual,
            predicted,
            labels=BINARY_LABELS,
        ),
        "roc_auc": roc_auc_score(
            actual_numeric,
            predictions["vader_compound"],
        ),
        "article_count": len(predictions),
    }


def create_classification_report(
    actual: pd.Series,
    predicted: pd.Series,
    labels: list[str],
) -> pd.DataFrame:
    """
    Create a classification report as a dataframe.

    Parameters
    ----------
    actual : pd.Series
        Ground-truth labels.
    predicted : pd.Series
        Model predictions.
    labels : list[str]
        Ordered class labels.

    Returns
    -------
    pd.DataFrame
        Classification report with one row per class and summary metric.
    """

    report = classification_report(
        actual,
        predicted,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    report_df = (
        pd.DataFrame(report)
        .transpose()
        .reset_index()
        .rename(columns={"index": "class"})
    )

    return report_df


def create_confusion_matrix_dataframe(
    actual: pd.Series,
    predicted: pd.Series,
    labels: list[str],
) -> pd.DataFrame:
    """
    Create a labeled confusion-matrix dataframe.

    Parameters
    ----------
    actual : pd.Series
        Ground-truth labels.
    predicted : pd.Series
        Model predictions.
    labels : list[str]
        Ordered class labels.

    Returns
    -------
    pd.DataFrame
        Confusion matrix with labeled rows and columns.
    """

    matrix = confusion_matrix(
        actual,
        predicted,
        labels=labels,
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=[
            f"actual_{label}"
            for label in labels
        ],
        columns=[
            f"predicted_{label}"
            for label in labels
        ],
    )

    return matrix_df


def save_confusion_matrix_figure(
    matrix_df: pd.DataFrame,
    labels: list[str],
    title: str,
    output_file: Path,
) -> None:
    """
    Save a confusion matrix as a presentation-ready image.

    Parameters
    ----------
    matrix_df : pd.DataFrame
        Labeled confusion matrix.
    labels : list[str]
        Ordered class labels.
    title : str
        Figure title.
    output_file : Path
        Destination PNG file.
    """

    matrix = matrix_df.to_numpy()

    figure, axis = plt.subplots(
        figsize=(7, 6)
    )

    image = axis.imshow(matrix)

    axis.set_title(title)
    axis.set_xlabel("Predicted sentiment")
    axis.set_ylabel("Actual sentiment")

    axis.set_xticks(
        range(len(labels)),
        labels=labels,
        rotation=30,
        ha="right",
    )

    axis.set_yticks(
        range(len(labels)),
        labels=labels,
    )

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axis.text(
                column_index,
                row_index,
                str(matrix[row_index, column_index]),
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
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_outputs(
    predictions: pd.DataFrame,
) -> None:
    """
    Create and save all VADER evaluation outputs.

    Parameters
    ----------
    predictions : pd.DataFrame
        Article-level VADER prediction dataset.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_csv(
        PREDICTIONS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    three_class_metrics = (
        calculate_three_class_metrics(predictions)
    )

    binary_metrics = (
        calculate_binary_metrics(predictions)
    )

    metrics_df = pd.DataFrame(
        [
            three_class_metrics,
            binary_metrics,
        ]
    )

    metrics_df.to_csv(
        METRICS_FILE,
        index=False,
    )

    three_class_report = create_classification_report(
        actual=predictions["manual_sentiment"],
        predicted=predictions["vader_sentiment"],
        labels=THREE_CLASS_LABELS,
    )

    three_class_report.to_csv(
        THREE_CLASS_REPORT_FILE,
        index=False,
    )

    binary_report = create_classification_report(
        actual=predictions["modeling_sentiment"],
        predicted=predictions["vader_binary"],
        labels=BINARY_LABELS,
    )

    binary_report.to_csv(
        BINARY_REPORT_FILE,
        index=False,
    )

    three_class_confusion = (
        create_confusion_matrix_dataframe(
            actual=predictions["manual_sentiment"],
            predicted=predictions["vader_sentiment"],
            labels=THREE_CLASS_LABELS,
        )
    )

    three_class_confusion.to_csv(
        THREE_CLASS_CONFUSION_FILE,
    )

    binary_confusion = (
        create_confusion_matrix_dataframe(
            actual=predictions["modeling_sentiment"],
            predicted=predictions["vader_binary"],
            labels=BINARY_LABELS,
        )
    )

    binary_confusion.to_csv(
        BINARY_CONFUSION_FILE,
    )

    save_confusion_matrix_figure(
        matrix_df=three_class_confusion,
        labels=THREE_CLASS_LABELS,
        title="VADER Three-Class Confusion Matrix",
        output_file=THREE_CLASS_CONFUSION_FIGURE,
    )

    save_confusion_matrix_figure(
        matrix_df=binary_confusion,
        labels=BINARY_LABELS,
        title="VADER Binary Confusion Matrix",
        output_file=BINARY_CONFUSION_FIGURE,
    )


def print_results_summary(
    predictions: pd.DataFrame,
) -> None:
    """
    Print key VADER results to the terminal.

    Parameters
    ----------
    predictions : pd.DataFrame
        Article-level VADER prediction dataset.
    """

    three_class_metrics = (
        calculate_three_class_metrics(predictions)
    )

    binary_metrics = (
        calculate_binary_metrics(predictions)
    )

    print()
    print("VADER sentiment analysis completed successfully.")
    print(f"Articles analyzed: {len(predictions)}")

    print()
    print("Manual sentiment distribution:")
    print(
        predictions["manual_sentiment"]
        .value_counts()
        .to_string()
    )

    print()
    print("VADER sentiment distribution:")
    print(
        predictions["vader_sentiment"]
        .value_counts()
        .to_string()
    )

    print()
    print("Three-class evaluation:")
    print(
        f"Accuracy: "
        f"{three_class_metrics['accuracy']:.4f}"
    )
    print(
        f"Balanced accuracy: "
        f"{three_class_metrics['balanced_accuracy']:.4f}"
    )
    print(
        f"Macro F1: "
        f"{three_class_metrics['macro_f1']:.4f}"
    )
    print(
        f"Weighted F1: "
        f"{three_class_metrics['weighted_f1']:.4f}"
    )

    print()
    print("Binary evaluation:")
    print(
        f"Accuracy: "
        f"{binary_metrics['accuracy']:.4f}"
    )
    print(
        f"Balanced accuracy: "
        f"{binary_metrics['balanced_accuracy']:.4f}"
    )
    print(
        f"Macro F1: "
        f"{binary_metrics['macro_f1']:.4f}"
    )
    print(
        f"Weighted F1: "
        f"{binary_metrics['weighted_f1']:.4f}"
    )
    print(
        f"ROC AUC: "
        f"{binary_metrics['roc_auc']:.4f}"
    )

    print()
    print("Saved outputs:")
    print(f"- Predictions: {PREDICTIONS_FILE}")
    print(f"- Metrics: {METRICS_FILE}")
    print(
        "- Three-class confusion matrix: "
        f"{THREE_CLASS_CONFUSION_FILE}"
    )
    print(
        "- Binary confusion matrix: "
        f"{BINARY_CONFUSION_FILE}"
    )
    print(
        "- Three-class confusion figure: "
        f"{THREE_CLASS_CONFUSION_FIGURE}"
    )
    print(
        "- Binary confusion figure: "
        f"{BINARY_CONFUSION_FIGURE}"
    )


def main() -> None:
    """
    Run the complete VADER baseline workflow.
    """

    reviews = load_labeled_reviews()

    validate_input_data(reviews)

    reviews = prepare_modeling_targets(reviews)

    predictions = analyze_articles_with_vader(reviews)

    save_outputs(predictions)

    print_results_summary(predictions)


if __name__ == "__main__":
    main()