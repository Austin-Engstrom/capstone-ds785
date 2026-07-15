"""
Run pretrained transformer sentiment analysis on review text chunks.

This script:

1. Loads sentence-aware review chunks.
2. Loads a pretrained CardiffNLP RoBERTa sentiment model.
3. Runs batched inference using CUDA, Apple MPS, or CPU.
4. Saves chunk-level probabilities with resume support.
5. Aggregates chunk probabilities to article-level predictions.
6. Evaluates predictions against manual labels.
7. Saves metrics, classification reports, confusion matrices, and figures.

Transformer output labels:
- Negative
- Neutral
- Positive

Project labels:
- Negative
- Mixed
- Positive

The transformer Neutral class is mapped to Mixed for article-level comparison.
This mapping is necessary for evaluation but should not be interpreted as an
exact conceptual equivalence.

AI Use:
AI tools were used to assist with code design, documentation, and workflow
planning.
"""

from pathlib import Path
import math
import time
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import torch
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
from torch.nn.functional import softmax
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "modeling"
    / "review_text_chunks.csv"
)

OUTPUT_DIR = BASE_DIR / "data" / "modeling"

TABLE_DIR = (
    BASE_DIR
    / "sentiment_analysis"
    / "outputs"
    / "tables"
)

FIGURE_DIR = (
    BASE_DIR
    / "sentiment_analysis"
    / "outputs"
    / "figures"
)

CHUNK_PREDICTIONS_FILE = (
    OUTPUT_DIR
    / "transformer_chunk_predictions.csv"
)

ARTICLE_PREDICTIONS_FILE = (
    OUTPUT_DIR
    / "transformer_article_predictions.csv"
)

METRICS_FILE = (
    TABLE_DIR
    / "transformer_metrics.csv"
)

THREE_CLASS_REPORT_FILE = (
    TABLE_DIR
    / "transformer_three_class_classification_report.csv"
)

BINARY_REPORT_FILE = (
    TABLE_DIR
    / "transformer_binary_classification_report.csv"
)

THREE_CLASS_CONFUSION_FILE = (
    TABLE_DIR
    / "transformer_three_class_confusion_matrix.csv"
)

BINARY_CONFUSION_FILE = (
    TABLE_DIR
    / "transformer_binary_confusion_matrix.csv"
)

THREE_CLASS_CONFUSION_FIGURE = (
    FIGURE_DIR
    / "transformer_three_class_confusion_matrix.png"
)

BINARY_CONFUSION_FIGURE = (
    FIGURE_DIR
    / "transformer_binary_confusion_matrix.png"
)


# ---------------------------------------------------------------------------
# Model and inference settings
# ---------------------------------------------------------------------------

MODEL_NAME = (
    "cardiffnlp/"
    "twitter-roberta-base-sentiment-latest"
)

MAX_TOKEN_LENGTH = 512

# Conservative defaults for a local computer.
CPU_BATCH_SIZE = 8
MPS_BATCH_SIZE = 16
CUDA_BATCH_SIZE = 32

# Write progress after this many newly processed chunks.
CHECKPOINT_INTERVAL = 250

THREE_CLASS_LABELS = [
    "Negative",
    "Mixed",
    "Positive",
]

BINARY_LABELS = [
    "Not Positive",
    "Positive",
]


# ---------------------------------------------------------------------------
# Data loading and validation
# ---------------------------------------------------------------------------

def load_text_chunks() -> pd.DataFrame:
    """
    Load the prepared transformer text chunks.

    Returns
    -------
    pd.DataFrame
        One row per review text chunk.

    Raises
    ------
    FileNotFoundError
        If the prepared chunk file does not exist.
    ValueError
        If the file contains no records.
    """

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Transformer chunk file was not found: {INPUT_FILE}"
        )

    chunks = pd.read_csv(
        INPUT_FILE,
        keep_default_na=False,
    )

    if chunks.empty:
        raise ValueError(
            "The transformer chunk file contains no records."
        )

    return chunks


def validate_input_data(chunks: pd.DataFrame) -> None:
    """
    Validate the text-chunk dataset.

    Parameters
    ----------
    chunks : pd.DataFrame
        Transformer text-chunk dataset.

    Raises
    ------
    ValueError
        If required fields or values are missing.
    """

    required_columns = [
        "article_id",
        "chunk_id",
        "chunk_index",
        "manual_sentiment",
        "modeling_sentiment",
        "is_positive",
        "chunk_text",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in chunks.columns
    ]

    if missing_columns:
        raise ValueError(
            "The following required columns are missing: "
            + ", ".join(missing_columns)
        )

    duplicate_chunk_ids = (
        chunks["chunk_id"]
        .astype(str)
        .duplicated()
        .sum()
    )

    if duplicate_chunk_ids > 0:
        raise ValueError(
            f"{duplicate_chunk_ids} duplicated chunk IDs were found."
        )

    blank_chunk_count = (
        chunks["chunk_text"]
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    if blank_chunk_count > 0:
        raise ValueError(
            f"{blank_chunk_count} blank text chunks were found."
        )

    valid_manual_labels = set(THREE_CLASS_LABELS)

    observed_manual_labels = set(
        chunks["manual_sentiment"]
        .astype(str)
        .str.strip()
        .unique()
    )

    invalid_manual_labels = (
        observed_manual_labels
        - valid_manual_labels
    )

    if invalid_manual_labels:
        raise ValueError(
            "Unexpected manual sentiment labels were found: "
            + ", ".join(sorted(invalid_manual_labels))
        )


# ---------------------------------------------------------------------------
# Device and model loading
# ---------------------------------------------------------------------------

def select_device() -> torch.device:
    """
    Select the best available inference device.

    Returns
    -------
    torch.device
        CUDA, MPS, or CPU device.
    """

    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


def select_batch_size(
    device: torch.device,
) -> int:
    """
    Choose a conservative batch size for the active device.

    Parameters
    ----------
    device : torch.device
        Selected inference device.

    Returns
    -------
    int
        Batch size.
    """

    if device.type == "cuda":
        return CUDA_BATCH_SIZE

    if device.type == "mps":
        return MPS_BATCH_SIZE

    return CPU_BATCH_SIZE


def load_transformer(
    device: torch.device,
) -> tuple[
    AutoTokenizer,
    AutoModelForSequenceClassification,
]:
    """
    Load the pretrained tokenizer and model.

    Parameters
    ----------
    device : torch.device
        Device used for inference.

    Returns
    -------
    tuple
        Loaded tokenizer and sequence-classification model.
    """

    print(f"Loading tokenizer: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    print(f"Loading model: {MODEL_NAME}")

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(MODEL_NAME)
    )

    model.to(device)
    model.eval()

    return tokenizer, model


def get_model_label_mapping(
    model: AutoModelForSequenceClassification,
) -> dict[int, str]:
    """
    Read and standardize the model label mapping.

    Parameters
    ----------
    model : AutoModelForSequenceClassification
        Loaded sentiment model.

    Returns
    -------
    dict[int, str]
        Mapping from class index to Negative, Neutral, or Positive.

    Raises
    ------
    ValueError
        If the expected labels cannot be identified.
    """

    raw_mapping = model.config.id2label

    standardized_mapping: dict[int, str] = {}

    for raw_index, raw_label in raw_mapping.items():
        index = int(raw_index)

        label = str(raw_label).strip().lower()

        if "negative" in label:
            standardized_mapping[index] = "Negative"
        elif "neutral" in label:
            standardized_mapping[index] = "Neutral"
        elif "positive" in label:
            standardized_mapping[index] = "Positive"

    expected_labels = {
        "Negative",
        "Neutral",
        "Positive",
    }

    observed_labels = set(
        standardized_mapping.values()
    )

    if observed_labels != expected_labels:
        raise ValueError(
            "The model label configuration did not contain "
            "the expected Negative, Neutral, and Positive labels. "
            f"Observed mapping: {raw_mapping}"
        )

    return standardized_mapping


# ---------------------------------------------------------------------------
# Resume and checkpoint support
# ---------------------------------------------------------------------------

def load_existing_predictions() -> pd.DataFrame:
    """
    Load previously saved chunk predictions when available.

    Returns
    -------
    pd.DataFrame
        Existing chunk predictions or an empty dataframe.
    """

    if not CHUNK_PREDICTIONS_FILE.exists():
        return pd.DataFrame()

    existing = pd.read_csv(
        CHUNK_PREDICTIONS_FILE,
        keep_default_na=False,
    )

    required_prediction_columns = [
        "chunk_id",
        "transformer_negative_probability",
        "transformer_neutral_probability",
        "transformer_positive_probability",
        "transformer_chunk_label",
        "transformer_chunk_confidence",
    ]

    missing_columns = [
        column
        for column in required_prediction_columns
        if column not in existing.columns
    ]

    if missing_columns:
        print(
            "Existing prediction file is incomplete and will "
            "not be reused."
        )

        return pd.DataFrame()

    existing = existing.drop_duplicates(
        subset=["chunk_id"],
        keep="last",
    )

    return existing


def save_chunk_checkpoint(
    source_chunks: pd.DataFrame,
    prediction_records: list[dict[str, Any]],
) -> pd.DataFrame:
    """
    Merge and save accumulated chunk prediction records.

    Parameters
    ----------
    source_chunks : pd.DataFrame
        Original transformer chunk dataset.
    prediction_records : list[dict[str, Any]]
        Accumulated prediction results.

    Returns
    -------
    pd.DataFrame
        Full chunk prediction dataset saved to disk.
    """

    prediction_df = pd.DataFrame(
        prediction_records
    )

    prediction_df = prediction_df.drop_duplicates(
        subset=["chunk_id"],
        keep="last",
    )

    prediction_columns = [
        "chunk_id",
        "transformer_negative_probability",
        "transformer_neutral_probability",
        "transformer_positive_probability",
        "transformer_chunk_label",
        "transformer_chunk_confidence",
        "transformer_token_count",
        "transformer_was_truncated",
    ]

    source_without_predictions = (
        source_chunks
        .drop(
            columns=[
                column
                for column in prediction_columns
                if column in source_chunks.columns
            ],
            errors="ignore",
        )
    )

    combined = source_without_predictions.merge(
        prediction_df[prediction_columns],
        on="chunk_id",
        how="inner",
        validate="one_to_one",
    )

    combined = combined.sort_values(
        by=[
            "article_id",
            "chunk_index",
        ]
    ).reset_index(drop=True)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined.to_csv(
        CHUNK_PREDICTIONS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    return combined


# ---------------------------------------------------------------------------
# Transformer inference
# ---------------------------------------------------------------------------

def format_duration(
    seconds: float,
) -> str:
    """
    Format seconds as a compact elapsed-time string.

    Parameters
    ----------
    seconds : float
        Duration in seconds.

    Returns
    -------
    str
        Human-readable duration.
    """

    seconds = max(0, int(seconds))

    minutes, remaining_seconds = divmod(
        seconds,
        60,
    )

    hours, remaining_minutes = divmod(
        minutes,
        60,
    )

    if hours > 0:
        return (
            f"{hours}h "
            f"{remaining_minutes}m "
            f"{remaining_seconds}s"
        )

    if minutes > 0:
        return (
            f"{minutes}m "
            f"{remaining_seconds}s"
        )

    return f"{remaining_seconds}s"


def predict_chunk_batch(
    batch: pd.DataFrame,
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    device: torch.device,
    label_mapping: dict[int, str],
) -> list[dict[str, Any]]:
    """
    Run transformer inference on one chunk batch.

    Parameters
    ----------
    batch : pd.DataFrame
        Batch of chunk records.
    tokenizer : AutoTokenizer
        Pretrained tokenizer.
    model : AutoModelForSequenceClassification
        Pretrained sentiment model.
    device : torch.device
        Active inference device.
    label_mapping : dict[int, str]
        Model output class mapping.

    Returns
    -------
    list[dict[str, Any]]
        Chunk-level probability and prediction records.
    """

    texts = (
        batch["chunk_text"]
        .astype(str)
        .tolist()
    )

    untruncated_tokens = tokenizer(
        texts,
        add_special_tokens=True,
        truncation=False,
        padding=False,
    )

    original_token_counts = [
        len(token_ids)
        for token_ids in untruncated_tokens["input_ids"]
    ]

    encoded = tokenizer(
        texts,
        add_special_tokens=True,
        padding=True,
        truncation=True,
        max_length=MAX_TOKEN_LENGTH,
        return_tensors="pt",
    )

    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    with torch.inference_mode():
        outputs = model(**encoded)

        probabilities = softmax(
            outputs.logits,
            dim=-1,
        )

    probabilities = (
        probabilities
        .detach()
        .cpu()
        .numpy()
    )

    records: list[dict[str, Any]] = []

    for row_position, (_, row) in enumerate(
        batch.iterrows()
    ):
        class_probabilities = {
            label_mapping[class_index]:
                float(probabilities[row_position][class_index])
            for class_index in label_mapping
        }

        predicted_label = max(
            class_probabilities,
            key=class_probabilities.get,
        )

        predicted_confidence = (
            class_probabilities[predicted_label]
        )

        records.append(
            {
                "chunk_id": row["chunk_id"],
                "transformer_negative_probability":
                    class_probabilities["Negative"],
                "transformer_neutral_probability":
                    class_probabilities["Neutral"],
                "transformer_positive_probability":
                    class_probabilities["Positive"],
                "transformer_chunk_label":
                    predicted_label,
                "transformer_chunk_confidence":
                    predicted_confidence,
                "transformer_token_count":
                    original_token_counts[row_position],
                "transformer_was_truncated":
                    (
                        original_token_counts[row_position]
                        > MAX_TOKEN_LENGTH
                    ),
            }
        )

    return records


def run_transformer_inference(
    chunks: pd.DataFrame,
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    device: torch.device,
    label_mapping: dict[int, str],
) -> pd.DataFrame:
    """
    Run resumable batched inference on all review chunks.

    Parameters
    ----------
    chunks : pd.DataFrame
        Prepared review text chunks.
    tokenizer : AutoTokenizer
        Loaded tokenizer.
    model : AutoModelForSequenceClassification
        Loaded classification model.
    device : torch.device
        Active inference device.
    label_mapping : dict[int, str]
        Model output class mapping.

    Returns
    -------
    pd.DataFrame
        Complete chunk prediction dataset.
    """

    existing_predictions = (
        load_existing_predictions()
    )

    existing_records: list[dict[str, Any]] = []

    if not existing_predictions.empty:
        existing_records = (
            existing_predictions[
                [
                    "chunk_id",
                    "transformer_negative_probability",
                    "transformer_neutral_probability",
                    "transformer_positive_probability",
                    "transformer_chunk_label",
                    "transformer_chunk_confidence",
                    "transformer_token_count",
                    "transformer_was_truncated",
                ]
            ]
            .to_dict(orient="records")
        )

    completed_chunk_ids = {
        record["chunk_id"]
        for record in existing_records
    }

    remaining_chunks = chunks[
        ~chunks["chunk_id"].isin(
            completed_chunk_ids
        )
    ].copy()

    total_chunks = len(chunks)
    already_completed = len(completed_chunk_ids)

    if already_completed > 0:
        print(
            f"Resuming with {already_completed} "
            "previously processed chunks."
        )

    if remaining_chunks.empty:
        print(
            "All transformer chunks have already been processed."
        )

        return save_chunk_checkpoint(
            source_chunks=chunks,
            prediction_records=existing_records,
        )

    batch_size = select_batch_size(device)

    total_remaining = len(remaining_chunks)

    total_batches = math.ceil(
        total_remaining / batch_size
    )

    print(f"Inference device: {device.type}")
    print(f"Batch size: {batch_size}")
    print(f"Total chunks: {total_chunks}")
    print(f"Chunks remaining: {total_remaining}")
    print(f"Batches remaining: {total_batches}")

    all_records = list(existing_records)

    start_time = time.perf_counter()
    newly_processed = 0
    last_checkpoint_count = 0

    for batch_start in range(
        0,
        total_remaining,
        batch_size,
    ):
        batch_end = min(
            batch_start + batch_size,
            total_remaining,
        )

        batch = remaining_chunks.iloc[
            batch_start:batch_end
        ]

        batch_records = predict_chunk_batch(
            batch=batch,
            tokenizer=tokenizer,
            model=model,
            device=device,
            label_mapping=label_mapping,
        )

        all_records.extend(batch_records)

        newly_processed += len(batch_records)

        total_completed = (
            already_completed
            + newly_processed
        )

        elapsed_seconds = (
            time.perf_counter()
            - start_time
        )

        chunks_per_second = (
            newly_processed / elapsed_seconds
            if elapsed_seconds > 0
            else 0
        )

        remaining_count = (
            total_remaining
            - newly_processed
        )

        estimated_remaining_seconds = (
            remaining_count / chunks_per_second
            if chunks_per_second > 0
            else 0
        )

        if (
            newly_processed % 100 == 0
            or newly_processed == total_remaining
        ):
            progress_percent = (
                total_completed
                / total_chunks
                * 100
            )

            print(
                f"Processed {total_completed} / "
                f"{total_chunks} chunks "
                f"({progress_percent:.1f}%). "
                f"Elapsed: "
                f"{format_duration(elapsed_seconds)}. "
                f"Estimated remaining: "
                f"{format_duration(estimated_remaining_seconds)}."
            )

        if (
            newly_processed
            - last_checkpoint_count
            >= CHECKPOINT_INTERVAL
        ):
            save_chunk_checkpoint(
                source_chunks=chunks,
                prediction_records=all_records,
            )

            last_checkpoint_count = newly_processed

            print(
                f"Checkpoint saved after "
                f"{total_completed} total chunks."
            )

    completed_predictions = (
        save_chunk_checkpoint(
            source_chunks=chunks,
            prediction_records=all_records,
        )
    )

    return completed_predictions


# ---------------------------------------------------------------------------
# Article aggregation
# ---------------------------------------------------------------------------

def map_transformer_label_to_project_label(
    transformer_label: str,
) -> str:
    """
    Convert transformer labels to project labels.

    Parameters
    ----------
    transformer_label : str
        Negative, Neutral, or Positive.

    Returns
    -------
    str
        Negative, Mixed, or Positive.
    """

    if transformer_label == "Neutral":
        return "Mixed"

    return transformer_label


def convert_to_binary_sentiment(
    sentiment: str,
) -> str:
    """
    Convert three-class sentiment to binary sentiment.

    Parameters
    ----------
    sentiment : str
        Negative, Mixed, or Positive.

    Returns
    -------
    str
        Positive or Not Positive.
    """

    if sentiment == "Positive":
        return "Positive"

    return "Not Positive"


def aggregate_article_predictions(
    chunk_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate mean chunk probabilities to article-level predictions.

    Parameters
    ----------
    chunk_predictions : pd.DataFrame
        Complete chunk-level prediction dataset.

    Returns
    -------
    pd.DataFrame
        One row per article with aggregated probabilities.
    """

    metadata_columns = [
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
        "is_positive",
        "label_confidence",
    ]

    available_metadata_columns = [
        column
        for column in metadata_columns
        if column in chunk_predictions.columns
    ]

    article_metadata = (
        chunk_predictions[
            available_metadata_columns
        ]
        .drop_duplicates(
            subset=["article_id"],
            keep="first",
        )
    )

    probability_summary = (
        chunk_predictions
        .groupby(
            "article_id",
            as_index=False,
        )
        .agg(
            transformer_negative_probability=(
                "transformer_negative_probability",
                "mean",
            ),
            transformer_neutral_probability=(
                "transformer_neutral_probability",
                "mean",
            ),
            transformer_positive_probability=(
                "transformer_positive_probability",
                "mean",
            ),
            transformer_average_chunk_confidence=(
                "transformer_chunk_confidence",
                "mean",
            ),
            transformer_minimum_chunk_confidence=(
                "transformer_chunk_confidence",
                "min",
            ),
            transformer_maximum_chunk_confidence=(
                "transformer_chunk_confidence",
                "max",
            ),
            transformer_chunk_count=(
                "chunk_id",
                "count",
            ),
            transformer_truncated_chunk_count=(
                "transformer_was_truncated",
                "sum",
            ),
        )
    )

    article_predictions = (
        article_metadata
        .merge(
            probability_summary,
            on="article_id",
            how="inner",
            validate="one_to_one",
        )
    )

    probability_columns = {
        "Negative":
            "transformer_negative_probability",
        "Neutral":
            "transformer_neutral_probability",
        "Positive":
            "transformer_positive_probability",
    }

    article_predictions[
        "transformer_raw_label"
    ] = article_predictions.apply(
        lambda row: max(
            probability_columns,
            key=lambda label:
                row[probability_columns[label]],
        ),
        axis=1,
    )

    article_predictions[
        "transformer_sentiment"
    ] = (
        article_predictions[
            "transformer_raw_label"
        ]
        .apply(
            map_transformer_label_to_project_label
        )
    )

    article_predictions[
        "transformer_confidence"
    ] = article_predictions.apply(
        lambda row: max(
            row[
                "transformer_negative_probability"
            ],
            row[
                "transformer_neutral_probability"
            ],
            row[
                "transformer_positive_probability"
            ],
        ),
        axis=1,
    )

    article_predictions[
        "transformer_binary"
    ] = (
        article_predictions[
            "transformer_sentiment"
        ]
        .apply(convert_to_binary_sentiment)
    )

    article_predictions[
        "transformer_is_positive"
    ] = (
        article_predictions[
            "transformer_binary"
        ]
        .eq("Positive")
        .astype(int)
    )

    article_predictions[
        "three_class_correct"
    ] = (
        article_predictions[
            "manual_sentiment"
        ]
        .eq(
            article_predictions[
                "transformer_sentiment"
            ]
        )
    )

    article_predictions[
        "binary_correct"
    ] = (
        article_predictions[
            "modeling_sentiment"
        ]
        .eq(
            article_predictions[
                "transformer_binary"
            ]
        )
    )

    return article_predictions


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def calculate_metrics(
    actual: pd.Series,
    predicted: pd.Series,
    labels: list[str],
    evaluation_type: str,
    positive_scores: pd.Series | None = None,
    numeric_actual: pd.Series | None = None,
) -> dict[str, Any]:
    """
    Calculate classification metrics.

    Parameters
    ----------
    actual : pd.Series
        Ground-truth labels.
    predicted : pd.Series
        Predicted labels.
    labels : list[str]
        Ordered evaluation classes.
    evaluation_type : str
        Three Class or Binary.
    positive_scores : pd.Series, optional
        Continuous positive-class probabilities for ROC AUC.
    numeric_actual : pd.Series, optional
        Binary numeric labels for ROC AUC.

    Returns
    -------
    dict[str, Any]
        Model performance metrics.
    """

    roc_auc: float | None = None

    if (
        positive_scores is not None
        and numeric_actual is not None
    ):
        roc_auc = roc_auc_score(
            numeric_actual,
            positive_scores,
        )

    return {
        "model": "Transformer",
        "evaluation_type": evaluation_type,
        "accuracy": accuracy_score(
            actual,
            predicted,
        ),
        "balanced_accuracy":
            balanced_accuracy_score(
                actual,
                predicted,
            ),
        "macro_precision": precision_score(
            actual,
            predicted,
            labels=labels,
            average="macro",
            zero_division=0,
        ),
        "macro_recall": recall_score(
            actual,
            predicted,
            labels=labels,
            average="macro",
            zero_division=0,
        ),
        "macro_f1": f1_score(
            actual,
            predicted,
            labels=labels,
            average="macro",
            zero_division=0,
        ),
        "weighted_precision": precision_score(
            actual,
            predicted,
            labels=labels,
            average="weighted",
            zero_division=0,
        ),
        "weighted_recall": recall_score(
            actual,
            predicted,
            labels=labels,
            average="weighted",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            actual,
            predicted,
            labels=labels,
            average="weighted",
            zero_division=0,
        ),
        "cohen_kappa": cohen_kappa_score(
            actual,
            predicted,
            labels=labels,
        ),
        "roc_auc": roc_auc,
        "article_count": len(actual),
    }


def create_classification_report(
    actual: pd.Series,
    predicted: pd.Series,
    labels: list[str],
) -> pd.DataFrame:
    """
    Create a classification report dataframe.
    """

    report = classification_report(
        actual,
        predicted,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    return (
        pd.DataFrame(report)
        .transpose()
        .reset_index()
        .rename(columns={"index": "class"})
    )


def create_confusion_matrix_dataframe(
    actual: pd.Series,
    predicted: pd.Series,
    labels: list[str],
) -> pd.DataFrame:
    """
    Create a labeled confusion-matrix dataframe.
    """

    matrix = confusion_matrix(
        actual,
        predicted,
        labels=labels,
    )

    return pd.DataFrame(
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


def save_confusion_matrix_figure(
    matrix_df: pd.DataFrame,
    labels: list[str],
    title: str,
    output_file: Path,
) -> None:
    """
    Save a confusion matrix as a PNG figure.
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
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


# ---------------------------------------------------------------------------
# Output creation
# ---------------------------------------------------------------------------

def save_evaluation_outputs(
    article_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Save article predictions and evaluation outputs.

    Parameters
    ----------
    article_predictions : pd.DataFrame
        Article-level transformer predictions.

    Returns
    -------
    pd.DataFrame
        Summary metric table.
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

    article_predictions.to_csv(
        ARTICLE_PREDICTIONS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    three_class_metrics = calculate_metrics(
        actual=article_predictions[
            "manual_sentiment"
        ],
        predicted=article_predictions[
            "transformer_sentiment"
        ],
        labels=THREE_CLASS_LABELS,
        evaluation_type="Three Class",
    )

    binary_metrics = calculate_metrics(
        actual=article_predictions[
            "modeling_sentiment"
        ],
        predicted=article_predictions[
            "transformer_binary"
        ],
        labels=BINARY_LABELS,
        evaluation_type="Binary",
        positive_scores=article_predictions[
            "transformer_positive_probability"
        ],
        numeric_actual=article_predictions[
            "is_positive"
        ],
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

    three_class_report = (
        create_classification_report(
            actual=article_predictions[
                "manual_sentiment"
            ],
            predicted=article_predictions[
                "transformer_sentiment"
            ],
            labels=THREE_CLASS_LABELS,
        )
    )

    three_class_report.to_csv(
        THREE_CLASS_REPORT_FILE,
        index=False,
    )

    binary_report = (
        create_classification_report(
            actual=article_predictions[
                "modeling_sentiment"
            ],
            predicted=article_predictions[
                "transformer_binary"
            ],
            labels=BINARY_LABELS,
        )
    )

    binary_report.to_csv(
        BINARY_REPORT_FILE,
        index=False,
    )

    three_class_confusion = (
        create_confusion_matrix_dataframe(
            actual=article_predictions[
                "manual_sentiment"
            ],
            predicted=article_predictions[
                "transformer_sentiment"
            ],
            labels=THREE_CLASS_LABELS,
        )
    )

    three_class_confusion.to_csv(
        THREE_CLASS_CONFUSION_FILE,
    )

    binary_confusion = (
        create_confusion_matrix_dataframe(
            actual=article_predictions[
                "modeling_sentiment"
            ],
            predicted=article_predictions[
                "transformer_binary"
            ],
            labels=BINARY_LABELS,
        )
    )

    binary_confusion.to_csv(
        BINARY_CONFUSION_FILE,
    )

    save_confusion_matrix_figure(
        matrix_df=three_class_confusion,
        labels=THREE_CLASS_LABELS,
        title=(
            "Transformer Three-Class "
            "Confusion Matrix"
        ),
        output_file=(
            THREE_CLASS_CONFUSION_FIGURE
        ),
    )

    save_confusion_matrix_figure(
        matrix_df=binary_confusion,
        labels=BINARY_LABELS,
        title=(
            "Transformer Binary "
            "Confusion Matrix"
        ),
        output_file=BINARY_CONFUSION_FIGURE,
    )

    return metrics_df


def print_results_summary(
    chunk_predictions: pd.DataFrame,
    article_predictions: pd.DataFrame,
    metrics_df: pd.DataFrame,
    device: torch.device,
) -> None:
    """
    Print key transformer results.
    """

    three_class_metrics = (
        metrics_df[
            metrics_df["evaluation_type"]
            == "Three Class"
        ]
        .iloc[0]
    )

    binary_metrics = (
        metrics_df[
            metrics_df["evaluation_type"]
            == "Binary"
        ]
        .iloc[0]
    )

    truncated_chunk_count = int(
        chunk_predictions[
            "transformer_was_truncated"
        ].sum()
    )

    print()
    print(
        "Transformer sentiment analysis "
        "completed successfully."
    )
    print(f"Inference device: {device.type}")
    print(
        f"Chunks analyzed: "
        f"{len(chunk_predictions)}"
    )
    print(
        f"Articles analyzed: "
        f"{len(article_predictions)}"
    )
    print(
        f"Chunks truncated: "
        f"{truncated_chunk_count}"
    )

    print()
    print("Manual sentiment distribution:")
    print(
        article_predictions[
            "manual_sentiment"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print("Transformer sentiment distribution:")
    print(
        article_predictions[
            "transformer_sentiment"
        ]
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
    print(
        f"- Chunk predictions: "
        f"{CHUNK_PREDICTIONS_FILE}"
    )
    print(
        f"- Article predictions: "
        f"{ARTICLE_PREDICTIONS_FILE}"
    )
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


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Run the complete transformer sentiment workflow.
    """

    chunks = load_text_chunks()

    validate_input_data(chunks)

    device = select_device()

    tokenizer, model = load_transformer(
        device=device
    )

    label_mapping = get_model_label_mapping(
        model=model
    )

    print(
        f"Model label mapping: "
        f"{label_mapping}"
    )

    chunk_predictions = (
        run_transformer_inference(
            chunks=chunks,
            tokenizer=tokenizer,
            model=model,
            device=device,
            label_mapping=label_mapping,
        )
    )

    article_predictions = (
        aggregate_article_predictions(
            chunk_predictions
        )
    )

    metrics_df = save_evaluation_outputs(
        article_predictions
    )

    print_results_summary(
        chunk_predictions=chunk_predictions,
        article_predictions=article_predictions,
        metrics_df=metrics_df,
        device=device,
    )


if __name__ == "__main__":
    main()