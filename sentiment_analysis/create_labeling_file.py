"""
Create a manual sentiment-labeling file from the processed Pinkbike review dataset.

Purpose:
This script prepares one row per review article for manual sentiment labeling.
The completed labels will serve as the ground-truth dataset used to evaluate
VADER, transformer-based sentiment analysis, and supervised machine-learning
models.

Label definitions:
- Positive: Strengths clearly outweigh weaknesses and the overall recommendation
  is favorable.
- Mixed: The review contains substantial praise and criticism, significant
  caveats, or no clearly dominant positive or negative conclusion.
- Negative: Weaknesses clearly outweigh strengths and the overall recommendation
  is unfavorable.

AI Use:
AI tools were used to assist with code design, documentation, and workflow planning.
"""

from pathlib import Path

import pandas as pd


# Project file paths

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = BASE_DIR / "data" / "processed" / "pinkbike_reviews.csv"

OUTPUT_DIR = BASE_DIR / "data" / "modeling"
OUTPUT_FILE = OUTPUT_DIR / "manual_sentiment_labels.csv"


# Columns retained from the processed review dataset

SOURCE_COLUMNS = [
    "source_url",
    "title",
    "author",
    "publish_date",
    "brand",
    "product_name",
    "product_category",
    "product_subcategory",
    "review_type",
    "article_text_length",
    "article_text",
]


# Columns completed manually during labeling

LABEL_COLUMNS = [
    "manual_sentiment",
    "label_confidence",
    "label_notes",
]


def load_reviews() -> pd.DataFrame:
    """
    Load the processed Pinkbike review dataset.

    Returns
    -------
    pd.DataFrame
        Processed review-level dataset.
    """

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input dataset was not found: {INPUT_FILE}"
        )

    reviews = pd.read_csv(INPUT_FILE)

    if reviews.empty:
        raise ValueError("The processed review dataset contains no rows.")

    return reviews


def validate_required_columns(reviews: pd.DataFrame) -> None:
    """
    Confirm that the columns required for labeling exist.

    Parameters
    ----------
    reviews : pd.DataFrame
        Processed review dataset.

    Raises
    ------
    ValueError
        If one or more required columns are missing.
    """

    required_columns = [
        "source_url",
        "title",
        "article_text",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in reviews.columns
    ]

    if missing_columns:
        raise ValueError(
            "The following required columns are missing from the dataset: "
            + ", ".join(missing_columns)
        )


def create_article_id(reviews: pd.DataFrame) -> pd.Series:
    """
    Create a stable sequential article identifier.

    The identifier is based on the sorted article URL order so it remains
    reproducible as long as the underlying dataset does not change.

    Parameters
    ----------
    reviews : pd.DataFrame
        Review dataset sorted into labeling order.

    Returns
    -------
    pd.Series
        Article identifiers formatted as PB0001, PB0002, and so on.
    """

    return pd.Series(
        [
            f"PB{number:04d}"
            for number in range(1, len(reviews) + 1)
        ],
        index=reviews.index,
    )


def prepare_labeling_dataset(reviews: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the processed dataset into a manual-labeling dataset.

    Parameters
    ----------
    reviews : pd.DataFrame
        Processed Pinkbike review dataset.

    Returns
    -------
    pd.DataFrame
        One row per article with metadata, full article text, and blank
        manual-labeling fields.
    """

    available_columns = [
        column
        for column in SOURCE_COLUMNS
        if column in reviews.columns
    ]

    labeling_df = reviews[available_columns].copy()

    # Remove rows without usable article text
    labeling_df["article_text"] = (
        labeling_df["article_text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    labeling_df = labeling_df[
        labeling_df["article_text"] != ""
    ].copy()

    # Remove duplicated articles using URL as the primary identifier
    labeling_df = labeling_df.drop_duplicates(
        subset=["source_url"],
        keep="first",
    )

    # Sort records into a repeatable order
    sort_columns = [
        column
        for column in [
            "publish_date",
            "title",
            "source_url",
        ]
        if column in labeling_df.columns
    ]

    labeling_df = labeling_df.sort_values(
        by=sort_columns,
        na_position="last",
    ).reset_index(drop=True)

    # Create stable article IDs after sorting
    labeling_df.insert(
        0,
        "article_id",
        create_article_id(labeling_df),
    )

    # Add useful text-length information if it does not already exist
    labeling_df["article_word_count"] = (
        labeling_df["article_text"]
        .str.split()
        .str.len()
    )

    # Add blank fields for manual labeling
    for column in LABEL_COLUMNS:
        labeling_df[column] = ""

    # Track labeling progress without altering the sentiment label
    labeling_df["label_status"] = "Not Labeled"

    # Place manual-labeling fields before the full article text
    preferred_order = [
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
        "article_text_length",
        "article_word_count",
        "manual_sentiment",
        "label_confidence",
        "label_notes",
        "label_status",
        "article_text",
    ]

    final_columns = [
        column
        for column in preferred_order
        if column in labeling_df.columns
    ]

    return labeling_df[final_columns]


def save_labeling_dataset(labeling_df: pd.DataFrame) -> None:
    """
    Save the manual-labeling dataset.

    Existing completed labels are preserved when possible. If the output
    already exists, this function matches records by URL and carries the
    existing manual fields into the regenerated file.

    Parameters
    ----------
    labeling_df : pd.DataFrame
        Newly prepared manual-labeling dataset.
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_FILE.exists():
        existing_labels = pd.read_csv(
            OUTPUT_FILE,
            keep_default_na=False,
        )

        preserved_columns = [
            "source_url",
            "manual_sentiment",
            "label_confidence",
            "label_notes",
            "label_status",
        ]

        available_preserved_columns = [
            column
            for column in preserved_columns
            if column in existing_labels.columns
        ]

        if "url" in available_preserved_columns:
            existing_labels = existing_labels[
                available_preserved_columns
            ].drop_duplicates(
                subset=["source_url"],
                keep="last",
            )

            labeling_df = labeling_df.drop(
                columns=[
                    column
                    for column in LABEL_COLUMNS + ["label_status"]
                    if column in labeling_df.columns
                ],
            ).merge(
                existing_labels,
                on="source_url",
                how="left",
            )

            for column in LABEL_COLUMNS:
                if column not in labeling_df.columns:
                    labeling_df[column] = ""

                labeling_df[column] = (
                    labeling_df[column]
                    .fillna("")
                    .astype(str)
                )

            if "label_status" not in labeling_df.columns:
                labeling_df["label_status"] = ""

            labeling_df["label_status"] = labeling_df.apply(
                lambda row: (
                    "Labeled"
                    if row["manual_sentiment"].strip()
                    else "Not Labeled"
                ),
                axis=1,
            )

            preferred_order = [
                "article_id",
                "url",
                "title",
                "author",
                "publish_date",
                "brand",
                "product_name",
                "product_category",
                "product_subcategory",
                "review_type",
                "article_text_length",
                "article_word_count",
                "manual_sentiment",
                "label_confidence",
                "label_notes",
                "label_status",
                "article_text",
            ]

            labeling_df = labeling_df[
                [
                    column
                    for column in preferred_order
                    if column in labeling_df.columns
                ]
            ]

    labeling_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )


def print_summary(labeling_df: pd.DataFrame) -> None:
    """
    Print a brief summary of the generated labeling file.

    Parameters
    ----------
    labeling_df : pd.DataFrame
        Final labeling dataset.
    """

    labeled_count = (
        labeling_df["manual_sentiment"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
        .sum()
    )

    print("Manual sentiment-labeling file created successfully.")
    print(f"Input file: {INPUT_FILE}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Articles available for labeling: {len(labeling_df)}")
    print(f"Articles already labeled: {labeled_count}")
    print(f"Articles remaining: {len(labeling_df) - labeled_count}")


def main() -> None:
    """
    Run the manual-labeling file creation workflow.
    """

    reviews = load_reviews()

    validate_required_columns(reviews)

    labeling_df = prepare_labeling_dataset(reviews)

    save_labeling_dataset(labeling_df)

    # Reload the saved file so the summary reflects preserved labels
    saved_labeling_df = pd.read_csv(
        OUTPUT_FILE,
        keep_default_na=False,
    )

    print_summary(saved_labeling_df)


if __name__ == "__main__":
    main()