"""
Prepare manually labeled review articles for transformer sentiment analysis.

The processed article text does not retain dependable paragraph boundaries.
This script therefore:

1. Loads the manually labeled article dataset.
2. Splits each article into sentences.
3. Groups neighboring sentences into coherent, token-safe text chunks.
4. Preserves article metadata and manual sentiment labels.
5. Saves one row per text chunk for transformer inference.

The chunks are not treated as manually labeled observations. Manual sentiment
remains an article-level ground-truth label and is included only so the chunks
can later be aggregated and evaluated by article.

AI Use:
AI tools were used to assist with code design, documentation, and workflow
planning.
"""

from pathlib import Path
import re

import pandas as pd


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

OUTPUT_FILE = OUTPUT_DIR / "review_text_chunks.csv"

ARTICLE_SUMMARY_FILE = OUTPUT_DIR / "review_chunk_summary.csv"


# ---------------------------------------------------------------------------
# Chunking settings
# ---------------------------------------------------------------------------

# The target length encourages coherent but reasonably compact chunks.
TARGET_CHUNK_WORDS = 180

# This remains safely below the 512-token limit for most English text.
MAX_CHUNK_WORDS = 250

# Very small trailing chunks are merged with the preceding chunk when possible.
MIN_CHUNK_WORDS = 30


# ---------------------------------------------------------------------------
# Dataset fields
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "article_id",
    "source_url",
    "title",
    "manual_sentiment",
    "article_text",
]

METADATA_COLUMNS = [
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


def load_labeled_reviews() -> pd.DataFrame:
    """
    Load the manually labeled review dataset.

    Returns
    -------
    pd.DataFrame
        One row per manually labeled review article.

    Raises
    ------
    FileNotFoundError
        If the labeling file does not exist.
    ValueError
        If the file contains no records.
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
    Confirm that required fields and article text are available.

    Parameters
    ----------
    reviews : pd.DataFrame
        Manual sentiment-labeling dataset.

    Raises
    ------
    ValueError
        If required columns, text, labels, or article IDs are missing.
    """

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
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
            f"{blank_text_count} articles contain blank article text."
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
            f"{blank_label_count} articles are missing manual labels."
        )


def prepare_modeling_targets(
    reviews: pd.DataFrame,
) -> pd.DataFrame:
    """
    Standardize manual labels and recreate binary modeling targets.

    Parameters
    ----------
    reviews : pd.DataFrame
        Manually labeled article dataset.

    Returns
    -------
    pd.DataFrame
        Dataset containing consistent three-class and binary targets.
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

    if prepared["modeling_sentiment"].isna().any():
        invalid_labels = sorted(
            prepared.loc[
                prepared["modeling_sentiment"].isna(),
                "manual_sentiment",
            ].unique()
        )

        raise ValueError(
            "Unexpected manual sentiment labels were found: "
            + ", ".join(invalid_labels)
        )

    prepared["is_positive"] = (
        prepared["manual_sentiment"]
        .eq("Positive")
        .astype(int)
    )

    return prepared


def normalize_text(text: str) -> str:
    """
    Normalize whitespace while retaining punctuation.

    Parameters
    ----------
    text : str
        Raw article text.

    Returns
    -------
    str
        Cleaned article text.
    """

    normalized = str(text)

    normalized = normalized.replace(
        "\u00a0",
        " ",
    )

    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    )

    return normalized.strip()


def split_into_sentences(text: str) -> list[str]:
    """
    Split normalized article text into approximate sentences.

    A regular-expression splitter is used to avoid requiring an external
    sentence-tokenizer download. It identifies sentence boundaries after
    periods, question marks, and exclamation points followed by likely
    sentence-starting text.

    Parameters
    ----------
    text : str
        Normalized article text.

    Returns
    -------
    list[str]
        Ordered sentence-like text units.
    """

    if not text:
        return []

    sentence_boundary_pattern = re.compile(
        r"(?<=[.!?])\s+(?=[A-Z0-9“\"'])"
    )

    raw_sentences = sentence_boundary_pattern.split(text)

    sentences = [
        sentence.strip()
        for sentence in raw_sentences
        if sentence.strip()
    ]

    return sentences


def split_long_sentence(
    sentence: str,
    max_words: int,
) -> list[str]:
    """
    Split an unusually long sentence into word-bounded segments.

    Parameters
    ----------
    sentence : str
        Sentence exceeding the maximum chunk length.
    max_words : int
        Maximum words allowed in each segment.

    Returns
    -------
    list[str]
        Ordered text segments.
    """

    words = sentence.split()

    return [
        " ".join(
            words[start_index:start_index + max_words]
        )
        for start_index in range(
            0,
            len(words),
            max_words,
        )
    ]


def build_article_chunks(
    sentences: list[str],
) -> list[str]:
    """
    Group neighboring sentences into coherent text chunks.

    Parameters
    ----------
    sentences : list[str]
        Ordered article sentences.

    Returns
    -------
    list[str]
        Ordered sentence-aware text chunks.
    """

    chunks: list[str] = []

    current_sentences: list[str] = []
    current_word_count = 0

    for sentence in sentences:
        sentence_word_count = len(sentence.split())

        # Handle individual sentences longer than the maximum chunk size.
        if sentence_word_count > MAX_CHUNK_WORDS:
            if current_sentences:
                chunks.append(
                    " ".join(current_sentences)
                )

                current_sentences = []
                current_word_count = 0

            long_sentence_segments = split_long_sentence(
                sentence=sentence,
                max_words=MAX_CHUNK_WORDS,
            )

            chunks.extend(long_sentence_segments)

            continue

        proposed_word_count = (
            current_word_count
            + sentence_word_count
        )

        if (
            current_sentences
            and proposed_word_count > MAX_CHUNK_WORDS
        ):
            chunks.append(
                " ".join(current_sentences)
            )

            current_sentences = [sentence]
            current_word_count = sentence_word_count

            continue

        current_sentences.append(sentence)
        current_word_count = proposed_word_count

        # Finish a chunk once the target has been reached and the current
        # sentence provides a natural stopping point.
        if current_word_count >= TARGET_CHUNK_WORDS:
            chunks.append(
                " ".join(current_sentences)
            )

            current_sentences = []
            current_word_count = 0

    if current_sentences:
        final_chunk = " ".join(current_sentences)

        final_word_count = len(final_chunk.split())

        if (
            chunks
            and final_word_count < MIN_CHUNK_WORDS
        ):
            combined_chunk = (
                chunks[-1]
                + " "
                + final_chunk
            )

            combined_word_count = len(
                combined_chunk.split()
            )

            if combined_word_count <= MAX_CHUNK_WORDS:
                chunks[-1] = combined_chunk
            else:
                chunks.append(final_chunk)
        else:
            chunks.append(final_chunk)

    return [
        chunk.strip()
        for chunk in chunks
        if chunk.strip()
    ]


def create_chunk_dataset(
    reviews: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert article-level data into one row per text chunk.

    Parameters
    ----------
    reviews : pd.DataFrame
        Prepared manually labeled article dataset.

    Returns
    -------
    pd.DataFrame
        Text-chunk dataset for transformer inference.
    """

    available_metadata_columns = [
        column
        for column in METADATA_COLUMNS
        if column in reviews.columns
    ]

    chunk_records: list[dict[str, object]] = []

    total_articles = len(reviews)

    for article_number, (_, article) in enumerate(
        reviews.iterrows(),
        start=1,
    ):
        cleaned_text = normalize_text(
            article["article_text"]
        )

        sentences = split_into_sentences(cleaned_text)

        chunks = build_article_chunks(sentences)

        if not chunks:
            raise ValueError(
                "No text chunks were generated for article "
                f"{article['article_id']}."
            )

        total_chunks = len(chunks)

        for chunk_index, chunk_text in enumerate(
            chunks,
            start=1,
        ):
            record = {
                column: article[column]
                for column in available_metadata_columns
            }

            record.update(
                {
                    "chunk_id": (
                        f"{article['article_id']}"
                        f"_C{chunk_index:03d}"
                    ),
                    "chunk_index": chunk_index,
                    "total_chunks": total_chunks,
                    "chunk_word_count": len(
                        chunk_text.split()
                    ),
                    "chunk_character_count": len(
                        chunk_text
                    ),
                    "chunk_text": chunk_text,
                }
            )

            chunk_records.append(record)

        if (
            article_number % 25 == 0
            or article_number == total_articles
        ):
            print(
                "Prepared "
                f"{article_number} of {total_articles} articles."
            )

    chunks_df = pd.DataFrame(chunk_records)

    preferred_column_order = [
        "article_id",
        "chunk_id",
        "chunk_index",
        "total_chunks",
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
        "chunk_word_count",
        "chunk_character_count",
        "chunk_text",
    ]

    final_columns = [
        column
        for column in preferred_column_order
        if column in chunks_df.columns
    ]

    return chunks_df[final_columns]


def create_article_summary(
    chunks_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create an article-level summary of generated chunks.

    Parameters
    ----------
    chunks_df : pd.DataFrame
        Transformer text-chunk dataset.

    Returns
    -------
    pd.DataFrame
        Article-level chunk quality summary.
    """

    summary = (
        chunks_df
        .groupby(
            [
                "article_id",
                "title",
                "manual_sentiment",
            ],
            as_index=False,
        )
        .agg(
            chunk_count=(
                "chunk_id",
                "count",
            ),
            total_chunk_words=(
                "chunk_word_count",
                "sum",
            ),
            minimum_chunk_words=(
                "chunk_word_count",
                "min",
            ),
            average_chunk_words=(
                "chunk_word_count",
                "mean",
            ),
            maximum_chunk_words=(
                "chunk_word_count",
                "max",
            ),
        )
    )

    summary["average_chunk_words"] = (
        summary["average_chunk_words"]
        .round(2)
    )

    return summary


def validate_chunk_dataset(
    reviews: pd.DataFrame,
    chunks_df: pd.DataFrame,
) -> None:
    """
    Validate the generated transformer text chunks.

    Parameters
    ----------
    reviews : pd.DataFrame
        Original article-level dataset.
    chunks_df : pd.DataFrame
        Generated text-chunk dataset.

    Raises
    ------
    ValueError
        If articles are missing, chunk IDs are duplicated, or chunks exceed
        the configured maximum.
    """

    expected_article_ids = set(
        reviews["article_id"]
    )

    chunk_article_ids = set(
        chunks_df["article_id"]
    )

    missing_articles = (
        expected_article_ids
        - chunk_article_ids
    )

    if missing_articles:
        raise ValueError(
            "No chunks were generated for article IDs: "
            + ", ".join(sorted(missing_articles))
        )

    duplicated_chunk_ids = (
        chunks_df["chunk_id"]
        .duplicated()
        .sum()
    )

    if duplicated_chunk_ids > 0:
        raise ValueError(
            f"{duplicated_chunk_ids} duplicated chunk IDs were found."
        )

    oversized_chunks = (
        chunks_df["chunk_word_count"]
        > MAX_CHUNK_WORDS
    ).sum()

    if oversized_chunks > 0:
        raise ValueError(
            f"{oversized_chunks} chunks exceed "
            f"{MAX_CHUNK_WORDS} words."
        )

    blank_chunks = (
        chunks_df["chunk_text"]
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    if blank_chunks > 0:
        raise ValueError(
            f"{blank_chunks} blank chunks were generated."
        )


def save_outputs(
    chunks_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> None:
    """
    Save transformer text chunks and article-level QA summary.

    Parameters
    ----------
    chunks_df : pd.DataFrame
        One row per transformer text chunk.
    summary_df : pd.DataFrame
        One row per article with chunk statistics.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    chunks_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    summary_df.to_csv(
        ARTICLE_SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig",
    )


def print_summary(
    reviews: pd.DataFrame,
    chunks_df: pd.DataFrame,
) -> None:
    """
    Print text-preparation results.

    Parameters
    ----------
    reviews : pd.DataFrame
        Article-level input dataset.
    chunks_df : pd.DataFrame
        Generated text-chunk dataset.
    """

    print()
    print("Transformer text preparation completed successfully.")
    print(f"Articles processed: {len(reviews)}")
    print(f"Text chunks created: {len(chunks_df)}")
    print(
        "Average chunks per article: "
        f"{len(chunks_df) / len(reviews):.2f}"
    )
    print(
        "Average words per chunk: "
        f"{chunks_df['chunk_word_count'].mean():.2f}"
    )
    print(
        "Minimum words in a chunk: "
        f"{chunks_df['chunk_word_count'].min()}"
    )
    print(
        "Maximum words in a chunk: "
        f"{chunks_df['chunk_word_count'].max()}"
    )

    print()
    print("Saved outputs:")
    print(f"- Transformer chunks: {OUTPUT_FILE}")
    print(f"- Article chunk summary: {ARTICLE_SUMMARY_FILE}")


def main() -> None:
    """
    Run the complete transformer text-preparation workflow.
    """

    reviews = load_labeled_reviews()

    validate_input_data(reviews)

    reviews = prepare_modeling_targets(reviews)

    chunks_df = create_chunk_dataset(reviews)

    validate_chunk_dataset(
        reviews=reviews,
        chunks_df=chunks_df,
    )

    summary_df = create_article_summary(chunks_df)

    save_outputs(
        chunks_df=chunks_df,
        summary_df=summary_df,
    )

    print_summary(
        reviews=reviews,
        chunks_df=chunks_df,
    )


if __name__ == "__main__":
    main()