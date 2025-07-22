# G004: Logging f-strings temporarily allowed for development
import logging
import re

import nltk
import numpy as np
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature engineering for journal entries."""

    def __init__(self) -> None:
        """Initialize feature engineer."""
        # Ensure NLTK resources are downloaded
        try:
            nltk.download("vader_lexicon", quiet=True)
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
        except Exception:
            logger.error(
                "Failed to initialize sentiment analyzer: {e}",
                extra={"format_args": True},
            )
            self.sentiment_analyzer = None

    def extract_basic_features(
        self, df: pd.DataFrame, text_column: str = "content"
    ) -> pd.DataFrame:
        """Extract basic statistical features from text.

        Args:
            df: DataFrame containing journal entries
            text_column: Name of column containing entry content

        Returns:
            DataFrame with basic features added

        """
        df = df.copy()

        # Ensure text column is string type
        df[text_column] = df[text_column].astype(str)

        # Character count
        df["char_count"] = df[text_column].apply(len)

        # Word count
        df["word_count"] = df[text_column].apply(lambda x: len(x.split()))

        # Average word length
        df["avg_word_length"] = df[text_column].apply(
            lambda x: np.mean([len(word) for word in x.split()])
            if len(x.split()) > 0
            else 0
        )

        # Sentence count
        df["sentence_count"] = df[text_column].apply(
            lambda x: len(re.split(r"[.!?]+", x)) - 1
        )

        # Words per sentence
        df["words_per_sentence"] = df.apply(
            lambda row: row["word_count"] / row["sentence_count"]
            if row["sentence_count"] > 0
            else 0,
            axis=1,
        )

        # Unique word count
        df["unique_word_count"] = df[text_column].apply(lambda x: len(set(x.split())))

        # Lexical diversity (unique words / total words)
        df["lexical_diversity"] = df.apply(
            lambda row: row["unique_word_count"] / row["word_count"]
            if row["word_count"] > 0
            else 0,
            axis=1,
        )

        return df

    def extract_sentiment_features(
        self, df: pd.DataFrame, text_column: str = "content"
    ) -> pd.DataFrame:
        """Extract sentiment features from text using NLTK's VADER.

        Args:
            df: DataFrame containing journal entries
            text_column: Name of column containing entry content

        Returns:
            DataFrame with sentiment features added

        """
        if self.sentiment_analyzer is None:
            logger.warning(
                "Sentiment analyzer not available. Skipping sentiment feature extraction."
            )
            return df

        df = df.copy()

        # Ensure text column is string type
        df[text_column] = df[text_column].astype(str)

        logger.info("Extracting sentiment features")

        # Apply sentiment analyzer to get scores
        sentiments = df[text_column].apply(self.sentiment_analyzer.polarity_scores)

        # Extract sentiment components into separate columns
        df["sentiment_negative"] = sentiments.apply(lambda x: x["neg"])
        df["sentiment_neutral"] = sentiments.apply(lambda x: x["neu"])
        df["sentiment_positive"] = sentiments.apply(lambda x: x["pos"])
        df["sentiment_compound"] = sentiments.apply(lambda x: x["compound"])

        # Create sentiment category based on compound score
        df["sentiment_category"] = df["sentiment_compound"].apply(
            lambda score: "positive"
            if score > 0.05
            else ("negative" if score < -0.05 else "neutral")
        )

        return df

    def extract_topic_features(
        self,
        df: pd.DataFrame,
        text_column: str = "content",
        n_topics: int = 10,
        n_top_words: int = 5,
    ) -> pd.DataFrame:
        """Extract topic-related features using TF-IDF and SVD.

        Args:
            df: DataFrame containing journal entries
            text_column: Name of column containing entry content
            n_topics: Number of topics to extract
            n_top_words: Number of top words to include per topic

        Returns:
            DataFrame with topic features added

        """
        df = df.copy()

        # Ensure text column is string type
        df[text_column] = df[text_column].astype(str)

        logger.info(
            "Extracting {n_topics} topic features using TF-IDF and SVD",
            extra={"format_args": True},
        )

        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")

        # Transform texts to TF-IDF matrix
        tfidf_matrix = vectorizer.fit_transform(df[text_column])

        # Get feature names (words)
        feature_names = vectorizer.get_feature_names_out()

        # Apply SVD to reduce dimensions and extract topics
        svd = TruncatedSVD(n_components=n_topics, random_state=42)
        topic_matrix = svd.fit_transform(tfidf_matrix)

        # Add topic scores as features
        for i in range(n_topics):
            df[f"topic_{i+1}_score"] = topic_matrix[:, i]

        # Get top words for each topic
        topic_words = {}
        for i, comp in enumerate(svd.components_):
            # Get top word indices for this topic
            top_word_indices = comp.argsort()[: -n_top_words - 1 : -1]
            # Get the actual words
            top_words = [feature_names[idx] for idx in top_word_indices]
            topic_words[f"topic_{i+1}"] = top_words

        # Convert topics to DataFrame for easier inspection
        topics_df = pd.DataFrame(topic_words)

        # Assign dominant topic to each document
        df["dominant_topic"] = np.argmax(topic_matrix, axis=1) + 1

        logger.info(
            "Extracted {n_topics} topics from {len(df)} documents",
            extra={"format_args": True},
        )

        return df, topics_df

    def extract_time_features(
        self, df: pd.DataFrame, timestamp_column: str = "created_at"
    ) -> pd.DataFrame:
        """Extract time-related features from timestamp.

        Args:
            df: DataFrame containing journal entries
            timestamp_column: Name of column containing timestamps

        Returns:
            DataFrame with time features added

        """
        df = df.copy()

        if timestamp_column not in df.columns:
            logger.warning(
                f"Timestamp column '{timestamp_column}' not found in DataFrame"
            )
            return df

        # Try to ensure timestamp column is datetime type
        try:
            df[timestamp_column] = pd.to_datetime(df[timestamp_column])
        except Exception:
            logger.error(
                "Failed to convert '{timestamp_column}' to datetime: {e}",
                extra={"format_args": True},
            )
            return df

        logger.info("Extracting time features")

        # Extract basic time components
        df["year"] = df[timestamp_column].dt.year
        df["month"] = df[timestamp_column].dt.month
        df["day"] = df[timestamp_column].dt.day
        df["day_of_week"] = df[timestamp_column].dt.dayofweek
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        df["hour"] = df[timestamp_column].dt.hour

        # Time of day features
        df["time_of_day"] = pd.cut(
            df["hour"],
            bins=[0, 6, 12, 18, 24],
            labels=["night", "morning", "afternoon", "evening"],
            right=False,
        )

        return df

    def extract_all_features(
        self,
        df: pd.DataFrame,
        text_column: str = "content",
        timestamp_column: str = "created_at",
        extract_topics: bool = True,
    ) -> pd.DataFrame:
        """Extract all features from journal entries.

        Args:
            df: DataFrame containing journal entries
            text_column: Name of column containing entry content
            timestamp_column: Name of column containing timestamps
            extract_topics: Whether to extract topic features

        Returns:
            DataFrame with all features added

        """
        logger.info(
            "Extracting all features for {len(df)} journal entries",
            extra={"format_args": True},
        )

        # Extract basic text features
        df = self.extract_basic_features(df, text_column)

        # Extract sentiment features
        df = self.extract_sentiment_features(df, text_column)

        # Extract time features
        df = self.extract_time_features(df, timestamp_column)

        # Extract topic features if requested
        if extract_topics:
            df, topics_df = self.extract_topic_features(df, text_column)
            return df, topics_df

        return df
