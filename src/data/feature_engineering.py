            # Get the actual words
            # Get top word indices for this topic
        # Add topic scores as features
        # Apply SVD to reduce dimensions and extract topics
        # Apply sentiment analyzer to get scores
        # Assign dominant topic to each document
        # Average word length
        # Character count
        # Convert topics to DataFrame for easier inspection
        # Create TF-IDF vectorizer
        # Create sentiment category based on compound score
        # Ensure NLTK resources are downloaded
        # Ensure text column is string type
        # Ensure text column is string type
        # Ensure text column is string type
        # Extract basic text features
        # Extract basic time components
        # Extract sentiment components into separate columns
        # Extract sentiment features
        # Extract time features
        # Extract topic features if requested
        # Get feature names (words)
        # Get top words for each topic
        # Lexical diversity (unique words / total words)
        # Sentence count
        # Time of day features
        # Transform texts to TF-IDF matrix
        # Try to ensure timestamp column is datetime type
        # Unique word count
        # Word count
        # Words per sentence
# Configure logging
# G004: Logging f-strings temporarily allowed for development
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import nltk
import numpy as np
import pandas as pd
import re




logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature engineering for journal entries."""

    def __init__(self) -> None:
        """Initialize feature engineer."""
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

        df[text_column] = df[text_column].astype(str)

        df["char_count"] = df[text_column].apply(len)

        df["word_count"] = df[text_column].apply(lambda x: len(x.split()))

        df["avg_word_length"] = df[text_column].apply(
            lambda x: np.mean(
                              [len(word) for word in x.split()]) if len(x.split()) > 0 else 0
        )

        df["sentence_count"] = df[text_column].apply(
                                                     lambda x: len(re.split(r"[.!?]+",
                                                     x)) - 1
                                                    )

        df["words_per_sentence"] = df.apply(
            lambda row: row["word_count"] / row["sentence_count"]
            if row["sentence_count"] > 0
            else 0,
            axis=1,
        )

        df["unique_word_count"] = df[text_column].apply(lambda x: len(set(x.split())))

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

        df[text_column] = df[text_column].astype(str)

        logger.info("Extracting sentiment features")

        sentiments = df[text_column].apply(self.sentiment_analyzer.polarity_scores)

        df["sentiment_negative"] = sentiments.apply(lambda x: x["neg"])
        df["sentiment_neutral"] = sentiments.apply(lambda x: x["neu"])
        df["sentiment_positive"] = sentiments.apply(lambda x: x["pos"])
        df["sentiment_compound"] = sentiments.apply(lambda x: x["compound"])

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

        df[text_column] = df[text_column].astype(str)

        logger.info(
            "Extracting {n_topics} topic features using TF-IDF and SVD",
            extra={"format_args": True},
        )

        vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")

        tfidf_matrix = vectorizer.fit_transform(df[text_column])

        feature_names = vectorizer.get_feature_names_out()

        svd = TruncatedSVD(n_components=n_topics, random_state=42)
        topic_matrix = svd.fit_transform(tfidf_matrix)

        for i in range(n_topics):
            df["topic_{i + 1}_score"] = topic_matrix[:, i]

        topic_words = {}
        for i, comp in enumerate(svd.components_):
            top_word_indices = comp.argsort()[: -n_top_words - 1 : -1]
            top_words = [feature_names[idx] for idx in top_word_indices]
            topic_words["topic_{i + 1}"] = top_words

        topics_df = pd.DataFrame(topic_words)

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
                           "Timestamp column '{timestamp_column}' not found in DataFrame"
                          )
            return df

        try:
            df[timestamp_column] = pd.to_datetime(df[timestamp_column])
        except Exception:
            logger.error(
                "Failed to convert '{timestamp_column}' to datetime: {e}",
                extra={"format_args": True},
            )
            return df

        logger.info("Extracting time features")

        df["year"] = df[timestamp_column].dt.year
        df["month"] = df[timestamp_column].dt.month
        df["day"] = df[timestamp_column].dt.day
        df["day_of_week"] = df[timestamp_column].dt.dayofweek
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        df["hour"] = df[timestamp_column].dt.hour

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

        df = self.extract_basic_features(df, text_column)

        df = self.extract_sentiment_features(df, text_column)

        df = self.extract_time_features(df, timestamp_column)

        if extract_topics:
            df, topics_df = self.extract_topic_features(df, text_column)
            return df, topics_df

        return df
