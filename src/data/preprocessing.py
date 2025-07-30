            import nltk
        # Apply lemmatization if enabled
        # Apply stemming if enabled
        # Apply text preprocessing to content
        # Average word length
        # Character count
        # Combine title and content for full text analysis
        # Convert to lowercase if enabled
        # Extract text features
        # Handle missing values
        # Initialize NLP components
        # Join tokens back into text
        # Make a copy to avoid modifying the original
        # Remove punctuation if enabled
        # Remove stopwords if enabled
        # Sentence count (approximation)
        # Tokenize text
        # Word count
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
import string






class TextPreprocessor:
    """Text preprocessing pipeline for journal entries."""

    def __init__(
        self,
        remove_stopwords: bool = True,
        remove_punctuation: bool = True,
        lowercase: bool = True,
        stemming: bool = False,
        lemmatization: bool = True,
    ) -> None:
        """Initialize text preprocessor.

        Args:
            remove_stopwords: Whether to remove stopwords
            remove_punctuation: Whether to remove punctuation
            lowercase: Whether to convert text to lowercase
            stemming: Whether to apply stemming
            lemmatization: Whether to apply lemmatization

        """
        self.remove_stopwords = remove_stopwords
        self.remove_punctuation = remove_punctuation
        self.lowercase = lowercase
        self.stemming = stemming
        self.lemmatization = lemmatization

        try:
            nltk.download("punkt", quiet=True)
            nltk.download("stopwords", quiet=True)
            nltk.download("wordnet", quiet=True)
            self.stop_words = set(stopwords.words("english"))
            self.stemmer = PorterStemmer()
            self.lemmatizer = WordNetLemmatizer()
        except ImportError:
            self.stop_words = set()
            self.stemmer = None
            self.lemmatizer = None

    def preprocess_text(self, text: str) -> str:
        """Apply full preprocessing pipeline to text.

        Args:
            text: Input text to preprocess

        Returns:
            Preprocessed text

        """
        if not text or not isinstance(text, str):
            return ""

        if self.lowercase:
            text = text.lower()

        if self.remove_punctuation:
            text = text.translate(str.maketrans("", "", string.punctuation))

        tokens = word_tokenize(text)

        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]

        if self.stemming:
            tokens = [self.stemmer.stem(token) for token in tokens]

        if self.lemmatization:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]

        return " ".join(tokens)

    def preprocess_df(
        self,
        df: pd.DataFrame,
        text_column: str = "content",
        output_column: str = "processed_text",
    ) -> pd.DataFrame:
        """Preprocess text in a DataFrame column.

        Args:
            df: DataFrame containing text data
            text_column: Name of column containing raw text
            output_column: Name of column to store processed text

        Returns:
            DataFrame with processed text column added

        """
        df = df.copy()
        df[output_column] = df[text_column].astype(str).apply(self.preprocess_text)
        return df

    def extract_features(
        self, df: pd.DataFrame, text_column: str = "processed_text"
    ) -> pd.DataFrame:
        """Extract basic text features from preprocessed text.

        Args:
            df: DataFrame containing processed text
            text_column: Name of column containing processed text

        Returns:
            DataFrame with text features added

        """
        df = df.copy()

        df["char_count"] = df[text_column].apply(len)

        df["word_count"] = df[text_column].apply(lambda x: len(x.split()))

        df["sentence_count"] = df[text_column].apply(
            lambda x: x.count(".") + x.count("!") + x.count("?") + 1
        )

        df["avg_word_length"] = df[text_column].apply(
            lambda x: np.mean([len(word) for word in x.split()]) if len(x.split()) > 0 else 0
        )

        return df


class JournalEntryPreprocessor:
    """Preprocessing pipeline for journal entries."""

    def __init__(self, text_preprocessor: TextPreprocessor | None = None) -> None:
        """Initialize journal entry preprocessor.

        Args:
            text_preprocessor: Text preprocessor to use

        """
        self.text_preprocessor = text_preprocessor or TextPreprocessor()

    def preprocess(
        self,
        df: pd.DataFrame,
        text_column: str = "content",
        title_column: str = "title",
    ) -> pd.DataFrame:
        """Preprocess journal entries DataFrame.

        Args:
            df: DataFrame containing journal entries
            text_column: Name of column containing entry content
            title_column: Name of column containing entry titles

        Returns:
            DataFrame with processed entries

        """
        df = df.copy()

        df[text_column] = df[text_column].fillna("")
        df[title_column] = df[title_column].fillna("")

        df["full_text"] = df[title_column] + " " + df[text_column]

        df = self.text_preprocessor.preprocess_df(df, text_column=text_column)

        return self.text_preprocessor.extract_features(df, text_column="processed_text")
