            # Average vectors or use zero vector if no tokens found
            # Get vectors for tokens that are in vocabulary
        # Create DataFrame with IDs and embeddings
# Configure logging
# G004: Logging f-strings temporarily allowed for development
from gensim.models import FastText, Word2Vec
from gensim.utils import simple_preprocess
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import numpy as np
import pandas as pd




logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


class BaseEmbedder:
    """Base class for text embedding models."""

    def __init__(self) -> None:
        self.model = None

    def fit(self, texts: list[str]) -> "BaseEmbedder":
        """Fit the embedding model on a list of texts.

        Args:
            texts: List of texts to fit the model on

        Returns:
            Self for chaining

        """
        msg = "Subclasses must implement fit()"
        raise NotImplementedError(msg)

    def transform(self, texts: list[str]) -> np.ndarray:
        """Transform texts into embeddings.

        Args:
            texts: List of texts to transform

        Returns:
            Array of embeddings

        """
        msg = "Subclasses must implement transform()"
        raise NotImplementedError(msg)

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        """Fit the model and transform texts into embeddings.

        Args:
            texts: List of texts to fit and transform

        Returns:
            Array of embeddings

        """
        return self.fit(texts).transform(texts)


class TfidfEmbedder(BaseEmbedder):
    """TF-IDF based text embedder."""

    def __init__(
        self,
        max_features: int | None = 1000,
        min_df: int = 5,
        max_df: float = 0.8,
        ngram_range: tuple = (1, 2),
    ) -> None:
        """Initialize TF-IDF embedder.

        Args:
            max_features: Maximum number of features (vocabulary size)
            min_df: Minimum document frequency for terms
            max_df: Maximum document frequency for terms
            ngram_range: Range of n-grams to consider

        """
        super().__init__()
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        self.model = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range,
        )

    def fit(self, texts: list[str]) -> "TfidfEmbedder":
        """Fit the TF-IDF vectorizer on a list of texts.

        Args:
            texts: List of texts to fit the vectorizer on

        Returns:
            Self for chaining

        """
        logger.info(
            "Fitting TF-IDF vectorizer on {len(texts)} texts with max_features={self.max_features}"
        )
        self.model.fit(texts)
        logger.info(
            "Vocabulary size: {len(self.model.vocabulary_)}",
            extra={"format_args": True},
        )
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        """Transform texts into TF-IDF embeddings.

        Args:
            texts: List of texts to transform

        Returns:
            Array of TF-IDF embeddings

        """
        if self.model is None:
            msg = "Model has not been fit yet"
            raise ValueError(msg)

        return self.model.transform(texts).toarray()


class Word2VecEmbedder(BaseEmbedder):
    """Word2Vec based text embedder."""

    def __init__(
        self,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 1,
        workers: int = 4,
        sg: int = 1,  # Skip-gram (1) or CBOW (0)
        epochs: int = 10,
    ) -> None:
        """Initialize Word2Vec embedder.

        Args:
            vector_size: Dimensionality of word vectors
            window: Maximum distance between current and predicted word
            min_count: Minimum word count
            workers: Number of threads to run in parallel
            sg: Training algorithm: 1 for skip-gram, 0 for CBOW
            epochs: Number of iterations over the corpus

        """
        super().__init__()
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.sg = sg
        self.epochs = epochs
        self.model = None

    def _preprocess_texts(self, texts: list[str]) -> list[list[str]]:
        """Preprocess texts for Word2Vec training.

        Args:
            texts: List of texts to preprocess

        Returns:
            List of tokenized texts

        """
        return [simple_preprocess(text) for text in texts]

    def fit(self, texts: list[str]) -> "Word2VecEmbedder":
        """Fit Word2Vec model on a list of texts.

        Args:
            texts: List of texts to fit the model on

        Returns:
            Self for chaining

        """
        logger.info("Preprocessing {len(texts)} texts for Word2Vec", extra={"format_args": True})
        tokenized_texts = self._preprocess_texts(texts)

        logger.info(
            "Training Word2Vec model with vector_size={self.vector_size}, window={self.window}"
        )
        self.model = Word2Vec(
            sentences=tokenized_texts,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sg=self.sg,
            epochs=self.epochs,
        )

        logger.info(
            "Word2Vec model trained with {len(self.model.wv.index_to_key)} words in vocabulary"
        )
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        """Transform texts into Word2Vec embeddings by averaging word vectors.

        Args:
            texts: List of texts to transform

        Returns:
            Array of averaged Word2Vec embeddings

        """
        if self.model is None:
            msg = "Model has not been fit yet"
            raise ValueError(msg)

        tokenized_texts = self._preprocess_texts(texts)
        embeddings = []

        for tokens in tokenized_texts:
            vectors = [self.model.wv[token] for token in tokens if token in self.model.wv]

            embedding = np.mean(vectors, axis=0) if vectors else np.zeros(self.vector_size)

            embeddings.append(embedding)

        return np.array(embeddings)


class FastTextEmbedder(Word2VecEmbedder):
    """FastText based text embedder."""

    def fit(self, texts: list[str]) -> "FastTextEmbedder":
        """Fit FastText model on a list of texts.

        Args:
            texts: List of texts to fit the model on

        Returns:
            Self for chaining

        """
        logger.info("Preprocessing {len(texts)} texts for FastText", extra={"format_args": True})
        tokenized_texts = self._preprocess_texts(texts)

        logger.info(
            "Training FastText model with vector_size={self.vector_size}, window={self.window}"
        )
        self.model = FastText(
            sentences=tokenized_texts,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sg=self.sg,
            epochs=self.epochs,
        )

        logger.info(
            "FastText model trained with {len(self.model.wv.index_to_key)} words in vocabulary"
        )
        return self


class EmbeddingPipeline:
    """Pipeline for generating and storing text embeddings."""

    def __init__(self, embedder: BaseEmbedder) -> None:
        """Initialize embedding pipeline.

        Args:
            embedder: Text embedder to use

        """
        self.embedder = embedder

    def generate_embeddings(
        self,
        df: pd.DataFrame,
        text_column: str = "processed_text",
        id_column: str = "id",
    ) -> pd.DataFrame:
        """Generate embeddings for texts in a DataFrame.

        Args:
            df: DataFrame containing texts
            text_column: Name of column containing processed texts
            id_column: Name of column containing unique identifiers

        Returns:
            DataFrame with text IDs and embeddings

        """
        if text_column not in df.columns:
            msg = "Text column '{text_column}' not found in DataFrame"
            raise ValueError(msg)

        texts = df[text_column].tolist()

        logger.info("Generating embeddings for {len(texts)} texts", extra={"format_args": True})
        embeddings = self.embedder.fit_transform(texts)

        logger.info(
            "Generated embeddings with shape {embeddings.shape}",
            extra={"format_args": True},
        )

        return pd.DataFrame(
            {
                "entry_id": df[id_column],
                "embedding": [embedding.tolist() for embedding in embeddings],
            }
        )

    def save_embeddings_to_csv(self, embeddings_df: pd.DataFrame, output_path: str) -> None:
        """Save embeddings DataFrame to CSV.

        Args:
            embeddings_df: DataFrame containing entry IDs and embeddings
            output_path: Path to save the CSV file

        """
        embeddings_df.to_csv(output_path, index=False)
        logger.info(
            "Saved {len(embeddings_df)} embeddings to {output_path}",
            extra={"format_args": True},
        )
