"""Utilities for working with the OpenAI API and PineCone API.

References:
  - OpenAI API: https://platform.openai.com/docs/introduction
  - Pinecone API: https://docs.pinecone.io/docs/overview
"""

from datetime import datetime
from typing import List, Tuple
import api_secrets
import openai
import pinecone
import sys


# Number of dimensions in the GPT embedding space.
OPENAI_GPT_EMBEDDING_DIMENSION = 1536
# OpenAI model for computing text embeddings.
OPENAI_TEXT_EMBEDDING_MODEL = "text-embedding-ada-002"

# Name of the PineCone index for storing memories.
PINECONE_INDEX = "chatgpt-mem"
# Dimension of the PineCone index for storing memories.
PINECONE_INDEX_DIMENSION = OPENAI_GPT_EMBEDDING_DIMENSION
# How to measure the distance between two vectors.
PINECONE_INDEX_METRIC = "cosine"
# All memories will be stored in this namespace.
PINECONE_INDEX_NAMESPACE = "memories"
# Key for storing the memory time.
PINECONE_INDEX_METADATA_KEY_MEMORY_TIME = "time"
# Key for storing the memory text.
PINECONE_INDEX_METADATA_KEY_MEMORY_TEXT = "memory"


class Memory:
    """A memory stored in the PineCone index."""

    def __init__(self, time: datetime, text: str) -> None:
        self.time = time
        self.text = text

    def __repr__(self) -> str:
        return f"Memory(time={self.time}, text={self.text})"


def init_environment(verbose: bool = False) -> None:
    """Initializes the environment for working with the OpenAI API and PineCone API."""

    # Specify the API keys.
    openai.api_key = api_secrets.OPENAI_API_KEY
    pinecone.init(
        api_key=api_secrets.PINECONE_API_KEY,
        environment=api_secrets.PINECONE_ENVIRONMENT,
    )

    # Verify that the chatgpt-mem index exists in PineCone, creating one if necessary.
    active_indexes = pinecone.list_indexes()
    if verbose:
        print(
            f"Found {len(active_indexes)} active PineCone indexes: {', '.join(active_indexes)}.",
            file=sys.stderr,
        )
    if PINECONE_INDEX not in active_indexes:
        print(
            f"Didn't find the {PINECONE_INDEX} index. Creating it now (this may take several minutes).",
            file=sys.stderr,
        )
        try:
            pinecone.create_index(
                name=PINECONE_INDEX,
                dimension=PINECONE_INDEX_DIMENSION,
                metric=PINECONE_INDEX_METRIC,
            )
        except pinecone.ApiException as e:
            sys.exit(f"Failed to create the {PINECONE_INDEX} index: {e}")
        print(f"Created the {PINECONE_INDEX} index.", file=sys.stderr)
        active_indexes = pinecone.list_indexes()
        if verbose:
            print(
                f"Found {len(active_indexes)} active PineCone indexes: {', '.join(active_indexes)}.",
                file=sys.stderr,
            )
        assert PINECONE_INDEX in active_indexes
    desc = pinecone.describe_index(PINECONE_INDEX)
    if verbose:
        print(f"Found the {PINECONE_INDEX} index.", file=sys.stderr)
        print(f"Index description: {desc}", file=sys.stderr)


def to_embedding(text: str) -> List[float]:
    """Converts the text to its embedding vector using OpenAI API."""

    embedding = openai.Embedding.create(
        input=[text], model=OPENAI_TEXT_EMBEDDING_MODEL
    )["data"][0]["embedding"]
    assert len(embedding) == OPENAI_GPT_EMBEDDING_DIMENSION
    return embedding


def string_to_timestamp_in_microseconds(string: str) -> int:
    """Converts a time string to its timestamp in microseconds.

    Args:
        string: The string to convert, in the format "YYYY-MM-DD hh:mm:ss.xxxxxx".

    Returns:
        The timestamp in microseconds.
    """

    return int(string_to_datetime(string).timestamp() * 1e6)


def string_to_datetime(string: str) -> datetime:
    """Converts a time string to its datetime.

    Args:
        string: The string to convert, in the format "YYYY-MM-DD hh:mm:ss.xxxxxx".

    Returns:
        The datetime.
    """

    return datetime.strptime(string, "%Y-%m-%d %H:%M:%S.%f")


def update_memory(id: str, memory: str) -> None:
    """Upserts a memory into the PineCone index.

    Args:
        id: The memory ID.
        memory: The memory text.
    """

    index = pinecone.Index(PINECONE_INDEX)
    embedding = to_embedding(memory)
    print(f"Embedding = {embedding[:10]} + ...", file=sys.stderr)
    index.upsert(
        vectors=[
            (
                id,  # Vector ID
                embedding,  # Dense vector
                # Store the memory time and text in the vector metadata.
                {
                    # Store the timestamp as an integer, as PineCone doesn't support < comparison
                    # for strings.
                    PINECONE_INDEX_METADATA_KEY_MEMORY_TIME: string_to_timestamp_in_microseconds(
                        id
                    ),
                    PINECONE_INDEX_METADATA_KEY_MEMORY_TEXT: memory,
                },
            ),
        ],
        namespace=PINECONE_INDEX_NAMESPACE,
    )


def add_memory(memory: str, utc_time: datetime = datetime.utcnow()) -> str:
    """Inserts a memory into the Pinecone index.

    Memories are keyed by the timestamp of the memory.

    Args:
        memory: The memory text.
        utc_time: The UTC timestamp of the memory (defaults to now).

    Returns:
        The memory timestamp.
    """

    # Identify memories via microsecond-level timestamps.
    # E.g. 2023-05-10 20:02:28.328142
    id = f"{utc_time}"
    update_memory(id, memory)
    return id


def query_memory(
    query: str, start_time: str = "", end_time: str = "", top_k=10
) -> List[Tuple[float, Memory]]:
    """Queries the PineCone index for matching memories.

    Args:
        query: The query text.
        start_time: The start time of the query (inclusive). Empty means infinite past.
        end_time: The end time of the query (not inclusive). Empty means infinite future.
        top_k: The number of matching memories to return.

    Returns:
        A list of matching memories (score, memory), sorted by relevance score (high to low).
    """

    index = pinecone.Index(PINECONE_INDEX)

    # Build a time-range filter.
    # See https://docs.pinecone.io/docs/metadata-filtering for the format of the filter.
    if start_time and end_time:
        filter = {
            "$and": [
                {
                    "time": {"$gte": string_to_timestamp_in_microseconds(start_time)},
                },
                {
                    "time": {"$lt": string_to_timestamp_in_microseconds(end_time)},
                },
            ]
        }
    elif start_time:
        filter = {"time": {"$gte": string_to_timestamp_in_microseconds(start_time)}}
    elif end_time:
        filter = {"time": {"$lt": string_to_timestamp_in_microseconds(end_time)}}
    else:
        filter = None

    result = index.query(
        namespace=PINECONE_INDEX_NAMESPACE,
        vector=to_embedding(query),
        filter=filter,
        top_k=top_k,
        include_values=False,
        include_metadata=True,
    )
    memories = []
    for item in result["matches"]:
        memories.append(
            (
                item["score"],
                Memory(
                    time=string_to_datetime(item["id"]),
                    text=item["metadata"][PINECONE_INDEX_METADATA_KEY_MEMORY_TEXT],
                ),
            )
        )
    return memories
