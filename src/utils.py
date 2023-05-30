"""Utilities for working with the OpenAI API and PineCone API.

References:
  - OpenAI API: https://platform.openai.com/docs/introduction
  - Pinecone API: https://docs.pinecone.io/docs/overview
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
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
# Key for storing the memory importance score (1-10).
PINECONE_INDEX_METADATA_KEY_MEMORY_IMPORTANCE = "importance"


class Memory:
    """A memory stored in the PineCone index."""

    def __init__(self, id: str, time: datetime, importance: int, text: str) -> None:
        self.id = id
        self.time = time
        self.importance = importance
        self.text = text

    def __repr__(self) -> str:
        return f"Memory(id={self.id}, time={datetime_to_string(self.time)}, importance={self.importance}, text={self.text})"


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

    result = openai.Embedding.create(input=[text], model=OPENAI_TEXT_EMBEDDING_MODEL)
    data = result["data"]  # type: ignore
    embedding = data[0]["embedding"]
    assert len(embedding) == OPENAI_GPT_EMBEDDING_DIMENSION
    return embedding


_DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"
_OLD_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f"


def datetime_to_string(dt: datetime) -> str:
    """Converts a datetime to its string representation.

    Args:
        dt: The datetime.

    Returns:
        The string representation in the format of "YYYY-MM-DDThh:mm:ss.xxxxxx".
    """

    return dt.strftime(_DATETIME_FORMAT)


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
        string: The string to convert, in the format "YYYY-MM-DDThh:mm:ss.xxxxxx".

    Returns:
        The datetime.
    """

    try:
        return datetime.strptime(string, _DATETIME_FORMAT)
    except ValueError as e:
        return datetime.strptime(string, _OLD_DATETIME_FORMAT)


def update_memory(id: str, memory: str, verbose: bool = False) -> None:
    """Upserts a memory into the PineCone index.

    Args:
        id: The memory ID.
        memory: The memory text.
        verbose: Whether to print verbose information.
    """

    index = pinecone.Index(PINECONE_INDEX)
    embedding = to_embedding(memory)
    if verbose:
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
                    PINECONE_INDEX_METADATA_KEY_MEMORY_IMPORTANCE: rate_importance(
                        memory
                    ),
                    PINECONE_INDEX_METADATA_KEY_MEMORY_TEXT: memory,
                },
            ),
        ],
        namespace=PINECONE_INDEX_NAMESPACE,
    )


def add_memory(memory: str, utc_time: Optional[datetime] = None) -> str:
    """Inserts a memory into the Pinecone index.

    Memories are keyed by the timestamp of the memory.

    Args:
        memory: The memory text.
        utc_time: The UTC timestamp of the memory (defaults to now).

    Returns:
        The memory timestamp.
    """

    # Default the timestamp to now.  We cannot set the default value in the function
    # declaration, as that will be evaluated only once (as opposed to every time we
    # call the function).
    if not utc_time:
        utc_time = datetime.utcnow()

    # Identify memories via microsecond-level timestamps.
    # E.g. 2023-05-10T20:02:28.328142
    id = datetime_to_string(utc_time)
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
    filter: Optional[Dict[str, Any]] = None
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
        id = item["id"]
        metadata = item["metadata"]
        memories.append(
            (
                item["score"],
                Memory(
                    id=id,
                    time=string_to_datetime(id),
                    importance=metadata.get(
                        PINECONE_INDEX_METADATA_KEY_MEMORY_IMPORTANCE, 0
                    ),
                    text=metadata[PINECONE_INDEX_METADATA_KEY_MEMORY_TEXT],
                ),
            )
        )
    return memories


def get_memories(ids: List[str]) -> List[Memory]:
    """Gets the memories with the given IDs.

    Args:
        ids: The list of IDs.

    Returns:
        The list of matching memories.
    """

    index = pinecone.Index(PINECONE_INDEX)
    for id in ids:
        assert (
            " " not in id
        ), f"ID '{id}' contains spaces, which is not allowed by fetch()."
    result = index.fetch(namespace=PINECONE_INDEX_NAMESPACE, ids=ids)
    # vectors is a map from ID to value.  Each mapped value looks like
    #   {'id': ID, 'metadata': {'time': TIME, 'memory': TEXT}, 'values': [...]}
    vectors = result["vectors"]
    memories = []
    for id in ids:
        item = vectors[id]
        metadata = item["metadata"]
        memories.append(
            Memory(
                id=id,
                time=string_to_datetime(id),
                importance=metadata.get(
                    PINECONE_INDEX_METADATA_KEY_MEMORY_IMPORTANCE, 0
                ),
                text=metadata[PINECONE_INDEX_METADATA_KEY_MEMORY_TEXT],
            )
        )
    return memories


def delete_memories(ids: List[str]) -> None:
    """Deletes the memories with the given IDs.

    Args:
        ids: The list of IDs.
    """

    index = pinecone.Index(PINECONE_INDEX)
    index.delete(namespace=PINECONE_INDEX_NAMESPACE, ids=ids)


def _make_messages(conversation: List[str]) -> List[Dict[str, str]]:
    """Makes the messages to send to GPT.

    Args:
        conversation: The conversation history.  It contains alternating text
        spoken by the user and the bot.  The length should always be odd.
        The last element should be the user's last message.

    Returns:
        The messages to send to GPT.
    """

    assert (
        len(conversation) % 2 == 1
    ), "The conversation history should have an odd length."

    # Generate the messages in reverse order, as we need to discard early conversation
    # when the total length exceeds the history window size.
    reverse_messages: List[Dict[str, str]] = []
    role = "user"  # user or assistant.
    total_len = 0
    for text in reversed(conversation):
        # Limit the total length of text sent to the bot.
        total_len += len(text)
        if total_len > 2048:
            break

        # In Python, it's more efficient to append to a list than to insert at the beginning.
        reverse_messages.append({"role": role, "content": text})
        # Reverse the role.
        role = "assistant" if role == "user" else "user"

    # Set the behavior of the bot.
    reverse_messages.append(
        {"role": "system", "content": "You are a helpful assistant."}
    )
    return list(reversed(reverse_messages))


def _get_gpt_answer(messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
    """Sends the list of messages to GPT and returns the answer.

    Args:
        messages: The messages to send to GPT.

    Returns:
        The GPT answer.
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=temperature,
    )
    return response["choices"][0]["message"]["content"].strip()  # type: ignore


def chat() -> None:
    """Chats with GPT, saving the history to external memory."""

    print("Type 'quit' to exit.", file=sys.stderr)
    conversation: List[str] = []  # Conversation history so far.
    while True:
        text = input(">>> ").strip()
        if not text:
            continue
        if text == "quit":
            break

        conversation.append(text)
        add_memory(f"I said: ```{text}```")

        answer = _get_gpt_answer(_make_messages(conversation))
        print(f"{answer}", file=sys.stderr)

        conversation.append(answer)
        add_memory(f"GPT said: ```{answer}```")


def rate_importance(memory: str) -> int:
    """Rates the importance of a memory by asking the GPT model.

    Args:
        memory: The memory text.

    Returns:
        A integer in the range [1, 10] where 1 is least important and 10 is most.
    """

    messages = _make_messages(
        [
            f"""On the scale of 1 to 10, where 1 is purely unimportant (e.g., saying hello) and 10 is extremely important and useful (e.g., saving mankind), rate the likely importance of the following piece of memory (delimited by ```). Just give me the numeric rating and nothing else.
Memory: ```{memory}```
Rating: <number>"""
        ]
    )
    answer = _get_gpt_answer(messages, temperature=0.0)
    print(answer)
    return int(answer)


def rate_memory_by_id(id: str) -> int:
    """Rates the importance of a memory by asking the GPT model.

    Args:
        id: The memory ID.

    Returns:
        A integer in the range [1, 10] where 1 is least important and 10 is most.
    """

    memory = get_memories(ids=[id])[0]
    return rate_importance(memory.text)
