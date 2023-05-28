"""Utilities for working with the OpenAI API and PineCone API.

References:
  - OpenAI API: https://platform.openai.com/docs/introduction
  - Pinecone API: https://docs.pinecone.io/docs/overview
"""

import api_secrets
import pinecone
import openai
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
# Key for storing the memory text.
PINECONE_INDEX_METADATA_KEY_MEMORY_TEXT = "memory"


def init_environment() -> None:
    """Initializes the environment for working with the OpenAI API and PineCone API."""

    # Specify the API keys.
    openai.api_key = api_secrets.OPENAI_API_KEY
    pinecone.init(
        api_key=api_secrets.PINECONE_API_KEY,
        environment=api_secrets.PINECONE_ENVIRONMENT,
    )

    # Verify that the chatgpt-mem index exists in PineCone, creating one if necessary.
    active_indexes = pinecone.list_indexes()
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
        print(
            f"Found {len(active_indexes)} active PineCone indexes: {', '.join(active_indexes)}.",
            file=sys.stderr,
        )
        assert PINECONE_INDEX in active_indexes
    print(f"Found the {PINECONE_INDEX} index.", file=sys.stderr)
    desc = pinecone.describe_index(PINECONE_INDEX)
    print(f"Index description: {desc}", file=sys.stderr)
