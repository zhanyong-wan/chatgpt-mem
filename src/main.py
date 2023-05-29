#!/usr/bin/env python3

"""Main program for ChatGPT with long-term memory.

USAGE:

    src/main.py init
        Initialilzes the environment.
"""

import sys

import utils


def main():
    """Main program."""

    args = sys.argv[1:]
    if not args:
        sys.exit(__doc__)

    command = args[0]
    if command == "init":
        utils.init_environment()
    elif command == "embed":
        utils.init_environment()
        text = args[1]
        embedding = utils.to_embedding(text)
        print(f"Text: {text}")
        print(f"Embedding: {embedding}")


if __name__ == "__main__":
    main()
