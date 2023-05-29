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

    utils.init_environment()

    command = args[0]
    if command == "init":
        return

    if command == "embed":
        text = args[1]
        embedding = utils.to_embedding(text)
        print(f"Text: {text}")
        print(f"Embedding: {embedding}")
        return

    if command == "add":
        text = args[1]
        mem_id = utils.add_memory(text)
        print(f"Added memory '{text}' with id {mem_id}.", file=sys.stderr)
        return

    if command == "update":
        mem_id = args[1]
        text = args[2]
        utils.update_memory(id=mem_id, memory=text)
        print(f"Updated memory {mem_id} with new content '{text}'.", file=sys.stderr)
        return


if __name__ == "__main__":
    main()
