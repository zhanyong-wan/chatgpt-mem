#!/usr/bin/env python3

"""Main program for ChatGPT with long-term memory.

USAGE:

    src/main.py init
        Initialilzes the environment.

    src/main.py embed <text>
        Prints the embedding of <text>.

    src/main.py add <text>
        Adds <text> with the current timestamp as its ID to the memory.

    src/main.py update <id> <text>
        Updates the memory with id <id> to <text>.
    
    src/main.py query <query> [<start-time> [<end-time>]]
        Finds the memories most relevant to <query>.  If <start-time> is given,
        include only memories that were added at or after <start-time>.  If
        <end-time> is given, include only memories that were added before<end-time>.
        The time format is "YYYY-MM-DDThh:mm:ss.xxxxxx" in UTC, e.g.
        "2021-01-01T12:34:56.123456".
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

    if command == "query":
        query = args[1]
        start_time = args[2] if len(args) > 2 else ""
        end_time = args[3] if len(args) > 3 else ""
        memories = utils.query_memory(
            query=query, start_time=start_time, end_time=end_time
        )
        time_range_str = (
            f" in time range [{start_time}..{end_time})"
            if start_time or end_time
            else ""
        )
        print(
            f"Found {len(memories)} memories matching '{query}'{time_range_str}.",
            file=sys.stderr,
        )
        for score, memory in memories:
            print(f"{memory.id} ({score}) {memory.text}")
        return

    if command == "get":
        mem_id = args[1]
        memory = utils.get_memories(ids=[mem_id])[0]
        print(f"Memory {mem_id}: {memory.text}")
        return


if __name__ == "__main__":
    main()
