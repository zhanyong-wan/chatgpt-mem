# chatgpt-mem

Demonstrates how to extend ChatGPT with long-term memory to bypass the limit of the
context window size.

## Prerequisite

* OpenAI account and API key: can be created at https://platform.openai.com
* Pinecone account and API key: can be created for free at https://www.pinecone.io

## How to use

First, clone the repo and specify your API keys:

```
$ git clone https://github.com/zhanyong-wan/chatgpt-mem
$ cd chatgpt-mem
$ cp src/api_secrets.py.template src/api_secrets.py
# Replace the ???s in src/api_secrets.py with your actual API keys and environment.
```

Then you can run the program like:

```
# Display the usage.
$ src/main.py

# Start a chat session.
$ src/main.py chat
```

During the chat, the program will store the chat history in a Pinecone vector
database.  It will also fetch relevant history and provide it as part of the
prompt to GPT.  This allows GPT to reference things chatted about in the past.
