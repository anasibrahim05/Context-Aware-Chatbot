# Context-Aware Chatbot Using LangChain and Retrieval-Augmented Generation (RAG)

## 📌 Project Overview

This project is a **context-aware chatbot** that answers questions using information
retrieved from a set of documents (Retrieval-Augmented Generation, or RAG), while also
remembering previous turns of the conversation.

It is built entirely with **free, open-source tools** so it can run on **Google Colab's
free tier** without any paid API keys (no OpenAI, no Pinecone, no paid vector database).

## ✨ Features

- **Retrieval-Augmented Generation (RAG):** answers are grounded in your own documents,
  not just the model's training data.
- **Conversational memory:** the bot remembers earlier messages, so follow-up questions
  like "Where was he born?" are understood correctly.
- **Fully free stack:** HuggingFace `flan-t5-base` LLM, `all-MiniLM-L6-v2` embeddings,
  and a local FAISS vector database.
- **Streamlit web UI:** chat interface with history, a clear-chat button, and source
  citations for every answer.
- **Sample documents included:** no dataset download required — 4 ready-to-use text
  files about AI, Machine Learning, Python, and LangChain are already in `documents/`.
- **Error handling:** missing documents, empty questions, a missing vector store, and
  model-loading failures are all handled with clear messages.

## 📁 Folder Structure

```
ContextAwareChatbot/
│
├── app.py                 # Streamlit web application
├── chatbot.py              # Core RAG + memory chatbot logic
├── ingest.py                # Builds the FAISS vector store from documents/
├── requirements.txt         # Python dependencies
├── README.md                 # This file
├── documents/                # Source documents used for retrieval
│   ├── ai.txt
│   ├── ml.txt
│   ├── python.txt
│   └── langchain.txt
└── vectorstore/               # Generated FAISS index (created by ingest.py)
```

## 📦 Where the "dataset" comes from

This project does not need you to find or download any external dataset. Four short,
self-contained text documents (on AI, Machine Learning, Python, and LangChain) are
generated directly inside `documents/` — either already included in this project, or
created automatically for you by **Cell 3** of the Colab notebook. This means the whole
project runs end-to-end with **no manual downloads and no internet dataset dependency**.

If you'd rather use your **own** documents (PDF or TXT), just drop them into the
`documents/` folder (in Colab: use the file upload button in the left sidebar, or
`files.upload()` in a cell) and re-run `ingest.py`. Any `.txt` or `.pdf` file works.

## 🛠️ Installation

1. Clone or download this project folder.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

> On Google Colab, use `!pip install -r requirements.txt` in a code cell instead.

## ▶️ Running `ingest.py`

This builds the FAISS vector database from the documents in `documents/`. Run it once
(and again any time you add or change documents):

```bash
python ingest.py
```

Expected output:

```
[INFO] Loaded 4 document(s) from 'documents'.
[INFO] Split documents into NN chunks.
[INFO] Loading embedding model 'sentence-transformers/all-MiniLM-L6-v2'...
[INFO] Creating FAISS vector store from chunks...
[INFO] Vector store saved to 'vectorstore'.
[SUCCESS] Ingestion complete! You can now run chatbot.py or app.py.
```

## ▶️ Running `app.py`

Launch the Streamlit chat interface:

```bash
streamlit run app.py
```

On Google Colab, Streamlit needs a tunnel to be viewable in the browser — see the
provided Colab notebook for a ready-made command using `localtunnel`.

You can also test the chatbot directly from the terminal, without Streamlit:

```bash
python chatbot.py
```

## 💬 Example Conversation

```
You: What is machine learning?
Bot: Machine Learning is a subset of Artificial Intelligence that gives computers
     the ability to learn from data and improve their performance over time without
     being explicitly programmed for every task.
(Sources: ml.txt)

You: What are its main types?
Bot: The main types are supervised learning, unsupervised learning, and
     reinforcement learning.
(Sources: ml.txt)

You: Which one uses labeled data?
Bot: Supervised learning uses labeled data, where each training example has an
     input and a known correct output.
(Sources: ml.txt)
```

Notice how the third question ("Which one uses labeled data?") relies on the
conversation history to understand "one" refers to the types of ML mentioned earlier —
this is the conversational memory in action.

## 🚀 Future Improvements

- Swap `flan-t5-base` for a larger instruction-tuned model (e.g. `flan-t5-large` or a
  quantized Llama/Mistral model) when running on a GPU runtime for higher quality answers.
- Add support for `.docx` and website URLs as document sources.
- Add a re-ranking step to improve retrieval precision on larger document sets.
- Persist chat history to disk/database so conversations survive app restarts.
- Add streaming token-by-token responses in the Streamlit UI.
- Add automated evaluation of answer quality (e.g. using RAGAS).

## 🧠 Tech Stack Summary

| Component        | Tool                                             |
|-------------------|---------------------------------------------------|
| Orchestration      | LangChain                                         |
| LLM                 | google/flan-t5-base (HuggingFace, free)            |
| Embeddings          | sentence-transformers/all-MiniLM-L6-v2             |
| Vector Database      | FAISS (local, free)                              |
| Memory                | LangChain ConversationBufferMemory              |
| Document Loading         | PyPDF, LangChain TextLoader                  |
| UI                          | Streamlit                                |
| Environment                   | Google Colab (Free tier compatible) |
