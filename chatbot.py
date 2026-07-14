"""
chatbot.py
----------
This module contains the core chatbot logic:

1. Loads the FAISS vector store created by ingest.py.
2. Loads a free, local HuggingFace LLM (google/flan-t5-base by default).
3. Builds a Conversational Retrieval Chain that combines:
     - the retriever (finds relevant chunks from the documents)
     - the LLM (generates the final answer)
     - conversation memory (remembers previous turns, so the bot
       understands follow-up questions like "where was he born?")
4. Exposes a simple ChatBot class that app.py (Streamlit) or a notebook
   can use with a single `chatbot.ask("your question")` call.

This file is written so it can be imported directly, or run on its own
for a quick command-line test:
    python chatbot.py
"""

import os

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
VECTORSTORE_DIR = "vectorstore"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# flan-t5-base is small, free, and runs comfortably on Google Colab's free CPU/GPU.
# You can switch to "google/flan-t5-large" if you have a GPU runtime for better quality.
LLM_MODEL_NAME = "google/flan-t5-base"

# Number of relevant chunks to retrieve from the vector store per question
TOP_K_CHUNKS = 3

# A custom prompt that tells the model to answer using retrieved context only,
# and to say so honestly if the answer isn't in the documents.
QA_PROMPT_TEMPLATE = """You are a helpful assistant that answers questions using the
provided context from the user's documents. If the answer is not contained in the
context, say "I could not find that information in the provided documents." Do not
make up information.

Context:
{context}

Chat History:
{chat_history}

Question: {question}
Helpful Answer:"""


class ChatBot:
    """
    A context-aware chatbot that combines retrieval-augmented generation (RAG)
    with conversational memory.
    """

    def __init__(
        self,
        vectorstore_dir: str = VECTORSTORE_DIR,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
        llm_model_name: str = LLM_MODEL_NAME,
        top_k: int = TOP_K_CHUNKS,
    ):
        self.vectorstore_dir = vectorstore_dir
        self.embedding_model_name = embedding_model_name
        self.llm_model_name = llm_model_name
        self.top_k = top_k

        self.vectorstore = self._load_vectorstore()
        self.llm = self._load_llm()
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True, output_key="answer"
        )
        self.chain = self._build_chain()

    # -----------------------------------------------------------------
    def _load_vectorstore(self):
        """
        Loads the FAISS vector store saved by ingest.py.
        Raises a clear error if the vector store was never created.
        """
        if not os.path.isdir(self.vectorstore_dir):
            raise FileNotFoundError(
                f"Vector store not found at '{self.vectorstore_dir}'. "
                f"Please run 'python ingest.py' first to build it."
            )

        print(f"[INFO] Loading embedding model '{self.embedding_model_name}'...")
        embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)

        print(f"[INFO] Loading FAISS vector store from '{self.vectorstore_dir}'...")
        try:
            vectorstore = FAISS.load_local(
                self.vectorstore_dir,
                embeddings,
                # Required by newer versions of langchain/FAISS for local pickle loading.
                # Safe here because we created this file ourselves in ingest.py.
                allow_dangerous_deserialization=True,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load FAISS vector store: {e}")

        return vectorstore

    # -----------------------------------------------------------------
    def _load_llm(self):
        """
        Loads a free HuggingFace sequence-to-sequence model
        (flan-t5-base by default) and wraps it as a LangChain LLM.
        """
        print(f"[INFO] Loading LLM '{self.llm_model_name}' (first run may take a while)...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(self.llm_model_name)

            hf_pipeline = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=256,
                temperature=0.3,
                do_sample=False,
            )
            llm = HuggingFacePipeline(pipeline=hf_pipeline)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load LLM '{self.llm_model_name}'. "
                f"Check your internet connection or try a smaller model. Error: {e}"
            )
        return llm

    # -----------------------------------------------------------------
    def _build_chain(self):
        """
        Builds a ConversationalRetrievalChain, which automatically:
          - rewrites follow-up questions using chat history
          - retrieves relevant chunks from FAISS
          - feeds them plus chat history into the LLM via our custom prompt
        """
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.top_k})

        qa_prompt = PromptTemplate(
            template=QA_PROMPT_TEMPLATE,
            input_variables=["context", "chat_history", "question"],
        )

        chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": qa_prompt},
            return_source_documents=True,
        )
        return chain

    # -----------------------------------------------------------------
    def ask(self, question: str) -> dict:
        """
        Sends a question to the chatbot and returns a dictionary containing
        the answer and the source document chunks used to generate it.

        Handles empty questions gracefully.
        """
        if not question or not question.strip():
            return {
                "answer": "Please enter a question before sending.",
                "sources": [],
            }

        try:
            result = self.chain.invoke({"question": question})
        except Exception as e:
            return {
                "answer": f"Sorry, something went wrong while generating a response: {e}",
                "sources": [],
            }

        sources = []
        for doc in result.get("source_documents", []):
            source_name = doc.metadata.get("source", "unknown")
            sources.append(source_name)

        return {
            "answer": result.get("answer", "").strip(),
            "sources": list(dict.fromkeys(sources)),  # de-duplicate, keep order
        }

    def clear_memory(self):
        """Clears the conversation history, starting a fresh conversation."""
        self.memory.clear()


# ---------------------------------------------------------------------------
# Quick manual test when running this file directly:  python chatbot.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    bot = ChatBot()
    print("\nContext-Aware Chatbot ready! Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in ("exit", "quit"):
            break
        response = bot.ask(user_input)
        print(f"Bot: {response['answer']}")
        if response["sources"]:
            print(f"(Sources: {', '.join(response['sources'])})")
