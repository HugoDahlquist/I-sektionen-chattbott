# src/chatlogic.py
import os
from openai import OpenAI
from pinecone import Pinecone

class ChatLogic:
    def __init__(self, openai_key: str, pinecone_key: str, index_name: str):
        # Init clients
        self.client = OpenAI(api_key=openai_key)
        self.pc = Pinecone(api_key=pinecone_key)
        self.index = self.pc.Index(index_name)
        self.model = "gpt-3.5-turbo"  # default, can be overridden

    # --- Embeddings
    def get_embedding(self, text: str):
        resp = self.client.embeddings.create(
            input=text, model="text-embedding-3-small"
        )
        return resp.data[0].embedding

    # --- Pinecone retrieval
    def retrieve_context(self, query: str, top_k: int = 5) -> str:
        query_vec = self.get_embedding(query)
        results = self.index.query(
            vector=query_vec, top_k=top_k, include_metadata=True
        )
        if not results.matches:
            return ""
        context_chunks = [
            match["metadata"]["text"]
            for match in results.matches
            if "metadata" in match
        ]
        return "\n\n".join(context_chunks)

    # --- Generate response with context + history
    def generate_response(self, messages: list[dict], query: str, top_k: int = 5):
        context = self.retrieve_context(query, top_k=top_k)

        system_prompt = (
            "You are a helpful assistant for the I-sektionen knowledge base. "
            "Always use the retrieved context to answer questions. "
            "If context is missing, say you don't know.\n\n"
            f"Context:\n{context}"
        )

        # Build full conversation (prepend system + history)
        full_messages = [{"role": "system", "content": system_prompt}] + messages

        # Stream response back
        return self.client.chat.completions.create(
            model=self.model, messages=full_messages, stream=True
        )