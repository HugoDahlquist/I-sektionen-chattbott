# src/chatlogic.py
from multiprocessing import context
import os
from openai import OpenAI
from pinecone import Pinecone

class ChatLogic:
    def __init__(self, openai_key: str, pinecone_key: str, index_name: str):
        # Init clients
        self.client = OpenAI(api_key=openai_key)
        self.pc = Pinecone(api_key=pinecone_key)
        self.index = self.pc.Index(index_name)
        self.model = "gpt-5-mini"  # default, can be overridden

    # --- Embeddings
    def get_embedding(self, text: str):
        resp = self.client.embeddings.create(
            input=text, model="text-embedding-3-small"
        )
        return resp.data[0].embedding

    # --- Pinecone retrieval
    def retrieve_context(self, query: str, course: str, top_k: int = 5) -> str:
        query_vec = self.get_embedding(query)
        #options = {"namespace": "TDEI76"}  # specify namespace if needed
        results = self.index.query(
            vector=query_vec, top_k=top_k, include_metadata=True, namespace=course
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
    def generate_response(self, messages: list[dict], query: str, course: str, top_k: int = 5):
        context = self.retrieve_context(query, top_k=top_k, course=course)

        system_prompt = (
            "You are a digital teacher for the I-sektionen knowledge base. "
            "Your role is to help students understand the material clearly and correctly. "
            "Always use the retrieved course context as the primary source when answering. "
            "If the context does not contain the answer, say that it is not covered and avoid making things up. "
            "Structure your answers in a pedagogical and easy-to-follow way, using examples, lists or step-by-step reasoning when appropriate. "
            "Adapt the difficulty of your explanation to the student's question "
            "(simplify for beginners, give more depth for advanced queries). "
            "Encourage understanding by explaining not only *what* the answer is, but also *why*. "
            "When useful, ask guiding follow-up questions to help the student learn more. "
            "Keep a supportive, motivating tone at all times.\n\n"
            f"Context:\n{context}"
        )

        # Build full conversation (prepend system + history)
        full_messages = [{"role": "system", "content": system_prompt}] + messages

        # Stream response back
        return self.client.chat.completions.create(
            model=self.model, messages=full_messages, stream=True
        )