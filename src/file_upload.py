import os
from PyPDF2 import PdfReader
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import textwrap

# ========================
# 1. Sätt dina API-nycklar
# ========================
import tomllib

with open("src/.streamlit/secrets.toml", "rb") as f:
    secrets = tomllib.load(f)

OPENAI_API_KEY = secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = secrets["PINECONE_API_KEY"]


# ========================
# 2. Initiera klienter
# ========================

INDEX_NAME = "isektionen-rag"  # använd ditt redan skapade index

openai = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# Istället för att skapa nytt index, anslut direkt
index = pc.Index(INDEX_NAME)

# ========================
# 3. PDF → text
# ========================
def pdf_to_text(path: str) -> str:
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


# ========================
# 4. Dela upp text i chunks
# ========================
def chunk_text(text, max_length=800):
    # Dela i ~800 tecken (inte tokens men funkar OK här)
    return textwrap.wrap(text, max_length)


# ========================
# 5. Skapa embeddings med OpenAI
# ========================
def get_embedding(text: str):
    resp = openai.embeddings.create(
        input=text, model="text-embedding-3-small"
    )
    return resp.data[0].embedding


# ========================
# 6. Ladda upp chunks till Pinecone
# ========================
def load_pdf_into_pinecone(pdf_path: str):
    text = pdf_to_text(pdf_path)
    chunks = chunk_text(text)

    vectors = []
    for i, chunk in enumerate(chunks):
        emb = get_embedding(chunk)
        vectors.append(
            {"id": f"{os.path.basename(pdf_path)}-chunk-{i}", "values": emb, "metadata": {"text": chunk}}
        )

    index.upsert(vectors=vectors)
    print(f"Laddade upp {len(chunks)} chunks från {pdf_path} till Pinecone!")


# ========================
# Körning
# ========================
if __name__ == "__main__":
    pdf_file = "din_fil.pdf"  # byt ut mot din lokala PDF
    load_pdf_into_pinecone(pdf_file)