import os
import textwrap
import tomllib
from PyPDF2 import PdfReader
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

# ========================
# 1. Ladda API-nycklar
# ========================
with open("src/.streamlit/secrets.toml", "rb") as f:
    secrets = tomllib.load(f)

OPENAI_API_KEY = secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = secrets["PINECONE_API_KEY"]

# ========================
# 2. Initiera klienter
# ========================
INDEX_NAME = "isektionen-rag-1536"  # ditt Pinecone-index

openai = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# Skapa index om det inte redan finns
if INDEX_NAME not in [idx["name"] for idx in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)

# ========================
# 3. PDF → text
# ========================
def pdf_to_text(path: str) -> str:
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

# ========================
# 4. Dela upp text i chunks
# ========================
def chunk_text(text, max_length=800):
    return textwrap.wrap(text, max_length)

# ========================
# 5. Skapa embeddings med OpenAI
# ========================
def get_embedding(text: str):
    resp = openai.embeddings.create(
        input=text,
        model="text-embedding-3-small"
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
        vectors.append({
            "id": f"{os.path.basename(pdf_path)}-chunk-{i}",
            "values": emb,
            "metadata": {"text": chunk}

        })

    index.upsert(vectors=vectors, namespace="TKMJ51")
    print(f"Laddade upp {len(chunks)} chunks från {pdf_path} till Pinecone!")

# ========================
# Körning
# ========================
if __name__ == "__main__":
    pdf_file = "huff3.pdf"
    file2 = "huff4.pdf"
    load_pdf_into_pinecone(pdf_file)
    load_pdf_into_pinecone(file2)