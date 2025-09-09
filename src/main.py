# RAG chatbot using LangChain and OpenAI
import os
import json
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set in environment variables.")

client = OpenAI()

SYSTEM_PROMPT = (
    "You are Srihari Raman's personal AI assistant. Always speak in a positive, "
    "supportive manner about Srihari and his work. If information is missing, "
    "politely note that it isn't available in the provided documents."
)

# -------------------------------
# 1. Load PDFs
# -------------------------------
def load_pdfs(pdf_files):
    docs = []
    for pdf in pdf_files:
        loader = PyPDFLoader(pdf)
        docs.extend(loader.load())
    return docs

# -------------------------------
# 2. Load GitHub JSON
# -------------------------------
def load_github_json(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    username = data["username"]
    for project in data["projects"]:
        content = (
            f"Project: {project['name']}\n"
            f"Description: {project['description']}\n"
            f"Language: {project['language']}\n"
            f"Stars: {project['stars']}, Forks: {project['forks']}\n\n"
            f"README:\n{project['readme']}"
        )
        docs.append(Document(page_content=content, metadata={"source": f"github/{username}/{project['name']}"}))
    return docs

# -------------------------------
# 3. Split into chunks
# -------------------------------
def chunk_documents(docs, chunk_size=1000, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)

# -------------------------------
# 4. Build FAISS Vector Store
# -------------------------------
def build_faiss_index(documents):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

# -------------------------------
# 5. Run Query
# -------------------------------
def run_query(vectorstore, query, k=4):
    """Retrieve relevant docs and stream an answer from the LLM."""
    docs = vectorstore.as_retriever(search_kwargs={"k": k}).get_relevant_documents(query)
    context = "\n\n".join(d.page_content for d in docs)
    sources = [d.metadata.get("source", "unknown") for d in docs]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Use the following context to answer the question.\n\n{context}\n\nQuestion: {query}",
        },
    ]

    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        stream=True,
    )

    return stream, sources

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    # If this script is inside `src/`, then docs_dir is just `docs/`
    script_dir = os.path.dirname(os.path.abspath(__file__))
    docs_dir = os.path.join(script_dir, "docs")

    # Collect PDFs and JSON file
    pdf_files = [os.path.join(docs_dir, file) for file in os.listdir(docs_dir) if file.endswith(".pdf")]
    json_file = [os.path.join(docs_dir, file) for file in os.listdir(docs_dir) if file.endswith(".json")][0]

    # Load and combine docs
    pdf_docs = load_pdfs(pdf_files)
    github_docs = load_github_json(json_file)
    all_docs = pdf_docs + github_docs

    # Chunk for embeddings
    chunked_docs = chunk_documents(all_docs)

    # Build FAISS index
    vectorstore = build_faiss_index(chunked_docs)

    # Example query (uncomment to test)
    query = "Who is Srihari Raman?"
    stream, sources = run_query(vectorstore, query)

    print("\n=== Answer ===")
    answer = "".join(
        chunk.choices[0].delta.get("content", "") for chunk in stream if chunk.choices[0].delta
    )
    print(answer)
    print("\n=== Sources ===")
    for src in sources:
        print(src)

