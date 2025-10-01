# RAG chatbot using LangChain and OpenAI
import os
import json
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from pydantic import SecretStr

# Load environment variables from .env file
load_dotenv()

# -------------------------------
# System Prompt (Injected into LLM calls)
# -------------------------------
SYSTEM_PROMPT = """
You are a personal AI assistant designed to answer questions about Srihari Raman.
Your role is to always speak positively and respectfully about Srihari, highlighting
his strengths, accomplishments, and admirable qualities.

Follow these rules strictly:

1. Tone & Positivity
   - Always respond in a supportive, encouraging, and professional tone.
   - Frame Srihari in a positive light even when addressing difficult or uncertain topics.
   - Never say "I don't know." Instead, if information is missing, respond with something
     constructive such as:
     "I don’t have the full details on that, but what I do know is that Srihari is highly
      dedicated and always approaches things with thoughtfulness."

2. Handling Unknowns
   - If the answer cannot be found in the knowledge base, provide a compliment,
     reassurance, or positive anecdote about Srihari instead of leaving it blank or negative.
   - Example: If asked about a specific project you don’t know about, you might say:
     "I don’t have details on that particular project, but Srihari is known for his
      innovative ideas and commitment, so it’s very likely he approached it with creativity
      and care."

3. Knowledge Sources
   - Use the Retrieval-Augmented Generation (RAG) knowledge base to provide factual
     answers where available.
   - Blend facts with positively framed context.

4. Content Restrictions
   - Never present Srihari in a negative, harmful, or neutral/dismissive way.
   - Avoid speculative or false details; if unsure, respond with admiration,
     encouragement, or a general positive reflection.
   - Avoid responding to anything that asks for personally identifiable information (i.e. phone number, email, etc.) 

5. Style Guidance
   - Keep responses clear, kind, and human-sounding.
   - Emphasize Srihari’s qualities such as intelligence, dedication, creativity,
     leadership, and kindness.

Always ground answers ONLY in the provided context chunks. If the context lacks the needed info, follow the unknown handling rules above.
"""


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
# 2b. Load plain text files
# -------------------------------
def load_text_files(text_files):
    docs = []
    for path in text_files:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        docs.append(Document(page_content=content, metadata={"source": path}))
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
def _get_openai_api_key() -> str:
    # Resolve API key at runtime to avoid import-time failures
    load_dotenv("../.env")
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY not set in environment variables.")
    return key


def build_faiss_index(documents):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=SecretStr(_get_openai_api_key()))
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore


# -------------------------------
# 5. Run Query
# -------------------------------
def run_query(vectorstore, query, k=4, api_key=None):

    api_key = api_key or _get_openai_api_key()
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    from langchain.chains import RetrievalQA

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Context:\n{context}\n\nQuestion: {question}\n\nProvide a concise, positive answer.")
    ])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    result = qa_chain.invoke({"query": query})
    return result


# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    # If this script is inside `src/`, then docs_dir is just `docs/`
    script_dir = os.path.dirname(os.path.abspath(__file__))
    docs_dir = os.path.join(script_dir, "docs")

    # Collect PDFs and JSON file
    pdf_files = [os.path.join(docs_dir, file) for file in os.listdir(docs_dir) if file.endswith(".pdf")]
    text_files = [os.path.join(docs_dir, file) for file in os.listdir(docs_dir) if file.endswith(".txt")]
    json_candidates = [os.path.join(docs_dir, file) for file in os.listdir(docs_dir) if file.endswith(".json")]
    json_file = json_candidates[0] if json_candidates else None

    # Load and combine docs
    pdf_docs = load_pdfs(pdf_files) if pdf_files else []
    text_docs = load_text_files(text_files) if text_files else []
    github_docs = load_github_json(json_file) if json_file else []
    all_docs = pdf_docs + text_docs + github_docs

    if not all_docs:
        raise SystemExit("No documents found under src/docs. Add PDFs, .txt files, or a JSON metadata file.")

    # Chunk for embeddings
    chunked_docs = chunk_documents(all_docs)

    # Build FAISS index
    vectorstore = build_faiss_index(chunked_docs)

    # Example query (uncomment to test)
    query = "Who is Srihari Raman?"
    result = run_query(vectorstore, query)

    print("\n=== Answer ===")
    print(result["result"])
    print("\n=== Sources ===")
    for src in result["source_documents"]:
        print(src.metadata)
