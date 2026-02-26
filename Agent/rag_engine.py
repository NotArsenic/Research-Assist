import os
import sys

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "chroma_db")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def _ingest_papers(pdf_path: str):
    """
    Loads a PDF, splits it into chunks, and creates a Chroma vector store.

    Args:
        pdf_path: The file path to the PDF to ingest.
    """

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
    )
    splits = text_splitter.split_documents(docs)

    Chroma.from_documents(
        documents=splits, embedding=embedding_model, persist_directory=DB_DIR
    )

    print(
        f"Successfully indexed and ingested {len(splits)} chunks from '{pdf_path}' into the Chroma vector store."
    )


def _get_retriever(filepath: str = None, k: int = 3):
    """
    Returns a Chroma retriever for the given PDF file. If the vector store doesn't exist, it will be created.
    Args:
        filepath: The file path to the PDF to create/get the vector store for.
        k: The number of top similar chunks to retrieve.
    """

    vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embedding_model)

    search_kwargs = {"k": k}

    if filepath:
        abs_path = os.path.abspath(filepath)
        search_kwargs["filter"] = {"source": abs_path}

    return vector_db.as_retriever(search_kwargs=search_kwargs)


def _list_indexed_papers():
    """
    Helper to see whats in our VectorDB
    """

    if not os.path.exists(DB_DIR):
        print("No vector store found. No papers have been indexed yet.")
        return []

    vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embedding_model)

    try:
        data = vector_db.get()
        metadatas = data["metadatas"]
        sources = set(m["source"] for m in metadatas if m and "source" in m)
        return [os.path.basename(s) for s in sources]

    except Exception:
        return []


if __name__ == "__main__":

    project_root = os.path.dirname(BASE_DIR)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from Services.paper_management.paper_manager import _get_papers, PAPER_DIR

    papers = _get_papers()
    if papers:
        test_pdf = os.path.join(PAPER_DIR, papers[0])

        _ingest_papers(test_pdf)

        print("\n--- Testing Retrieval ---")
        retriever = _get_retriever(filepath=test_pdf)

        try:
            docs = retriever.invoke("What is this paper about?")
        except Exception:
            docs = []

        if docs:
            print(f"Retrieved chunk: {docs[0].page_content[:200]}...")

        print("\n--- Indexed Papers ---")
        indexed_papers = _list_indexed_papers()
        print(f"Indexed papers: {indexed_papers}")

    else:
        print("No papers found. Run paper_manager.py first.")
