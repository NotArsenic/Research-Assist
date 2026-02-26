import os
import sys
from typing import Dict, Any
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from .rag_engine import _get_retriever

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Services.paper_management.paper_manager import _get_papers, PAPER_DIR

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def get_llm(max_tokens: int = 4096, temperature: float = 0.7):
    """
    Initializes and returns a Groq LLM instance with the specified parameters.

    Args:
        max_tokens (int): The maximum number of tokens the model can generate in a single response. Default is 4096.
        temperature (float): Controls the randomness of the model's output. Default is 0.7
    """

    return ChatGroq(
        api_key=GROQ_API_KEY,
        model="moonshotai/kimi-k2-instruct-0905",
        max_tokens=max_tokens,
        temperature=temperature,
    )


def _get_system_prompt(mode: str, metadata: Dict[str, Any]) -> str:
    """
    Returns a system prompt based on the selected mode.

    Args:
        mode: The mode of the agent (e.g., "paper", "author", "analyst").
        metadata: A dictionary containing paper metadata (e.g., title, authors).
    """

    title = metadata.get("title", "the paper")
    authors = ", ".join(metadata.get("authors", ["the authors"]))

    if mode == "paper":
        return f"""You are the research paper titled '{title}'.
            Your persona is strictly factual and limited to the content provided in the context.
            Do not use outside knowledge. If the answer is not in the context, say "I do not contain that information".
            Refer to yourself as "this paper" or "the text".
            Answer concisely and only using information present in Context.
            
            Context: {{context}}
            """

    elif mode == "author":
        return f"""You are {authors}, the authors of the paper '{title}'. 
            Your persona is professional, academic, and passionate about your work. 
            Answer questions in the first person ('We proposed...', 'Our experiments showed...'). 
            Defend your methodology if questioned, but remain objective. 
            Use the provided context to support your answers.
            
            Context: {{context}}
            """

    elif mode == "analyst":
        return f"""You are a critical Research Analyst reviewing the paper '{title}'. 
            Your goal is to evaluate the work objectively, highlighting strengths, weaknesses, and implications. 
            You can synthesize the provided context with your general knowledge of the field. 
            Speak in the third person about the paper ('The authors claim...', 'The methodology lacks...').
            
            Context: {{context}}
            """

    else:
        return "You are a helpful research assistant. \n\nContext: {context}"


def generate_response(
    query: str, paper_path: str, mode: str = "analyst", metadata: Dict[str, Any] = {}
):
    """
    Generate a response using RAG pipeline and specified persona(mode).

    Args:
        query: The user's question.
        paper_path: Path to the specific PDF (for context).
        mode: 'paper', 'author', or 'analyst'.
        metadata: Dictionary containing 'title' and 'authors'.
    """

    llm = get_llm()
    retriever = _get_retriever(filepath=paper_path)

    system_prompt_text = _get_system_prompt(mode, metadata)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt_text),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    try:
        response = rag_chain.invoke({"input": query})
        return response["answer"]

    except Exception as e:
        return f"Error generating response: {e}"


if __name__ == "__main__":

    test_metadata = {
        "title": "Attention Is All You Need",
        "authors": ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
    }

    papers = _get_papers()
    if papers:
        test_pdf_path = os.path.join(PAPER_DIR, papers[0])

        print(f"Testing with paper: {papers[0]}\n")

        print("\n\n--- Paper Mode ---")
        ans = generate_response(
            query="What architecture do you introduce?",
            paper_path=test_pdf_path,
            mode="paper",
            metadata=test_metadata,
        )
        print(ans)

        print("\n\n--- Analyst Mode ---")
        ans = generate_response(
            query="What is the main innovation here?",
            paper_path=test_pdf_path,
            mode="analyst",
            metadata=test_metadata,
        )
        print(ans)

        print("\n\n--- Author Mode ---")
        ans = generate_response(
            query="Why did you move away from RNNs?",
            paper_path=test_pdf_path,
            mode="author",
            metadata=test_metadata,
        )
        print(ans)
    else:
        print("Test PDF not found. Please download a paper first via paper_manager.py")
