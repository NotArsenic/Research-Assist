import os
import re
import arxiv
from datetime import datetime, timedelta


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.join(BASE_DIR, "papers")
os.makedirs(PAPER_DIR, exist_ok=True)


def _sanitize_filename(title: str) -> str:
    """
    Removes unsafe characters from the title for file saving.

    Args:
        title: The input string to sanitize.
    Returns:
        A sanitized string safe for filenames.
    """

    clean = re.sub(r'[<>:"/\\|?*]', "", title)
    return f"{clean}.pdf"


def _format_date(date_str: str) -> str:
    """
    Converts YYYY-MM-DD to YYYYMMDDHHMM format which is required for arXiv

    Args:
        date_str: The input date string in YYYY-MM-DD format.
    """
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.strftime("%Y%m%d0000")

    except ValueError:
        print(
            f"Error: Invalid date format for '{date_str}'. \nIgnoring the date filter."
        )
        return "*"


def _search_and_download(
    topic: str, limit: int = 1, start_date: str = None, end_date: str = None
):
    """
    Searches for papers on arXiv and downloads it.
    Returns the metadata directory of the downloaded paper.

    Args:
        topic: The main search topic.
        limit: Max papers to download.
        start_date: "YYYY-MM-DD" (Search papers AFTER this date).
        end_date: "YYYY-MM-DD" (Search papers BEFORE this date).
    """

    date_query = ""

    if start_date or end_date:
        start_dt = _format_date(start_date) if start_date else "000000000000"

        if end_date:
            end_dt = _format_date(end_date)

        else:
            end_dt = (datetime.now() + timedelta(days=1)).strftime("%Y%m%d0000")

    date_query = f" AND submittedDate:[{start_dt} TO {end_dt}]"

    full_query = f"{topic}{date_query}"
    print(f"Searching arXiv with query: '{full_query}'")

    client = arxiv.Client()
    search = arxiv.Search(
        query=full_query, max_results=limit, sort_by=arxiv.SortCriterion.Relevance
    )

    downloaded_papers = []

    try:
        results_list = list(client.results(search))

        if not results_list:
            print(f"No papers found for the query {full_query}.")
            return []

        for result in results_list:
            filename = _sanitize_filename(result.title)
            filepath = os.path.join(PAPER_DIR, filename)

            if not os.path.exists(filepath):
                print(f"Downloading paper: '{result.title}'")
                result.download_pdf(dirpath=PAPER_DIR, filename=filename)
            else:
                print(f"Paper already exists: '{result.title}'")

            paper_metadata = {
                "title": result.title,
                "path": filepath,
                "authors": [str(author) for author in result.authors],
                "published_date": result.published.date(),
                "summary": result.summary,
            }

            downloaded_papers.append(paper_metadata)

        return downloaded_papers

    except Exception as e:
        print(f"Error during arXiv search/download: {e}")
        return []


def _download_by_arxiv_id(arxiv_identifier: str):
    """
    Downloads a paper by its arXiv ID or URL.

    Args:
        arxiv_identifier: The arXiv ID (e.g., "1706.03762" or "arXiv:1706.03762") or full URL.
    Returns:
        The metadata of the downloaded paper or None if download failed.
    """

    if "arxiv.org" in arxiv_identifier:
        match = re.search(r"arxiv\.org/(?:abs|pdf)/(\d+\.\d+)", arxiv_identifier)
        if match:
            arxiv_id = match.group(1)
        else:
            print(f"Invalid arXiv URL format: {arxiv_identifier}")
            return None
    elif arxiv_identifier.startswith("arXiv:"):
        arxiv_id = arxiv_identifier[6:]
    else:
        arxiv_id = arxiv_identifier

    try:
        client = arxiv.Client()
        search = arxiv.Search(id_list=[arxiv_id])
        result = next(client.results(search), None)

        if not result:
            print(f"Paper with ID '{arxiv_id}' not found on arXiv.")
            return None

        filename = _sanitize_filename(result.title)
        filepath = os.path.join(PAPER_DIR, filename)

        if not os.path.exists(filepath):
            print(f"Downloading paper: '{result.title}'")
            result.download_pdf(dirpath=PAPER_DIR, filename=filename)
        else:
            print(f"Paper already exists: '{result.title}'")

        paper_metadata = {
            "title": result.title,
            "path": filepath,
            "authors": [str(author) for author in result.authors],
            "published_date": result.published.date(),
            "summary": result.summary,
            "arxiv_id": arxiv_id,
        }

        return paper_metadata

    except Exception as e:
        print(f"Error downloading paper with ID '{arxiv_id}': {e}")
        return None


def _get_papers():
    """
    Retrieves all papers in the PAPER_DIR and returns their metadata.
    """

    if not os.path.exists(PAPER_DIR):
        print("Paper directory does not exist.")
        return []

    return [f for f in os.listdir(PAPER_DIR) if f.endswith(".pdf")]


def _delete_all_papers():
    """
    Deletes all papers from the PAPER_DIR.
    """

    if not os.path.exists(PAPER_DIR):
        print("Paper directory does not exist.")
        return

    papers = _get_papers()
    if not papers:
        print("No papers to delete.")
        return

    for paper in papers:
        filepath = os.path.join(PAPER_DIR, paper)
        try:
            os.remove(filepath)
            print(f"Deleted: {paper}")
        except Exception as e:
            print(f"Error deleting {paper}: {e}")

    print(f"Deleted {len(papers)} paper(s) total.")


if __name__ == "__main__":

    print("--- Testing Date Filter ---")
    papers = _search_and_download(
        topic="Transformers", limit=2, start_date="2024-01-01"
    )

    if papers:
        print(f"Downloaded: {papers[0]['title']}")
        print(f"Published Date: {papers[0]['published_date']}")

    print("\n--- Testing Download by arXiv ID ---")
    paper = _download_by_arxiv_id("1706.03762")
    if paper:
        print(f"Downloaded: {paper['title']}")

    all_papers = _get_papers()
    print(f"\nTotal papers in directory: {len(all_papers)}")
    for paper in all_papers:
        print(f"  - {paper}")

    print("\n press 'x' to delete all papers, any other key to exit.")

    choice = input().strip().lower()

    if choice == "x":
        _delete_all_papers()

    else:
        print("Exiting without deleting papers.")
