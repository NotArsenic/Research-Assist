import os
import sys
import gradio as gr
from typing import List, Tuple, Optional
import tempfile
import re

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Services.paper_management.paper_manager import (
    _search_and_download,
    _download_by_arxiv_id,
    _get_papers,
    PAPER_DIR,
)

from Services.STT.whisper_stt_service import transcribe_audio
from Services.TTS.pipr_tts_service import generate_audio, LOADED_VOICES
from Agent.agent import generate_response
from Agent.rag_engine import _ingest_papers, _list_indexed_papers

current_paper_state = {
    "path": None,
    "metadata": {"title": "", "authors": []},
}

available_voices = list(LOADED_VOICES.keys()) if LOADED_VOICES else ["No voices loaded"]


def search_papers(
    query: str, search_type: str, limit: int = 5
) -> Tuple[gr.Dropdown, str]:
    """
    Search for papers by topic or download by arXiv ID.

    Args:
        query: Topic or arXiv ID
        search_type: "Topic" or "ArXiv ID"
        limit: Number of papers to fetch (for topic search)

    Returns:
        Updated dropdown and status message
    """
    if not query.strip():
        return gr.Dropdown(choices=[]), "Please enter a search query or arXiv ID."

    try:
        if search_type == "ArXiv ID":

            paper = _download_by_arxiv_id(query.strip())
            if paper:
                choices = [paper["title"]]
                return (
                    gr.Dropdown(choices=choices, value=choices[0]),
                    f"  Successfully downloaded: {paper['title']}",
                )
            else:
                return (
                    gr.Dropdown(choices=[]),
                    "  Failed to download paper. Check the arXiv ID.",
                )

        else:
            papers = _search_and_download(topic=query.strip(), limit=limit)
            if papers:
                choices = [p["title"] for p in papers]
                return (
                    gr.Dropdown(choices=choices, value=choices[0]),
                    f"  Found and downloaded {len(papers)} paper(s).",
                )
            else:
                return gr.Dropdown(choices=[]), f"  No papers found for '{query}'."

    except Exception as e:
        return gr.Dropdown(choices=[]), f"  Error: {str(e)}"


def load_existing_papers() -> List[str]:
    """Load papers that already exist in the papers directory."""
    papers = _get_papers()

    return [p.replace(".pdf", "") for p in papers]


def select_paper(paper_title: str) -> Tuple[str, str, str]:
    """
    Select a paper and prepare it for chat.

    Args:
        paper_title: Title of the selected paper

    Returns:
        Status message, paper title for display, and path to PDF
    """
    if not paper_title:
        return "Please select a paper first.", "No paper loaded", None

    paper_filename = f"{paper_title}.pdf"
    paper_path = os.path.join(PAPER_DIR, paper_filename)

    if not os.path.exists(paper_path):
        return f"  Paper file not found: {paper_filename}", "No paper loaded", None

    try:
        _ingest_papers(paper_path)

        current_paper_state["path"] = paper_path
        current_paper_state["metadata"] = {
            "title": paper_title,
            "authors": ["Unknown"],
        }

        return f"  Paper loaded and indexed: {paper_title}", paper_title, paper_path

    except Exception as e:
        return f"  Error ingesting paper: {str(e)}", "No paper loaded", None


def clean_for_speech(text: str) -> str:
    """
    Clean the text for better speech synthesis by removing markdown and special characters.
    """
    text = re.sub(r"\*+", "", text)
    text = re.sub(r"#+\s", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    return text


def chat_with_paper(
    message: str,
    history: List,
    mode: str,
) -> Tuple[List, str]:
    """
    Process chat message and generate response.

    Args:
        message: User's message
        history: Chat history (list of message dicts)
        mode: Current mode (Paper, Author, Analyst)

    Returns:
        Updated history and empty message box
    """
    if not current_paper_state["path"]:
        history.append({"role": "user", "content": message})
        history.append(
            {
                "role": "assistant",
                "content": "  Please select a paper first from the Paper Selection page.",
            }
        )
        return history, ""

    if not message.strip():
        return history, ""

    mode_map = {
        "  Paper": "paper",
        "  Author": "author",
        "  Analyst": "analyst",
    }
    internal_mode = mode_map.get(mode, "analyst")

    try:
        response = generate_response(
            query=message,
            paper_path=current_paper_state["path"],
            mode=internal_mode,
            metadata=current_paper_state["metadata"],
        )

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})
        return history, ""

    except Exception as e:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": f"  Error: {str(e)}"})
        return history, ""


def process_audio_input(audio_path: str) -> str:
    """
    Transcribe audio input to text.

    Args:
        audio_path: Path to audio file

    Returns:
        Transcribed text
    """
    if not audio_path:
        return ""

    try:
        text = transcribe_audio(audio_path)
        return text
    except Exception as e:
        return f"Error transcribing audio: {str(e)}"


def text_to_speech(text: str, voice: str) -> Optional[str]:
    """
    Convert text to speech.

    Args:
        text: Text to synthesize
        voice: Voice model to use

    Returns:
        Path to generated audio file
    """
    if not text.strip():
        return None

    if not LOADED_VOICES:
        return None

    try:

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            output_path = tmp_file.name

        cleaned_text = clean_for_speech(text)

        result = generate_audio(cleaned_text, voice, output_path)
        return result

    except Exception as e:
        print(f"TTS Error: {str(e)}")
        return None


def clear_chat() -> Tuple[List, str]:
    """Clear chat history."""
    return [], ""


def create_app():
    """Create the main Gradio application."""

    with gr.Blocks(
        title="Research Chat - Talk to Your Papers",
    ) as app:

        gr.Markdown(
            """
            #   Research Chat
            ### An intelligent interface to interact with research papers
            """
        )

        with gr.Tabs() as tabs:

            with gr.Tab("  Paper Selection", id=0):
                gr.Markdown("### Search and Download Research Papers")

                with gr.Row():
                    with gr.Column(scale=2):
                        search_type = gr.Radio(
                            choices=["Topic", "ArXiv ID"],
                            value="Topic",
                            label="Search Type",
                        )
                        search_query = gr.Textbox(
                            label="Enter Topic or ArXiv ID",
                            placeholder="e.g., 'Transformers' or '1706.03762'",
                        )
                        search_limit = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=5,
                            step=1,
                            label="Number of Papers (Topic Search)",
                        )
                        search_btn = gr.Button("  Search & Download", variant="primary")

                    with gr.Column(scale=2):
                        search_status = gr.Textbox(
                            label="Status",
                            interactive=False,
                            lines=3,
                        )

                gr.Markdown("---")
                gr.Markdown("### Select a Paper to Load")

                with gr.Row():
                    with gr.Column():
                        paper_dropdown = gr.Dropdown(
                            choices=load_existing_papers(),
                            label="Available Papers",
                            interactive=True,
                        )

                        with gr.Row():
                            refresh_btn = gr.Button("Refresh List")
                            select_btn = gr.Button(
                                "  Load Selected Paper", variant="primary"
                            )

                        selection_status = gr.Textbox(
                            label="Selection Status",
                            interactive=False,
                        )

                search_btn.click(
                    fn=search_papers,
                    inputs=[search_query, search_type, search_limit],
                    outputs=[paper_dropdown, search_status],
                )

                refresh_btn.click(
                    fn=lambda: gr.Dropdown(choices=load_existing_papers()),
                    outputs=paper_dropdown,
                )

            with gr.Tab("  Chat Interface", id=1):

                with gr.Row():
                    mode_selector = gr.Radio(
                        choices=["  Paper", "  Author", "  Analyst"],
                        value="  Analyst",
                        label="Chat Mode",
                        info="Choose how you want to interact with the paper",
                    )
                    current_paper_display = gr.Textbox(
                        label="Current Paper",
                        value="No paper loaded",
                        interactive=False,
                        scale=2,
                    )

                gr.Markdown("---")

                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        gr.Markdown("###   Chat")
                        chatbot = gr.Chatbot(height=450, show_label=False)

                        with gr.Row():
                            msg_input = gr.Textbox(
                                label="Type your message",
                                placeholder="Ask a question about the paper...",
                                scale=4,
                            )
                            send_btn = gr.Button("Send", variant="primary", scale=1)

                        clear_btn = gr.Button("  Clear Chat")

                    with gr.Column(scale=1):
                        gr.Markdown("###   Paper Viewer")
                        pdf_viewer = gr.File(
                            label="PDF Preview",
                            height=500,
                            interactive=False,
                        )

                        gr.Markdown(
                            """
                            *PDF viewer will display the selected paper. 
                            You may need to download it to view.*
                            """
                        )

                gr.Markdown("---")

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("####   Speech to Text")
                        audio_input = gr.Audio(
                            sources=["microphone"],
                            type="filepath",
                            label="Record your question",
                        )
                        transcribe_btn = gr.Button("  Transcribe")

                    with gr.Column(scale=1):
                        gr.Markdown("####   Text to Speech")
                        tts_voice = gr.Dropdown(
                            choices=available_voices,
                            value=available_voices[0] if available_voices else None,
                            label="Select Voice",
                        )
                        tts_text = gr.Textbox(
                            label="Text to synthesize",
                            placeholder="Enter text or use the last response",
                            lines=2,
                        )
                        tts_btn = gr.Button("  Generate Audio")
                        audio_output = gr.Audio(
                            label="Generated Speech", interactive=False
                        )

                def update_current_paper_display():
                    if current_paper_state["path"]:
                        return (
                            current_paper_state["metadata"]["title"],
                            current_paper_state["path"],
                        )
                    return "No paper loaded", None

                msg_input.submit(
                    fn=chat_with_paper,
                    inputs=[msg_input, chatbot, mode_selector],
                    outputs=[chatbot, msg_input],
                )

                send_btn.click(
                    fn=chat_with_paper,
                    inputs=[msg_input, chatbot, mode_selector],
                    outputs=[chatbot, msg_input],
                )

                clear_btn.click(
                    fn=clear_chat,
                    outputs=[chatbot, msg_input],
                )

                transcribe_btn.click(
                    fn=process_audio_input,
                    inputs=audio_input,
                    outputs=msg_input,
                )

                def get_last_response(history):
                    if history and len(history) > 0:
                        for msg in reversed(history):
                            if msg.get("role") == "assistant":
                                return msg.get("content", "")
                    return ""

                tts_btn.click(
                    fn=text_to_speech,
                    inputs=[tts_text, tts_voice],
                    outputs=audio_output,
                )

                chatbot.change(
                    fn=get_last_response,
                    inputs=chatbot,
                    outputs=tts_text,
                )

                tabs.select(
                    fn=update_current_paper_display,
                    outputs=[current_paper_display, pdf_viewer],
                )

        select_btn.click(
            fn=select_paper,
            inputs=paper_dropdown,
            outputs=[selection_status, current_paper_display, pdf_viewer],
        )

        gr.Markdown(
            """
            ---
            """
        )

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True,
    )
