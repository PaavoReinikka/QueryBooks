import spacy
import re
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    TokenTextSplitter
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import DeterministicFakeEmbedding


def clean_mid_sentence_newlines(text):
    """
    Cleans text where sentences are broken across lines (common in PDFs).
    Replaces single newlines with spaces, but preserves double newlines (paragraphs).
    """
    # Replace single newline with space, but keep double or more newlines
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    # Collapse multiple spaces
    text = re.sub(r' +', ' ', text)
    return text.strip()

# --- OPTIONS FOR CHUNKING ---

def chunk_by_character(text, chunk_size=1000, chunk_overlap=100):
    """
    Splits text into fixed character counts. 
    Good for simple, uniform data but may cut words or sentences.
    """
    splitter = CharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        separator=""
    )
    return [d.page_content for d in splitter.create_documents([text])]

def chunk_recursively(text, chunk_size=1000, chunk_overlap=100, pre_clean=True):
    """
    The most common LangChain splitter. Tries to split on paragraphs, 
    then sentences, then words to keep semantic content together.
    
    If pre_clean=True, mid-sentence newlines (common in PDFs) are removed first.
    """
    if pre_clean:
        text = clean_mid_sentence_newlines(text)
        
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Priority sequence: Paragraphs > Sentences > Single Newlines > Words
        separators=["\n\n", ". ", "\n", " ", ""]
    )
    return [d.page_content for d in splitter.create_documents([text])]

def chunk_markdown(text):
    """
    Specifically for Markdown files. Splits based on headers (#, ##, ###).
    Useful for technical documentation.
    """
    headers = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
    # Note: This returns Document objects with metadata about the header
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)
    return [d.page_content for d in splitter.split_text(text)]

def chunk_by_token(text, model_name="gpt-3.5-turbo", chunk_size=500, chunk_overlap=50):
    """
    Splits based on token counts. Essential for LLM context windows.
    Requires 'tiktoken' for OpenAI models.
    """
    splitter = TokenTextSplitter(
        model_name=model_name,
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    return [d.page_content for d in splitter.create_documents([text])]

def chunk_semantically_spacy(text, max_sentences=5, spacy_model="en_core_web_sm", pre_clean=True):
    """
    Linguistic chunking using spaCy (Best for Knowledge Graphs).
    Ensures that triples ARE NOT split across chunks.
    """
    if pre_clean:
        text = clean_mid_sentence_newlines(text)
        
    nlp = spacy.load(spacy_model, disable=["ner", "lemmatizer", "textcat"])
    # Handle potential large string issues
    nlp.max_length = max(nlp.max_length, len(text) + 100)
    
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    chunks = []
    
    for i in range(0, len(sentences), max_sentences):
        chunk = " ".join(sentences[i : i + max_sentences])
        chunks.append(chunk)
        
    return chunks


# If you want to use real embeddings, you can use:
# from langchain_huggingface import HuggingFaceEmbeddings
# or any other LangChain embedding provider.

def chunk_semantically_embeddings(text, embeddings=None, breakpoint_threshold_type="percentile"):
    """
    LangChain's Experimental Semantic Chunker.
    It looks at the cosine distance between sentence embeddings and splits
    when the difference exceeds a threshold.
    
    threshold_types: "percentile", "standard_deviation", "interquartile"
    """
    if embeddings is None:
        print("No embeddings provided, using DeterministicFakeEmbedding for demonstration/development purposes.")
        # In a real app, pass an Embeddings Interface compatible model here (e.g., HuggingFaceEmbeddings)
        embeddings = DeterministicFakeEmbedding(size=384)
    
    splitter = SemanticChunker(
        embeddings, 
        breakpoint_threshold_type=breakpoint_threshold_type
    )
    
    return [d.page_content for d in splitter.create_documents([text])]




def normalize_pdf_markdown(text: str) -> str:
    # Normalize whitespace
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)

    lines = [ln.strip() for ln in text.split("\n")]
    paragraphs, buffer = [], []

    for ln in lines:
        if not ln:
            if buffer:
                paragraphs.append(" ".join(buffer))
                buffer = []
            continue

        is_heading = ln.startswith("#") or bool(re.fullmatch(r"[A-Z0-9 ,./()\-]{8,}", ln))
        is_list = bool(re.match(r"^([-*+]|\d+[.)])\s+", ln))

        if is_heading or is_list:
            if buffer:
                paragraphs.append(" ".join(buffer))
                buffer = []
            paragraphs.append(ln)
            continue

        buffer.append(ln)

        # If line ends like a sentence, flush paragraph candidate
        if re.search(r"[.!?:;]$", ln):
            paragraphs.append(" ".join(buffer))
            buffer = []

    if buffer:
        paragraphs.append(" ".join(buffer))

    cleaned = "\n\n".join(p.strip() for p in paragraphs if p.strip())
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned
