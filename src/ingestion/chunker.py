from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_text(text, source_name):
    """
    Splits cleaned text into overlapping chunks with metadata.
    """
    # Create the splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )

    # Split the text into chunks
    chunks = splitter.split_text(text)

    # Add metadata to each chunk
    chunks_with_metadata = []
    for i, chunk in enumerate(chunks):
        chunks_with_metadata.append({
            "text": chunk,
            "metadata": {
                "source": source_name,
                "chunk_id": i
            }
        })

    return chunks_with_metadata