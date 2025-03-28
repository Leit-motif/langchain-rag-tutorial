# from langchain.document_loaders import DirectoryLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
# Use the official embeddings module updated for Pydantic v2:
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import openai
from dotenv import load_dotenv
import os
import shutil
import re
from datetime import datetime

# Load environment variables. Assumes that project contains .env file with API keys
load_dotenv()
#---- Set OpenAI API key 
# Change environment variable name from "OPENAI_API_KEY" to the name given in your .env file.
openai.api_key = os.environ['OPENAI_API_KEY']

CHROMA_PATH = "chroma"
DATA_PATH = "data/books"


def main():
    generate_data_store()


def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    
    # Enhance metadata with date and tag information
    for doc in documents:
        filename = os.path.basename(doc.metadata["source"])
        # Extract date
        date_match = re.search(r"\d{4}-\d{2}-\d{2}", filename)
        if date_match:
            doc.metadata["date"] = date_match.group(0)  # String format
            doc.metadata["year"] = date_match.group(0)[:4]
            doc.metadata["month"] = date_match.group(0)[5:7]
            doc.metadata["day"] = date_match.group(0)[8:10]
            doc.metadata["datetime"] = doc.metadata["date"]  # Keep as a string
        
        # Extract tags and store as comma-separated string
        tags = re.findall(r'\[\[(.*?)\]\]', doc.page_content)
        doc.metadata["tags"] = ",".join([tag.strip() for tag in tags]) if tags else ""

    return documents


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks


def save_to_chroma(chunks: list[Document]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents.
    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )
    # db.persist() Chroma now auto-persists documents 
    # so you can safely remove or comment out db.persist() 
    # if you wish to avoid the warning. 
    # For learning purposes, it's okay to leave it for now 
    # since it doesn't break the code.
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


if __name__ == "__main__":
    main()
