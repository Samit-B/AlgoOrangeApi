import os
import re
import fitz  # PyMuPDF
import chromadb
from chromadb.config import Settings
from uuid import uuid4
from fastapi import FastAPI, UploadFile, File, HTTPException, APIRouter
from fastapi.responses import JSONResponse
from io import BytesIO
import groq
from langchain_ollama import OllamaLLM
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import ollama
from app.domain.interfaces import Agent
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Initialize FastAPI and Router
app = FastAPI()
pdf2Router = APIRouter()

# Initialize embedding model
embedding_model = OllamaEmbeddings(model="llama3.1:8b")
# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chromaa_db")

# Store ChromaDB collections and track latest file
vector_db = {}
latest_file_id = None  # Track the latest uploaded PDF

import pdfplumber
from io import BytesIO

import tiktoken

# Groq uses OpenAI tokenization models like `cl100k_base` (same as gpt-3.5/4)
encoding = tiktoken.get_encoding("cl100k_base")


def chunk_text_by_tokens(text: str, max_tokens: int = 2000):
    tokens = encoding.encode(text)

    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk = tokens[i : i + max_tokens]
        chunk_text = encoding.decode(chunk)
        chunks.append(chunk_text)

    return chunks


latest_file_id = None


# def extract_and_store_pdf(pdf_bytes: bytes) -> str:
#     """Extracts text from a PDF and stores it in ChromaDB with structured metadata."""
#     global latest_file_id
#     file_id = str(uuid4())  # Generate a unique file ID
#     pages_data = []

#     try:
#         # Open the PDF document from bytes
#         with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
#             # Iterate through each page and extract text
#             for page_number, page in enumerate(pdf.pages, start=1):
#                 page_content = page.extract_text()
#                 if page_content and page_content.strip():
#                     pages_data.append(
#                         {"page_number": page_number, "text": page_content}
#                     )

#         # Store the structured data in ChromaDB
#         store_structured_data_in_chroma(file_id, pages_data)
#         latest_file_id = file_id
#         return file_id

#     except Exception as e:
#         return None


# def store_structured_data_in_chroma(file_id: str, pages_data: list):
#     """Stores structured PDF data in ChromaDB for retrieval."""
#     try:
#         # Create or get the collection for the file
#         collection = chroma_client.get_or_create_collection(
#             name=file_id, metadata={"hnsw:space": "cosine"}
#         )

#         batch_size = 50  # Adjust batch size for performance
#         for i in range(0, len(pages_data), batch_size):
#             batch = pages_data[i : i + batch_size]
#             _store_batch(collection, batch, file_id)

#     except Exception as e:
#         pass


# def _store_batch(collection, pages_data, file_id):
#     """Store each page's content in ChromaDB with structured metadata."""
#     if not pages_data:
#         return

#     # Filter out pages with None or empty text
#     valid_pages = [p for p in pages_data if p.get("text") and p["text"].strip()]

#     if not valid_pages:
#         return

#     ids = [str(uuid4()) for _ in valid_pages]
#     texts = [page["text"] for page in valid_pages]
#     try:
#         vectors = embedding_model.embed_query(texts)
#     except Exception as e:
#         print(f"Error embedding texts: {str(e)}")
#         return

#     metadatas = [
#         {"file_id": file_id, "page_number": page["page_number"], "text": page["text"]}
#         for page in valid_pages
#     ]


#     collection.add(
#         ids=ids,
#         embeddings=vectors,
#         documents=[metadata["text"] for metadata in metadatas],
#         metadatas=metadatas,
#     )
def extract_and_store_pdf(pdf_bytes: bytes) -> str:
    """Extract full text, generate a summary, and store both in ChromaDB."""
    file_id = str(uuid4())
    global latest_file_id
    latest_file_id = file_id

    try:
        # Step 1: Extract full text from PDF
        full_text = ""
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                page_content = page.extract_text()
                if page_content:
                    full_text += page_content + "\n"

        if not full_text.strip():
            print("❌ PDF is empty.")
            return None

        # Step 2: Generate summary using Ollama or Groq LLM
        summary_prompt = f"""
        Summarize the following Information Security Manual. 
        Keep it concise and highlight core principles, roles, frameworks, and any important sections or responsibilities.

        PDF TEXT:
        {full_text}  # limit to 5K characters for summarization
        """

        ollama_llm = OllamaLLM(
            model="llama3.1:8b", temperature=0.1
        )  # Initialize OllamaLLM
        summary = ollama_llm.invoke(input=summary_prompt).strip()

        # Step 3: Embed both full text and summary
        vector_full = embedding_model.embed_query(full_text)
        vector_summary = embedding_model.embed_query(summary)

        # Step 4: Store both in ChromaDB
        collection = chroma_client.get_or_create_collection(
            name="pdf_with_summary", metadata={"hnsw:space": "cosine"}
        )

        collection.add(
            ids=[str(uuid4()), str(uuid4())],
            documents=[full_text, summary],
            embeddings=[vector_full, vector_summary],
            metadatas=[
                {
                    "file_id": file_id,
                    "type": "full_text",
                    "length": len(full_text.split()),
                },
                {"file_id": file_id, "type": "summary", "length": len(summary.split())},
            ],
        )

        print(f"✅ Stored full text and summary for file_id: {file_id}")
        return file_id

    except Exception as e:
        print(f"❌ Error: {e}")
        return None


@pdf2Router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Uploads a PDF, extracts text, stores it in ChromaDB, and assigns a file_id."""
    try:

        pdf_bytes = await file.read()

        # Extract text from PDF
        extracted_text = extract_and_store_pdf(pdf_bytes)

        if not extracted_text.strip():
            raise HTTPException(
                status_code=400, detail="No extractable text found in the PDF."
            )

        return JSONResponse(content={"message": "PDF uploaded and processed."})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


class PdfAgent(Agent):
    def __init__(self):
        self.client = groq.Client(api_key=os.getenv("GROQ_API_KEY"))
        self.model_name = "llama3.1:8b"
        self.embeddings = OllamaEmbeddings(model=self.model_name)
        self.llm = OllamaLLM(model=self.model_name, temperature=0.1)

    async def handle_query(self, userChatQuery: str, userChatHistory: str) -> str:
        if latest_file_id is None:
            return "No PDF found. Please upload a document first."

        try:
            # Create a vector store from the ChromaDB collection
            vectorstore = Chroma(
                collection_name="pdf_with_summary",  # Fixed
                embedding_function=self.embeddings,
                persist_directory="./chroma_db",
            )

            # Create a retriever
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": 25,
                    "filter": {
                        "file_id": latest_file_id
                    },  # ✅ Only get chunks from this file
                },
            )

            # Create a prompt template
            prompt = PromptTemplate.from_template(
                """
                You are a helpful assistant analyzing PDF content.

                PDF CONTENT:
                {context}

                USER QUERY: {input}

                Your task is to extract JSON-formatted compliance clauses related to the user query.
                Extract the following details from the PDF content:
                - Category-extract from the PDF content
                - Requirement-extract from the PDF content
                - Risk Level-extract from the PDF content
                - Compliance Criteria-extract from the PDF content

                ✅ Return only the relevant clauses based on the user query in the same format.
                🚫 Do not include any text outside of the JSON format.
                """
            )

            # Create document chain
            document_chain = create_stuff_documents_chain(self.llm, prompt)

            # Create retrieval chain
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            # Execute the query
            response = retrieval_chain.invoke({"input": userChatQuery})

            # Extract answer from response
            answer = response.get("answer", "")

            return answer
        except Exception as e:
            import traceback

            trace = traceback.format_exc()
            return f"Error processing query: {str(e)}\n{trace}"


@pdf2Router.get("/query")
async def query_handler(userChatQuery: str):

    work_agent = PdfAgent()
    response = await work_agent.handle_query(userChatQuery, userChatHistory="")
    return {"response": response}


app.include_router(pdf2Router)

# Include router
app.include_router(pdf2Router)
