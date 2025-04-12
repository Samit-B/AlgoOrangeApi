import os
import fitz  # PyMuPDF
import chromadb
from chromadb.config import Settings
from uuid import uuid4
from fastapi import FastAPI, UploadFile, File, HTTPException, APIRouter
from fastapi.responses import JSONResponse
from io import BytesIO
import groq
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from app.domain.interfaces import Agent

# Initialize FastAPI and Router
app = FastAPI()
pdfRouter = APIRouter()

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

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


# def extract_text_from_pdf(pdf_bytes: bytes) -> str:
#     """Extracts text from a PDF file using pdfplumber."""
#     text = ""
#     try:
#         # Open the PDF document from bytes
#         with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
#             # Iterate through each page and extract text
#             for page in pdf.pages:
#                 text += page.extract_text() + "\n"
#     except Exception as e:
#         print(f"Error extracting text: {str(e)}")

#     return text.strip() if text else "Failed to extract text."


# def store_text_in_chroma(file_id: str, text: str):
#     """Stores extracted text in ChromaDB for retrieval."""
#     global latest_file_id
#     try:
#         text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#         documents = text_splitter.create_documents([text])

#         # Create or get the collection for the file
#         collection = chroma_client.get_or_create_collection(
#             name=file_id, metadata={"hnsw:space": "cosine"}
#         )

#         # Add documents to the collection in batches
#         _store_batch(collection, documents)

#         latest_file_id = file_id  # Set as the latest uploaded file
#         print(f"Text from {file_id} stored in ChromaDB successfully.")
#     except Exception as e:
#         print(f"Error storing text in ChromaDB: {str(e)}")


# def _store_batch(collection, pages_data, file_id):
#     """Store each page's content in ChromaDB with structured metadata."""
#     if not pages_data:
#         return

#     # Extract text and metadata for each page
#     ids = [str(uuid4()) for _ in pages_data]
#     texts = [page["text"] for page in pages_data]
#     vectors = embedding_model.embed_documents(texts)

#     # Include metadata with file_id and page_number
#     metadatas = [
#         {"file_id": file_id, "page_number": page["page_number"], "text": page["text"]}
#         for page in pages_data
#     ]

#     # Store in ChromaDB
#     collection.add(ids=ids, embeddings=vectors, metadatas=metadatas)

#     for i in range(len(ids)):
#         print(f"Stored in ChromaDB: ID={ids[i]}, Metadata={metadatas[i]}")

# Global variable to track the latest file ID
latest_file_id = None


def extract_and_store_pdf(pdf_bytes: bytes) -> str:
    """Extracts text from a PDF and stores it in ChromaDB with structured metadata."""
    global latest_file_id
    file_id = str(uuid4())  # Generate a unique file ID
    pages_data = []

    try:
        # Open the PDF document from bytes
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            # Iterate through each page and extract text
            for page_number, page in enumerate(pdf.pages, start=1):
                page_content = page.extract_text()
                if page_content and page_content.strip():
                    pages_data.append(
                        {"page_number": page_number, "text": page_content}
                    )

        # Store the structured data in ChromaDB
        store_structured_data_in_chroma(file_id, pages_data)
        latest_file_id = file_id
        return file_id

    except Exception as e:
        return None


def store_structured_data_in_chroma(file_id: str, pages_data: list):
    """Stores structured PDF data in ChromaDB for retrieval."""
    try:
        # Create or get the collection for the file
        collection = chroma_client.get_or_create_collection(
            name=file_id, metadata={"hnsw:space": "cosine"}
        )

        batch_size = 50  # Adjust batch size for performance
        for i in range(0, len(pages_data), batch_size):
            batch = pages_data[i : i + batch_size]
            _store_batch(collection, batch, file_id)

    except Exception as e:
        pass


def _store_batch(collection, pages_data, file_id):
    """Store each page's content in ChromaDB with structured metadata."""
    if not pages_data:
        return

    # Extract text and metadata for each page
    ids = [str(uuid4()) for _ in pages_data]
    texts = [page["text"] for page in pages_data]
    vectors = embedding_model.embed_documents(texts)

    # Include metadata with file_id and page_number
    metadatas = [
        {"file_id": file_id, "page_number": page["page_number"], "text": page["text"]}
        for page in pages_data
    ]

    # Store in ChromaDB
    collection.add(ids=ids, embeddings=vectors, metadatas=metadatas)


@pdfRouter.post("/upload")
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

    async def handle_query(self, userChatQuery: str, userChatHistory: str) -> str:
        if latest_file_id is None:
            return "No PDF found. Please upload a document first."

        collection = chroma_client.get_collection(name=latest_file_id)
        if collection is None:
            return "No PDF found. Please upload a document first."

        # Step 1: Get relevant context from ChromaDB
        results = collection.query(query_texts=[userChatQuery], n_results=50)
        all_texts = [doc["text"] for doc in results["metadatas"][0]]
        full_context = "\n".join(all_texts)

        # Step 2: Chunk the context
        context_chunks = chunk_text_by_tokens(full_context, max_tokens=1800)

        # Step 3: Process each chunk individually
        partial_answers = []
        for i, chunk in enumerate(context_chunks):
            chunk_prompt = (
                f"You are a helpful assistant. A user has a question about the following PDF content:\n\n"
                f"{chunk}\n\n"
                f"User's question: {userChatQuery}\n"
                f"Give a clear, concise, and well-structured answer based only on the content provided above. Avoid repeating or labeling chunk numbers."
                f"Omit any statements about content not being found or missing."
            )

            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a PDF assistant who answers based only on the given content.",
                    },
                    {"role": "user", "content": chunk_prompt},
                ],
                max_tokens=400,
            )

            partial_answer = response.choices[0].message.content.strip()
            partial_answers.append(partial_answer)

        # Step 4: Combine all partial answers and generate final response
        combined_summary = "".join(partial_answers)

        return combined_summary


@pdfRouter.get("/query")
async def query_handler(userChatQuery: str):

    work_agent = PdfAgent()
    response = await work_agent.handle_query(userChatQuery, userChatHistory="")
    return {"response": response}


app.include_router(pdfRouter)

# Include router
app.include_router(pdfRouter)
