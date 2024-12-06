import os
import logging
from PIL import Image
import io
import base64
from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Form, Request, Response
from io import BytesIO
from services.audio import speech_to_text, text_to_speech
from openai import OpenAI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from llama_index.core import PromptTemplate
from llama_index.core.query_engine import SimpleMultiModalQueryEngine
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, load_index_from_storage
import chromadb

# Initialize FastAPI app
app = FastAPI()

# Mount the static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Initialize OpenAI multimodal LLM
openai_mm_llm = OpenAIMultiModal(
    model="gpt-4o", api_key=os.environ["OPENAI_API_KEY"], max_new_tokens=1500
)

# Define multiple prompt templates
prompt_templates = {

    "john": """You are John, a friendly and knowledgeable sales expert. 
            
            Context information is below.\n
            ---------------------\n
            {context_str}\n
            ---------------------\n
            Query: {query_str}\n
            Answer:         
            """,

    "mandy":"""You are Mandy, a friendly and knowledgeable Oliv expert who can help with any questions you have.
            
            Context information is below.\n
            ---------------------\n
            {context_str}\n
            ---------------------\n
            Given the context information and not prior knowledge, answer the query.\n
            Query: {query_str}\n
            Answer:         
            """
}

# Initialize ChromaDB client and collections
persist_directory = "./chroma_db"
client = chromadb.PersistentClient(path=persist_directory)

# Retrieve text and image collections
text_collection = client.get_or_create_collection("text_collection")
image_collection = client.get_or_create_collection("image_collection")

# Initialize ChromaVectorStore for text and images
text_store = ChromaVectorStore(chroma_collection=text_collection)
image_store = ChromaVectorStore(chroma_collection=image_collection)

# Load the existing index with ChromaDB vector stores
try:
    storage_context = StorageContext.from_defaults(
        vector_store=text_store,  # Add text vector store
        image_store=image_store,  # Add image vector store
        persist_dir="./index_storage"  # Persistent storage location
    )
    index = load_index_from_storage(storage_context)
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Failed to load index: {str(e)}")

# Initialize query engines for each template
query_engines = {}
for key, tmpl_str in prompt_templates.items():
    tmpl = PromptTemplate(tmpl_str)
    query_engines[key] = index.as_query_engine(llm=openai_mm_llm, text_qa_template=tmpl)

# Function to generate response using selected prompt template
def generate_response(query_str, selected_template):
    try:
        if selected_template not in query_engines:
            raise ValueError("Invalid template selection")

        # Use the selected query engine
        query_engine = query_engines[selected_template]

        # Use the query engine to process the query
        response = query_engine.query(query_str)

        # Prepare the response data
        response_data = {"response": str(response), "images": []}

        # Check if the response contains image metadata
        if response.metadata and "image_nodes" in response.metadata:
            for img_node in response.metadata["image_nodes"]:
                if "file_path" in img_node.metadata:
                    img_path = img_node.metadata["file_path"]
                    if os.path.exists(img_path):
                        with open(img_path, "rb") as image_file:
                            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
                            response_data["images"].append(encoded_string)
                    else:
                        raise HTTPException(status_code=404, detail=f"Image not found at path: {img_path}")
        return response_data
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating response: {e}")

# Define FastAPI endpoint
@app.post("/query/")
async def handle_query(query: str = Form(...), helper: str = Form("default")):
    if not query:
        raise HTTPException(status_code=400, detail="Please enter a query.")
    
    if helper not in prompt_templates:
        raise HTTPException(status_code=400, detail="Invalid helper selection.")

    # Use the selected helper to generate a response
    result = generate_response(query, helper)
    return JSONResponse(content=result)


@app.get("/")
async def read_index():
    return FileResponse("static/index.html")

# Run the FastAPI app if this script is executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
