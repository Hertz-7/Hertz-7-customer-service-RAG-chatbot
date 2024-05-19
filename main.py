# Import necessary libraries
import sqlite3
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import nest_asyncio
from pyngrok import ngrok
import uvicorn
from transformers import BartForConditionalGeneration, BartTokenizer
import logging
import time
import torch
import aiosqlite
import random
import string

# Authenticate ngrok
ngrok.set_auth_token("2gRbg0PBlT2QOhIlZPx3vn91aTq_2BToa62C6K3s1oTqJapPy")  # Replace with your actual ngrok authtoken

# Allow asyncio to work in Jupyter notebook environment
nest_asyncio.apply()

# Set up logging
logging.basicConfig(filename='bot_interactions.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Load your dataset from a CSV file
csv_file_path = '/content/NikeProductDescriptions.csv'  # Update the path as necessary
df = pd.read_csv(csv_file_path)

# Concatenate the Title, Subtitle, and Product Description
documents = (df['Title'] + ' ' + df['Subtitle'] + ' ' + df['Product Description']).tolist()
logger.info(f"Documents: {documents[:5]}")  # Log the first 5 documents for verification

# Initialize the sentence transformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Create embeddings for the concatenated text
embeddings = model.encode(documents)
logger.info(f"Embeddings shape: {embeddings.shape}")

# Initialize ChromaDB client with default settings
client = chromadb.Client()

# Create a new collection
collection = client.create_collection(name="products")

# Add embeddings to the collection using the correct method
collection.upsert(
    ids=[str(idx) for idx in range(len(documents))],
    embeddings=[embedding.tolist() for embedding in embeddings],
    metadatas=[{"document": doc} for doc in documents]
)

# Load BART model and tokenizer
bart_model_name = 'facebook/bart-large-cnn'
tokenizer = BartTokenizer.from_pretrained(bart_model_name)
bart_model = BartForConditionalGeneration.from_pretrained(bart_model_name).to(device)

# Set up SQLite database for customer records
conn = sqlite3.connect('customer_db.sqlite')
cursor = conn.cursor()

# Create a table for customer records
cursor.execute('''
CREATE TABLE IF NOT EXISTS customers (
    id INTEGER PRIMARY KEY,
    name TEXT,
    email TEXT,
    account_number TEXT UNIQUE,
    purchase_history TEXT
)
''')

# Function to generate a unique account number
def generate_account_number():
    return ''.join(random.choices(string.digits, k=10))

# Function to fetch a customer record asynchronously
async def fetch_customer_record(customer_id):
    async with aiosqlite.connect('customer_db.sqlite') as db:
        cursor = await db.execute('SELECT * FROM customers WHERE id = ?', (customer_id,))
        customer = await cursor.fetchone()
        await cursor.close()
    return customer

# Function to update a customer record asynchronously
async def update_customer_record(customer_id, name, email):
    async with aiosqlite.connect('customer_db.sqlite') as db:
        await db.execute('''
        UPDATE customers
        SET name = ?, email = ?
        WHERE id = ?
        ''', (name, email, customer_id))
        await db.commit()

# Function to register a new customer
async def register_customer(name, email):
    account_number = generate_account_number()
    async with aiosqlite.connect('customer_db.sqlite') as db:
        cursor = await db.execute('''
        INSERT INTO customers (name, email, account_number)
        VALUES (?, ?, ?)
        ''', (name, email, account_number))
        await db.commit()
    return account_number

# Function to authenticate a customer
async def authenticate_customer(name, account_number):
    async with aiosqlite.connect('customer_db.sqlite') as db:
        cursor = await db.execute('SELECT * FROM customers WHERE name = ? AND account_number = ?', (name, account_number))
        customer = await cursor.fetchone()
        await cursor.close()
    return customer

# Create FastAPI app
app = FastAPI()

# class Query(BaseModel):
#     question: str

class Query(BaseModel):
    name: str
    account_number: str
    question: str

class UpdateCustomer(BaseModel):
    name: str
    account_number: str
    customer_id: int
    new_name: str = None
    new_email: str = None
    purchase_history: str = None

class RegisterCustomer(BaseModel):
    name: str
    email: str

@app.post("/register")
async def register(customer: RegisterCustomer):
    try:
        account_number = await register_customer(customer.name, customer.email)
        return {"detail": "Customer registered successfully", "account_number": account_number}
    except Exception as e:
        logger.error("Error occurred during registration: %s", e)
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/chat")
async def chat(query: Query):
    try:
        logger.info("Received question: %s", query.question)
        # Authenticate the user
        customer = await authenticate_customer(query.name, query.account_number)
        if not customer:
            logger.warning("Authentication failed for user: %s with account number: %s", query.name, query.account_number)
            raise HTTPException(status_code=401, detail="Authentication failed")

        # Create embedding for the user question
        question_embedding = model.encode(query.question)
        logger.info(f"Question embedding shape: {question_embedding.shape}")

        # Query the ChromaDB for the most relevant document
        results = collection.query(
            query_embeddings=[question_embedding.tolist()],
            n_results=1
        )
        logger.info(f"Query results: {results}")

        # Get the most relevant document
        if not results["metadatas"]:
            logger.warning("No relevant documents found for question: %s", query.question)
            raise HTTPException(status_code=404, detail="No relevant documents found")

        context = results["documents"][0]
        logger.info("Retrieved context: %s", context)

        # Use BART to generate a response
        input_text = f"Question: {query.question}\nContext: {context}\nAnswer:"
        input_ids = tokenizer.encode(input_text, return_tensors='pt')
        output = bart_model.generate(
            input_ids,
            max_length=150,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        generated_response = tokenizer.decode(output[0], skip_special_tokens=True)

        logger.info("Generated response: %s", generated_response)
        # Log the interaction and the component accessed
        log_interaction(query.question, generated_response, "RAG")
        return {"response": generated_response}
    except Exception as e:
        logger.error("Error occurred during chat: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/customer/{customer_id}")
async def get_customer(customer_id: int, name: str, account_number: str):
    try:
        # Authenticate the user
        customer = await authenticate_customer(name, account_number)
        if not customer:
            logger.warning("Authentication failed for user: %s with account number: %s", name, account_number)
            raise HTTPException(status_code=401, detail="Authentication failed")

        customer_record = await fetch_customer_record(customer_id)
        if not customer_record:
            logger.warning("Customer not found with ID: %d", customer_id)
            raise HTTPException(status_code=404, detail="Customer not found")
        response = {"id": customer_record[0], "name": customer_record[1], "email": customer_record[2], "purchase_history": customer_record[4]}
        # Log the interaction and the component accessed
        log_interaction(f"Fetch customer {customer_id}", response, "Database")
        return response
    except Exception as e:
        logger.error("Error occurred while fetching customer: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/customer")
async def update_customer(update: UpdateCustomer):
    try:
        # Authenticate the user
        customer = await authenticate_customer(update.name, update.account_number)
        if not customer:
            logger.warning("Authentication failed for user: %s with account number: %s", update.name, update.account_number)
            raise HTTPException(status_code=401, detail="Authentication failed")

        await update_customer_record(update.customer_id, update.new_name, update.new_email)
        response = {"detail": "Customer record updated"}
        # Log the interaction and the component accessed
        log_interaction(f"Update customer {update.customer_id}", response, "Database")
        return response
    except Exception as e:
        logger.error("Error occurred while updating customer: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

# Function to log interactions
def log_interaction(user_input, response, component_accessed):
    logger.info(f"User Input: {user_input}, Response: {response}, Component Accessed: {component_accessed}")

# Start ngrok and FastAPI server
public_url = ngrok.connect(8000)
print(f"Public URL: {public_url}")

# Run the FastAPI app
def run():
    time.sleep(2)
    uvicorn.run(app, host="0.0.0.0", port=8000)

run()