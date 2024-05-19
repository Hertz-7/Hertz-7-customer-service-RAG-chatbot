import sqlite3
import aiosqlite
import random
import string

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
conn.commit()
conn.close()

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
