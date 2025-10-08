from pymongo import AsyncMongoClient
import asyncio
import os
from dotenv import load_dotenv

# --------------------------
# Load environment
# --------------------------
load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI")

# --------------------------
# Sample Professionals (20)
# --------------------------
professionals = [
    {"name": "Dr. Raj Sharma", "profession": "Cardiologist", "city": "Delhi", "availability": ["Monday", "Wednesday", "Friday"], "working_hours": "10:00-16:00", "certification": "MD Cardiology"},
    {"name": "Dr. Aditi Verma", "profession": "Dermatologist", "city": "Delhi", "availability": ["Tuesday", "Thursday"], "working_hours": "11:00-17:00", "certification": "MD Dermatology"},
    {"name": "Dr. Priya Nair", "profession": "Pediatrician", "city": "Delhi", "availability": ["Monday", "Tuesday", "Thursday"], "working_hours": "09:00-15:00", "certification": "MD Pediatrics"},
    {"name": "Dr. Rahul Gupta", "profession": "Neurologist", "city": "Delhi", "availability": ["Wednesday", "Friday"], "working_hours": "10:00-16:00", "certification": "DM Neurology"},
    {"name": "Dr. Sneha Bose", "profession": "Cardiologist", "city": "Kolkata", "availability": ["Monday", "Thursday"], "working_hours": "10:00-15:00", "certification": "MD Cardiology"},
    {"name": "Dr. Amit Sen", "profession": "Dermatologist", "city": "Kolkata", "availability": ["Tuesday", "Thursday"], "working_hours": "11:00-17:00", "certification": "MD Dermatology"},
    {"name": "Dr. Rina Das", "profession": "Pediatrician", "city": "Kolkata", "availability": ["Monday", "Wednesday"], "working_hours": "09:00-14:00", "certification": "MD Pediatrics"},
    {"name": "Dr. Rohit Kapoor", "profession": "Neurologist", "city": "Pune", "availability": ["Wednesday", "Friday"], "working_hours": "10:00-16:00", "certification": "DM Neurology"},
    {"name": "Dr. Meera Joshi", "profession": "Cardiologist", "city": "Pune", "availability": ["Monday", "Friday"], "working_hours": "10:00-16:00", "certification": "MD Cardiology"},
    {"name": "Dr. Suresh Patil", "profession": "Dermatologist", "city": "Pune", "availability": ["Tuesday", "Thursday"], "working_hours": "11:00-17:00", "certification": "MD Dermatology"},
    {"name": "Dr. Anjali Mehta", "profession": "Dermatologist", "city": "Bangalore", "availability": ["Tuesday", "Thursday", "Friday"], "working_hours": "11:00-17:00", "certification": "MD Dermatology"},
    {"name": "Dr. Sameer Jain", "profession": "Cardiologist", "city": "Bangalore", "availability": ["Monday", "Wednesday"], "working_hours": "10:00-16:00", "certification": "MD Cardiology"},
    {"name": "Dr. Priya Reddy", "profession": "Pediatrician", "city": "Bangalore", "availability": ["Monday", "Thursday"], "working_hours": "09:00-15:00", "certification": "MD Pediatrics"},
    {"name": "Dr. Arjun Kumar", "profession": "Neurologist", "city": "Bangalore", "availability": ["Wednesday", "Friday"], "working_hours": "10:00-16:00", "certification": "DM Neurology"},
    {"name": "Dr. Kavita Sharma", "profession": "Dermatologist", "city": "Delhi", "availability": ["Monday", "Thursday"], "working_hours": "11:00-17:00", "certification": "MD Dermatology"},
    {"name": "Dr. Nikhil Verma", "profession": "Cardiologist", "city": "Kolkata", "availability": ["Tuesday", "Friday"], "working_hours": "10:00-16:00", "certification": "MD Cardiology"},
    {"name": "Dr. Shreya Gupta", "profession": "Pediatrician", "city": "Pune", "availability": ["Monday", "Wednesday"], "working_hours": "09:00-14:00", "certification": "MD Pediatrics"},
    {"name": "Dr. Varun Mehta", "profession": "Neurologist", "city": "Bangalore", "availability": ["Tuesday", "Thursday"], "working_hours": "10:00-16:00", "certification": "DM Neurology"},
    {"name": "Dr. Ananya Roy", "profession": "Dermatologist", "city": "Kolkata", "availability": ["Wednesday", "Friday"], "working_hours": "11:00-17:00", "certification": "MD Dermatology"},
    {"name": "Dr. Rohan Das", "profession": "Cardiologist", "city": "Delhi", "availability": ["Tuesday", "Thursday"], "working_hours": "10:00-16:00", "certification": "MD Cardiology"},
]
# --------------------------
# Async MongoDB Insertion
# --------------------------
async def populate():
    client = AsyncMongoClient(MONGODB_URI)
    db = client["healthcare_app"]
    collection = db["professionals"]

    # Clear existing collection
    await collection.delete_many({})

    # Insert sample professionals
    await collection.insert_many(professionals)
    print("✅ 20 sample professionals inserted successfully.")

if __name__ == "__main__":
    asyncio.run(populate())
