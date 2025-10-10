from pymongo import AsyncMongoClient
import asyncio
from config import MONGODB_URI
from loguru import logger

# --------------------------
# Sample Professionals (20)
# --------------------------
professionals = [
    {"name": "Dr. Raj Sharma", "profession": "Cardiologist", "city": "Delhi", "availability": ["Monday", "Wednesday", "Friday"], "working_hours": "10:00-16:00", "certification": "MD Cardiology"},
    {"name": "Dr. Aditi Verma", "profession": "Dermatologist", "city": "Delhi", "availability": ["Tuesday", "Thursday"], "working_hours": "11:00-17:00", "certification": "MD Dermatology"},
    {"name": "Dr. Priya Nair", "profession": "Dentist", "city": "Delhi", "availability": ["Monday", "Tuesday", "Thursday"], "working_hours": "09:00-15:00", "certification": "BDS"},
    {"name": "Dr. Rahul Gupta", "profession": "Neurologist", "city": "Delhi", "availability": ["Wednesday", "Friday"], "working_hours": "10:00-16:00", "certification": "DM Neurology"},
    {"name": "Dr. Sneha Bose", "profession": "Cardiologist", "city": "Kolkata", "availability": ["Monday", "Thursday"], "working_hours": "10:00-15:00", "certification": "MD Cardiology"},
    {"name": "Dr. Amit Sen", "profession": "Dermatologist", "city": "Kolkata", "availability": ["Tuesday", "Thursday"], "working_hours": "11:00-17:00", "certification": "MD Dermatology"},
    {"name": "Dr. Rina Das", "profession": "Dentist", "city": "Kolkata", "availability": ["Monday", "Wednesday"], "working_hours": "09:00-14:00", "certification": "BDS"},
    {"name": "Dr. Rohit Kapoor", "profession": "Neurologist", "city": "Pune", "availability": ["Wednesday", "Friday"], "working_hours": "10:00-16:00", "certification": "DM Neurology"},
    {"name": "Dr. Meera Joshi", "profession": "Cardiologist", "city": "Pune", "availability": ["Monday", "Friday"], "working_hours": "10:00-16:00", "certification": "MD Cardiology"},
    {"name": "Dr. Suresh Patil", "profession": "Dermatologist", "city": "Pune", "availability": ["Tuesday", "Thursday"], "working_hours": "11:00-17:00", "certification": "MD Dermatology"},
    {"name": "Dr. Anjali Mehta", "profession": "Dermatologist", "city": "Bangalore", "availability": ["Tuesday", "Thursday", "Friday"], "working_hours": "11:00-17:00", "certification": "MD Dermatology"},
    {"name": "Dr. Sameer Jain", "profession": "Cardiologist", "city": "Bangalore", "availability": ["Monday", "Wednesday"], "working_hours": "10:00-16:00", "certification": "MD Cardiology"},
    {"name": "Dr. Priya Reddy", "profession": "Dentist", "city": "Bangalore", "availability": ["Monday", "Thursday"], "working_hours": "09:00-15:00", "certification": "MDS"},
    {"name": "Dr. Arjun Kumar", "profession": "Neurologist", "city": "Bangalore", "availability": ["Wednesday", "Friday"], "working_hours": "10:00-16:00", "certification": "DM Neurology"},
    {"name": "Dr. Kavita Sharma", "profession": "Dermatologist", "city": "Delhi", "availability": ["Monday", "Thursday"], "working_hours": "11:00-17:00", "certification": "MD Dermatology"},
    {"name": "Dr. Nikhil Verma", "profession": "Cardiologist", "city": "Kolkata", "availability": ["Tuesday", "Friday"], "working_hours": "10:00-16:00", "certification": "MD Cardiology"},
    {"name": "Dr. Shreya Gupta", "profession": "Dentist", "city": "Pune", "availability": ["Monday", "Wednesday"], "working_hours": "09:00-14:00", "certification": "MDS"},
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
    logger.info("✅ 20 sample professionals inserted successfully.")

if __name__ == "__main__":
    asyncio.run(populate())
