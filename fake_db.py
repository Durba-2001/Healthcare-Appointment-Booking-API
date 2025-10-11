from pymongo import ASCENDING
from pymongo import AsyncMongoClient  
import asyncio
import random
from loguru import logger
from config import MONGODB_URI

# --------------------------
# Sample Professionals 
# --------------------------
professionals = [
    {"name": "Dr. Raj Sharma", "type": "Cardiologist", "city": "Delhi", "working_days": ["Monday", "Wednesday", "Friday"], "working_hours": "10:00-16:00", "certification": "MD Cardiology"},
    {"name": "Dr. Aditi Verma", "type": "Dermatologist", "city": "Delhi", "working_days": ["Tuesday", "Thursday"], "working_hours": "11:00-17:00", "certification": "MD Dermatology"},
    {"name": "Dr. Priya Nair", "type": "Dentist", "city": "Delhi", "working_days": ["Monday", "Tuesday", "Thursday"], "working_hours": "09:00-15:00", "certification": "BDS"},
    {"name": "Dr. Rahul Gupta", "type": "Neurologist", "city": "Delhi", "working_days": ["Wednesday", "Friday"], "working_hours": "10:00-16:00", "certification": "DM Neurology"},
    {"name": "Dr. Rohan Das", "type": "Cardiologist", "city": "Delhi", "working_days": ["Tuesday", "Thursday"], "working_hours": "10:00-16:00", "certification": "MD Cardiology"},
    {"name": "Dr. Sneha Bose", "type": "Cardiologist", "city": "Kolkata", "working_days": ["Monday", "Thursday"], "working_hours": "10:00-15:00", "certification": "MD Cardiology"},
    {"name": "Dr. Amit Sen", "type": "Dermatologist", "city": "Kolkata", "working_days": ["Tuesday", "Thursday"], "working_hours": "11:00-17:00", "certification": "MD Dermatology"},
    {"name": "Dr. Rina Das", "type": "Dentist", "city": "Kolkata", "working_days": ["Monday", "Wednesday"], "working_hours": "09:00-14:00", "certification": "BDS"},
    {"name": "Dr. Nikhil Verma", "type": "Cardiologist", "city": "Kolkata", "working_days": ["Tuesday", "Friday"], "working_hours": "10:00-16:00", "certification": "MD Cardiology"},
    {"name": "Dr. Ananya Roy", "type": "Dermatologist", "city": "Kolkata", "working_days": ["Wednesday", "Friday"], "working_hours": "11:00-17:00", "certification": "MD Dermatology"},
    {"name": "Dr. Rohit Kapoor", "type": "Neurologist", "city": "Pune", "working_days": ["Wednesday", "Friday"], "working_hours": "10:00-16:00", "certification": "DM Neurology"},
    {"name": "Dr. Meera Joshi", "type": "Cardiologist", "city": "Pune", "working_days": ["Monday", "Friday"], "working_hours": "10:00-16:00", "certification": "MD Cardiology"},
    {"name": "Dr. Suresh Patil", "type": "Dermatologist", "city": "Pune", "working_days": ["Tuesday", "Thursday"], "working_hours": "11:00-17:00", "certification": "MD Dermatology"},
    {"name": "Dr. Shreya Gupta", "type": "Dentist", "city": "Pune", "working_days": ["Monday", "Wednesday"], "working_hours": "09:00-14:00", "certification": "MDS"},
    {"name": "Dr. Vinay Kulkarni", "type": "Neurologist", "city": "Pune", "working_days": ["Monday", "Thursday"], "working_hours": "10:00-16:00", "certification": "DM Neurology"},
    {"name": "Dr. Anjali Mehta", "type": "Dermatologist", "city": "Bangalore", "working_days": ["Tuesday", "Thursday", "Friday"], "working_hours": "11:00-17:00", "certification": "MD Dermatology"},
    {"name": "Dr. Sameer Jain", "type": "Cardiologist", "city": "Bangalore", "working_days": ["Monday", "Wednesday"], "working_hours": "10:00-16:00", "certification": "MD Cardiology"},
    {"name": "Dr. Priya Reddy", "type": "Dentist", "city": "Bangalore", "working_days": ["Monday", "Thursday"], "working_hours": "09:00-15:00", "certification": "MDS"},
    {"name": "Dr. Arjun Kumar", "type": "Neurologist", "city": "Bangalore", "working_days": ["Wednesday", "Friday"], "working_hours": "10:00-16:00", "certification": "DM Neurology"},
    {"name": "Dr. Varun Mehta", "type": "Neurologist", "city": "Bangalore", "working_days": ["Tuesday", "Thursday"], "working_hours": "10:00-16:00", "certification": "DM Neurology"},
    {"name": "Dr. Kavita Sharma", "type": "Dermatologist", "city": "Delhi", "working_days": ["Monday", "Thursday"], "working_hours": "11:00-17:00", "certification": "MD Dermatology"},
    {"name": "Dr. Aakash Singh", "type": "Orthopedic", "city": "Delhi", "working_days": ["Tuesday", "Friday"], "working_hours": "10:00-15:00", "certification": "MS Orthopedics"},
    {"name": "Dr. Pooja Desai", "type": "Pediatrician", "city": "Kolkata", "working_days": ["Monday", "Wednesday", "Friday"], "working_hours": "09:00-14:00", "certification": "MD Pediatrics"},
    {"name": "Dr. Nivedita Rao", "type": "Gynecologist", "city": "Bangalore", "working_days": ["Tuesday", "Thursday"], "working_hours": "10:00-16:00", "certification": "MD Gynecology"},
    {"name": "Dr. Ritesh Naik", "type": "Cardiologist", "city": "Pune", "working_days": ["Wednesday", "Friday"], "working_hours": "10:00-16:00", "certification": "MD Cardiology"},
]

# Add random experience and rating
for p in professionals:
    p["years_experience"] = random.randint(3, 25)
    p["rating"] = round(random.uniform(3.5, 5.0), 1)

# --------------------------
# Async MongoDB Insertion
# --------------------------
async def populate():
    client = AsyncMongoClient(MONGODB_URI)
    db = client["healthcare_app"]
    collection = db["professionals"]

    # Clear old data
    await collection.delete_many({})
    logger.info("🧹 Cleared existing professionals collection.")

    # Insert new sample data
    await collection.insert_many(professionals)
    logger.info("✅ Inserted 25 sample professionals successfully.")

    # Create indexes
    await collection.create_index([("name", ASCENDING)])
    await collection.create_index([("city", ASCENDING)])
    await collection.create_index([("type", ASCENDING)])
    logger.info("📚 Created indexes on 'name', 'city', and 'type'.")

    client.close()

# --------------------------
# Run the population script
# --------------------------
if __name__ == "__main__":
    asyncio.run(populate())
