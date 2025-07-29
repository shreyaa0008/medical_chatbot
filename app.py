from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from starlette.middleware.sessions import SessionMiddleware
import base64
import requests
import io
from PIL import Image
from dotenv import load_dotenv
from jose import jwt
import logging
import os
from datetime import datetime, timedelta
from database import users_collection, init_db, close_db
from bson import ObjectId
import json
import asyncio
import aiohttp
from typing import List, Dict, Optional
import math
import random
import aiohttp
import asyncio
from fastapi.responses import StreamingResponse
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


load_dotenv()
SECRET_KEY = os.getenv("SECRET_KEY")

# ------------------- Load Env and Setup ---------------------
load_dotenv()

ACCESS_TOKEN_EXPIRE_MINUTES = 30
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Free API alternatives (add these to your .env file if you want to use them)
NOMINATIM_USER_AGENT = os.getenv("NOMINATIM_USER_AGENT", "medical-app-1.0")  # Required for Nominatim
OVERPASS_API_URL = "https://overpass-api.de/api/interpreter"
NOMINATIM_API_URL = "https://nominatim.openstreetmap.org"

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in the .env file")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)
template = Jinja2Templates(directory="template")

# Expanded mock doctor data with more variety
MOCK_DOCTORS_DATABASE = [
    {
        "name": "Dr. Sarah Johnson",
        "specialty": "Cardiology",
        "location": "Central Heart Hospital",
        "contact": "+1-555-0123",
        "rating": 4.8,
        "address": "123 Medical Center Dr, Downtown",
        "lat": 40.7580,
        "lng": -73.9855,
        "hospital": "Central Heart Hospital",
        "experience": "15 years",
        "education": "MD, Harvard Medical School"
    },
    {
        "name": "Dr. Michael Chen",
        "specialty": "Orthopedics",
        "location": "City Orthopedic Center",
        "contact": "+1-555-0456",
        "rating": 4.6,
        "address": "456 Bone & Joint Ave, Medical District",
        "lat": 40.7614,
        "lng": -73.9776,
        "hospital": "City Orthopedic Center",
        "experience": "12 years",
        "education": "MD, Johns Hopkins University"
    },
    {
        "name": "Dr. Emily Rodriguez",
        "specialty": "Internal Medicine",
        "location": "General Hospital",
        "contact": "+1-555-0789",
        "rating": 4.7,
        "address": "789 Care Ave, Health Center",
        "lat": 40.7505,
        "lng": -73.9934,
        "hospital": "General Hospital",
        "experience": "18 years",
        "education": "MD, Columbia University"
    },
    {
        "name": "Dr. David Park",
        "specialty": "Neurology",
        "location": "Brain & Spine Institute",
        "contact": "+1-555-0321",
        "rating": 4.9,
        "address": "321 Neuro Blvd, Specialist Complex",
        "lat": 40.7282,
        "lng": -73.9942,
        "hospital": "Brain & Spine Institute",
        "experience": "20 years",
        "education": "MD PhD, Stanford University"
    },
    {
        "name": "Dr. Lisa Thompson",
        "specialty": "Radiology",
        "location": "Advanced Imaging Center",
        "contact": "+1-555-0654",
        "rating": 4.5,
        "address": "654 Scan St, Diagnostic Center",
        "lat": 40.7831,
        "lng": -73.9712,
        "hospital": "Advanced Imaging Center",
        "experience": "10 years",
        "education": "MD, Mayo Clinic"
    },
    {
        "name": "Dr. James Wilson",
        "specialty": "Emergency Medicine",
        "location": "City Emergency Hospital",
        "contact": "+1-555-0987",
        "rating": 4.4,
        "address": "987 Emergency Blvd, Hospital District",
        "lat": 40.7589,
        "lng": -73.9851,
        "hospital": "City Emergency Hospital",
        "experience": "8 years",
        "education": "MD, University of Pennsylvania"
    },
    {
        "name": "Dr. Maria Garcia",
        "specialty": "Pediatrics",
        "location": "Children's Medical Center",
        "contact": "+1-555-0246",
        "rating": 4.9,
        "address": "246 Kids Care Dr, Family Health Center",
        "lat": 40.7505,
        "lng": -73.9855,
        "hospital": "Children's Medical Center",
        "experience": "14 years",
        "education": "MD, UCLA Medical School"
    },
    {
        "name": "Dr. Robert Kim",
        "specialty": "Oncology",
        "location": "Cancer Treatment Center",
        "contact": "+1-555-0135",
        "rating": 4.8,
        "address": "135 Hope St, Cancer Care Complex",
        "lat": 40.7614,
        "lng": -73.9934,
        "hospital": "Cancer Treatment Center",
        "experience": "16 years",
        "education": "MD, Memorial Sloan Kettering"
    },
    {
        "name": "Dr. Jennifer Lee",
        "specialty": "Dermatology",
        "location": "Skin Health Clinic",
        "contact": "+1-555-0468",
        "rating": 4.6,
        "address": "468 Skin Care Ave, Dermatology Center",
        "lat": 40.7282,
        "lng": -73.9776,
        "hospital": "Skin Health Clinic",
        "experience": "11 years",
        "education": "MD, NYU School of Medicine"
    },
    {
        "name": "Dr. Thomas Brown",
        "specialty": "Pulmonology",
        "location": "Respiratory Care Center",
        "contact": "+1-555-0579",
        "rating": 4.7,
        "address": "579 Breathing Way, Lung Health Institute",
        "lat": 40.7831,
        "lng": -73.9942,
        "hospital": "Respiratory Care Center",
        "experience": "13 years",
        "education": "MD, Mount Sinai School of Medicine"
    }
]

# Pool of realistic doctor names for generating from facilities
DOCTOR_NAMES_POOL = [
    "Dr. Amanda Williams", "Dr. Michael Johnson", "Dr. Sarah Davis", "Dr. James Brown",
    "Dr. Jennifer Miller", "Dr. David Wilson", "Dr. Lisa Anderson", "Dr. Robert Taylor",
    "Dr. Maria Martinez", "Dr. Christopher Jones", "Dr. Jessica Garcia", "Dr. Matthew Rodriguez",
    "Dr. Ashley Lewis", "Dr. Joshua Lee", "Dr. Amanda Walker", "Dr. Daniel Hall",
    "Dr. Stephanie Allen", "Dr. Ryan Young", "Dr. Nicole Hernandez", "Dr. Kevin King",
    "Dr. Rachel Wright", "Dr. Brandon Lopez", "Dr. Melissa Hill", "Dr. Eric Scott",
    "Dr. Michelle Green", "Dr. Jason Adams", "Dr. Laura Baker", "Dr. Anthony Gonzalez",
    "Dr. Kimberly Nelson", "Dr. Steven Carter", "Dr. Donna Mitchell", "Dr. Mark Perez",
    "Dr. Carol Roberts", "Dr. Paul Turner", "Dr. Sandra Phillips", "Dr. Kenneth Campbell",
    "Dr. Betty Parker", "Dr. Edward Evans", "Dr. Helen Edwards", "Dr. Brian Collins",
    "Dr. Dorothy Stewart", "Dr. Ronald Sanchez", "Dr. Lisa Morris", "Dr. George Rogers"
]

async def get_user_location(request: Request) -> Dict[str, float]:
    """Get user location from session or use default location"""
    user = request.session.get("user", {})
    
    # Try to get location from session
    if "location" in user:
        return user["location"]
    
    # Default location (New York City coordinates)
    return {
        "latitude": 40.7128,
        "longitude": -74.0060
    }

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two coordinates using Haversine formula"""
    R = 6371  # Earth's radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2) * math.sin(dlat/2) + 
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
         math.sin(dlon/2) * math.sin(dlon/2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c
    return distance

def generate_realistic_doctor_name(facility_name: str, specialty: str) -> str:
    """Generate a realistic doctor name based on facility and specialty"""
    # Check if facility name already contains "Dr." 
    if "Dr." in facility_name:
        return facility_name
    
    # Use a hash of facility name + specialty to consistently pick the same name for the same facility
    import hashlib
    hash_input = f"{facility_name}{specialty}".encode()
    hash_value = int(hashlib.md5(hash_input).hexdigest(), 16)
    name_index = hash_value % len(DOCTOR_NAMES_POOL)
    
    return DOCTOR_NAMES_POOL[name_index]

async def search_doctors_overpass_api(latitude: float, longitude: float, radius_km: float = 10) -> List[Dict]:
    """Search for hospitals and clinics using OpenStreetMap Overpass API (Free)"""
    try:
        # Overpass QL query to find hospitals and clinics
        overpass_query = f"""
        [out:json][timeout:25];
        (
          node["amenity"="hospital"](around:{radius_km * 1000},{latitude},{longitude});
          node["amenity"="clinic"](around:{radius_km * 1000},{latitude},{longitude});
          node["amenity"="doctors"](around:{radius_km * 1000},{latitude},{longitude});
          way["amenity"="hospital"](around:{radius_km * 1000},{latitude},{longitude});
          way["amenity"="clinic"](around:{radius_km * 1000},{latitude},{longitude});
          way["amenity"="doctors"](around:{radius_km * 1000},{latitude},{longitude});
        );
        out center meta;
        """
        
        async with aiohttp.ClientSession() as session:
            async with session.post(OVERPASS_API_URL, data=overpass_query) as response:
                if response.status == 200:
                    data = await response.json()
                    doctors = []
                    
                    for element in data.get("elements", [])[:10]:  # Limit to 10 results
                        # Get coordinates
                        if element.get("type") == "node":
                            lat, lng = element.get("lat"), element.get("lon")
                        elif element.get("center"):
                            lat, lng = element["center"]["lat"], element["center"]["lon"]
                        else:
                            continue
                        
                        tags = element.get("tags", {})
                        facility_name = tags.get("name", "Medical Facility")
                        
                        # Skip if no meaningful name available
                        if not facility_name or facility_name == "Medical Facility":
                            continue
                        
                        # Determine specialty based on tags
                        specialty = determine_specialty_from_tags(tags)
                        
                        # Generate realistic doctor name
                        doctor_name = generate_realistic_doctor_name(facility_name, specialty)
                        
                        doctor_info = {
                            "name": doctor_name,
                            "specialty": specialty,
                            "location": facility_name,
                            "contact": tags.get("phone", "Contact facility for details"),
                            "rating": round(random.uniform(4.2, 4.9), 1),  # Random rating for demo
                            "address": format_address(tags),
                            "distance": f"{calculate_distance(latitude, longitude, lat, lng):.1f} km",
                            "hospital": facility_name,
                            "experience": f"{random.randint(5, 25)} years",
                            "website": tags.get("website", "N/A")
                        }
                        doctors.append(doctor_info)
                    
                    return doctors
                else:
                    logger.warning(f"Overpass API returned status {response.status}")
                    return []
    except Exception as e:
        logger.error(f"Error with Overpass API: {str(e)}")
        return []

def determine_specialty_from_tags(tags: Dict) -> str:
    """Determine medical specialty from OpenStreetMap tags"""
    # Check healthcare specialty tag
    healthcare = tags.get("healthcare", "").lower()
    if "cardiology" in healthcare or "heart" in healthcare:
        return "Cardiology"
    elif "orthopedic" in healthcare or "orthopaedic" in healthcare:
        return "Orthopedics"
    elif "neurology" in healthcare or "neuro" in healthcare:
        return "Neurology"
    elif "pediatric" in healthcare or "children" in healthcare:
        return "Pediatrics"
    elif "cancer" in healthcare or "oncology" in healthcare:
        return "Oncology"
    elif "emergency" in healthcare:
        return "Emergency Medicine"
    
    # Check name for specialty indicators
    name = tags.get("name", "").lower()
    if any(word in name for word in ["heart", "cardiac", "cardio"]):
        return "Cardiology"
    elif any(word in name for word in ["bone", "joint", "orthopedic", "orthopaedic"]):
        return "Orthopedics"
    elif any(word in name for word in ["brain", "neuro", "neurological"]):
        return "Neurology"
    elif any(word in name for word in ["children", "pediatric", "paediatric", "kids"]):
        return "Pediatrics"
    elif any(word in name for word in ["cancer", "oncology", "tumor"]):
        return "Oncology"
    elif any(word in name for word in ["emergency", "trauma", "urgent"]):
        return "Emergency Medicine"
    elif any(word in name for word in ["women", "maternity", "obstetric"]):
        return "Obstetrics & Gynecology"
    elif any(word in name for word in ["eye", "vision", "optical"]):
        return "Ophthalmology"
    elif any(word in name for word in ["dental", "tooth", "oral"]):
        return "Dentistry"
    elif any(word in name for word in ["skin", "dermatology"]):
        return "Dermatology"
    
    # Default based on facility type
    amenity = tags.get("amenity", "")
    if amenity == "hospital":
        return "General Medicine"
    elif amenity == "clinic":
        return "General Practice"
    else:
        return "General Practice"

def format_address(tags: Dict) -> str:
    """Format address from OpenStreetMap tags"""
    address_parts = []
    
    # House number and street
    if tags.get("addr:housenumber") and tags.get("addr:street"):
        address_parts.append(f"{tags['addr:housenumber']} {tags['addr:street']}")
    elif tags.get("addr:street"):
        address_parts.append(tags["addr:street"])
    
    # City
    if tags.get("addr:city"):
        address_parts.append(tags["addr:city"])
    
    # State/Region
    if tags.get("addr:state"):
        address_parts.append(tags["addr:state"])
    
    # Postal code
    if tags.get("addr:postcode"):
        address_parts.append(tags["addr:postcode"])
    
    return ", ".join(address_parts) if address_parts else "Address not available"

async def get_location_name(latitude: float, longitude: float) -> str:
    """Get location name using Nominatim (Free reverse geocoding)"""
    try:
        url = f"{NOMINATIM_API_URL}/reverse"
        params = {
            "lat": latitude,
            "lon": longitude,
            "format": "json",
            "addressdetails": 1
        }
        headers = {
            "User-Agent": NOMINATIM_USER_AGENT
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    address = data.get("address", {})
                    
                    # Build location name
                    location_parts = []
                    if address.get("city"):
                        location_parts.append(address["city"])
                    elif address.get("town"):
                        location_parts.append(address["town"])
                    elif address.get("village"):
                        location_parts.append(address["village"])
                    
                    if address.get("state"):
                        location_parts.append(address["state"])
                    
                    return ", ".join(location_parts) if location_parts else "Unknown Location"
    except Exception as e:
        logger.error(f"Error getting location name: {str(e)}")
    
    return "Unknown Location"

async def search_nearby_doctors(latitude: float, longitude: float, specialty: Optional[str] = None) -> List[Dict]:
    """Search for nearby doctors using multiple free APIs and mock data"""
    doctors = []
    
    # Try OpenStreetMap Overpass API first
    try:
        osm_doctors = await search_doctors_overpass_api(latitude, longitude)
        doctors.extend(osm_doctors)
        logger.info(f"Found {len(osm_doctors)} doctors from OpenStreetMap")
    except Exception as e:
        logger.error(f"Error with OpenStreetMap search: {str(e)}")
    
    # Add mock doctors with realistic distances
    mock_doctors = []
    for mock_doc in MOCK_DOCTORS_DATABASE:
        # Calculate distance from user location
        distance = calculate_distance(latitude, longitude, mock_doc["lat"], mock_doc["lng"])
        if distance <= 15:  # Within 15km
            doctor_copy = mock_doc.copy()
            doctor_copy["distance"] = f"{distance:.1f} km"
            mock_doctors.append(doctor_copy)
    
    # Sort mock doctors by distance
    mock_doctors.sort(key=lambda x: float(x["distance"].split()[0]))
    doctors.extend(mock_doctors[:7])  # Add top 7 closest mock doctors
    
    # Filter by specialty if provided
    if specialty:
        doctors = [
            doc for doc in doctors 
            if specialty.lower() in doc["specialty"].lower()
        ]
    
    # Sort by distance and limit results
    doctors.sort(key=lambda x: float(x["distance"].split()[0]))
    
    # Ensure we have at least some doctors
    if len(doctors) < 3:
        # Add some random mock doctors if we don't have enough
        additional_doctors = random.sample(MOCK_DOCTORS_DATABASE, min(5, len(MOCK_DOCTORS_DATABASE)))
        for doc in additional_doctors:
            if doc not in doctors:
                doc_copy = doc.copy()
                doc_copy["distance"] = f"{random.uniform(2.0, 8.0):.1f} km"
                doctors.append(doc_copy)
    
    return doctors[:8]  # Return top 8 doctors

def extract_specialty_from_query(query: str) -> Optional[str]:
    """Extract potential medical specialty from user query"""
    query_lower = query.lower()
    
    specialty_keywords = {
        "heart": "Cardiology",
        "cardiac": "Cardiology",
        "chest": "Cardiology",
        "bone": "Orthopedics",
        "fracture": "Orthopedics",
        "joint": "Orthopedics",
        "brain": "Neurology",
        "neurological": "Neurology",
        "headache": "Neurology",
        "skin": "Dermatology",
        "rash": "Dermatology",
        "eye": "Ophthalmology",
        "vision": "Ophthalmology",
        "dental": "Dentistry",
        "tooth": "Dentistry",
        "lung": "Pulmonology",
        "breathing": "Pulmonology",
        "stomach": "Gastroenterology",
        "digestive": "Gastroenterology",
        "cancer": "Oncology",
        "tumor": "Oncology",
        "child": "Pediatrics",
        "pediatric": "Pediatrics",
        "emergency": "Emergency Medicine",
        "urgent": "Emergency Medicine"
    }
    
    for keyword, specialty in specialty_keywords.items():
        if keyword in query_lower:
            return specialty
    
    return None

# Database startup and shutdown events
@app.on_event("startup")
async def startup_event():
    await init_db()

@app.on_event("shutdown")
async def shutdown_event():
    await close_db()

# ------------------- Routes ----------------------------
@app.get("/debug/users")
async def debug_users():
    users = []
    async for user in users_collection.find():
        user["_id"] = str(user["_id"])
        users.append(user)
    return users

@app.delete("/debug/clear-users")
async def clear_all_users():
    result = await users_collection.delete_many({})
    return {"message": f"Deleted {result.deleted_count} users"}

@app.get("/get-user/{username}")
async def get_user(username: str):
    user = await users_collection.find_one({"username": username})
    if user:
        user["_id"] = str(user["_id"])
        return JSONResponse(content=user)
    return JSONResponse(content={"error": "User not found"}, status_code=404)

@app.get("/", response_class=HTMLResponse)
async def redirect_to_login(request: Request):
    return RedirectResponse("/login")

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    user = request.session.get("user")
    if not user:
        return RedirectResponse("/login")
    return template.TemplateResponse("index.html", {"request": request, "user": user})

@app.get("/login", response_class=HTMLResponse)
async def login_get(request: Request):
    return template.TemplateResponse("login.html", {"request": request})

@app.get("/register", response_class=HTMLResponse)
async def show_register_page(request: Request):
    return template.TemplateResponse("register.html", {"request": request})

@app.post("/register")
async def register_user(username: str = Form(...), email: str = Form(...), password: str = Form(...)):
    existing_user = await users_collection.find_one({"$or": [{"username": username}, {"email": email}]})
    if existing_user:
        if existing_user["username"] == username:
            raise HTTPException(status_code=400, detail="Username already exists")
        else:
            raise HTTPException(status_code=400, detail="Email already registered")

    if len(password) < 8 or not any(c.isalpha() for c in password) or not any(c.isdigit() for c in password):
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters long and contain both letters and numbers")

    if not "@" in email or not "." in email:
        raise HTTPException(status_code=400, detail="Invalid email format")

    new_user = {
        "username": username,
        "email": email,
        "password": password,
        "last_login": None
    }

    try:
        await users_collection.insert_one(new_user)
        return RedirectResponse("/login", status_code=302)
    except Exception as e:
        logger.error(f"Database error during registration: {str(e)}")
        raise HTTPException(status_code=500, detail="Registration failed due to server error")

@app.post("/login")
async def login_post(request: Request):
    try:
        form = await request.form()
        username = form.get("username")
        password = form.get("password")

        # Add debug logging
        logger.info(f"Login attempt - Username: {username}, Password provided: {bool(password)}")

        if not username or not password:
            return template.TemplateResponse("login.html", {
                "request": request,
                "error": "Username and password are required"
            })

        user = await users_collection.find_one({"username": username})
        
        if not user or password != user.get("password"):
            return template.TemplateResponse("login.html", {
                "request": request,
                "error": "Invalid credentials"
            })
        
        await users_collection.update_one(
            {"_id": user["_id"]},
            {"$set": {"last_login": datetime.utcnow()}}
        )
        
        request.session["user"] = {
            "id": str(user["_id"]),
            "username": user["username"],
            "email": user["email"]
        }
        
        return RedirectResponse("/dashboard", status_code=302)
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        return template.TemplateResponse("login.html", {
            "request": request,
            "error": "An error occurred during login"
        })

@app.get("/logout")
async def logout(request: Request):
    request.session.pop("user", None)
    return RedirectResponse("/login")

@app.post("/upload_and_query")
async def upload_and_query(
    request: Request,
    image: UploadFile = File(...),
    query: str = Form(...)
):
    try:
        image_content = await image.read()
        if not image_content:
            raise HTTPException(status_code=400, detail="Empty file")

        encoded_image = base64.b64encode(image_content).decode("utf-8")

        try:
            img = Image.open(io.BytesIO(image_content))
            img.verify()
        except Exception as e:
            logger.error(f"Invalid image format: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

        # Get user location
        location = await get_user_location(request)
        
        # Extract potential specialty from query
        specialty = extract_specialty_from_query(query)
        
        # Search for nearby doctors using free APIs
        doctors = await search_nearby_doctors(
            location["latitude"], 
            location["longitude"], 
            specialty
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                ]
            }
        ]

        def make_api_request(model):
            response = requests.post(
                GROQ_API_URL,
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": 1000
                },
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                timeout=30
            )
            return response

        llama_response = make_api_request("meta-llama/llama-4-scout-17b-16e-instruct")
        llava_response = make_api_request("meta-llama/llama-4-scout-17b-16e-instruct")

        responses = {}
        for model, response in [("llama", llama_response), ("llava", llava_response)]:
            if response.status_code == 200:
                result = response.json()
                answer = result["choices"][0]["message"]["content"]
                logger.info(f"Response from {model} API: {answer[:100]}...")
                responses[model] = answer
            else:
                logger.error(f"Error from {model} API: {response.status_code} - {response.text}")
                responses[model] = f"Error from {model} API: {response.status_code}"

        # Add doctors to the response
        responses["doctors"] = doctors

        return JSONResponse(status_code=200, content=responses)

    except HTTPException as he:
        logger.error(f"HTTP Exception: {str(he)}")
        raise he
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/update_location")
async def update_location(
    request: Request,
    latitude: float = Form(...),
    longitude: float = Form(...)
):
    """Update user's location in session"""
    try:
        user = request.session.get("user")
        if not user:
            raise HTTPException(status_code=401, detail="User not logged in")
        
        # Update location in session
        user["location"] = {"latitude": latitude, "longitude": longitude}
        request.session["user"] = user
        
        # Get location name for display
        location_name = await get_location_name(latitude, longitude)
        
        return {
            "message": "Location updated successfully",
            "location_name": location_name
        }
    except Exception as e:
        logger.error(f"Error updating location: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update location")

# New endpoint to test doctor search
@app.get("/test_doctors/{lat}/{lng}")
async def test_doctors(lat: float, lng: float, specialty: Optional[str] = None):
    """Test endpoint to check doctor search functionality"""
    doctors = await search_nearby_doctors(lat, lng, specialty)
    return {"doctors": doctors, "count": len(doctors)}


@app.post("/download_report")
async def download_report(request: Request):
    data = await request.json()
    query = data.get("query", "N/A")
    llama = data.get("llama", "No response")
    llava = data.get("llava", "No response")
    doctors = data.get("doctors", [])

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    y = height - 40
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, y, "Medical AI Analysis Report")
    y -= 30

    c.setFont("Helvetica", 12)
    c.drawString(40, y, f"User Query: {query}")
    y -= 30

    def draw_wrapped_text(title, text, y, max_width=500, line_height=15):
        c.setFont("Helvetica-Bold", 12)
        c.drawString(40, y, title)
        y -= line_height
        c.setFont("Helvetica", 11)
        for line in text.split('\n'):
            for subline in split_text(line, max_width):
                if y < 40:
                    c.showPage()
                    y = height - 40
                c.drawString(50, y, subline)
                y -= line_height
        return y - 10

    def split_text(text, max_width):
        from reportlab.pdfbase.pdfmetrics import stringWidth
        words = text.split()
        lines = []
        line = ''
        for word in words:
            if stringWidth(line + ' ' + word, 'Helvetica', 11) < max_width:
                line += ' ' + word if line else word
            else:
                lines.append(line)
                line = word
        lines.append(line)
        return lines

    y = draw_wrapped_text("LLaMA 11B Response:", llama, y)
    y = draw_wrapped_text("LLaMA 90B Response:", llava, y)

    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Nearby Doctors:")
    y -= 20
    c.setFont("Helvetica", 11)

    for doctor in doctors:
        text = f"{doctor.get('name')} - {doctor.get('specialty')} | {doctor.get('location')} | {doctor.get('contact')}"
        for line in split_text(text, 500):
            if y < 40:
                c.showPage()
                y = height - 40
            c.drawString(50, y, line)
            y -= 15
        y -= 10

    c.save()
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="application/pdf", headers={"Content-Disposition": "attachment; filename=Medical_Report.pdf"})
# ------------------- Run -------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)