"""
Artisan Backend - SIH Demo
---------------------------
This is the ENTIRE backend for the demo. It's intentionally simple:
- No login/auth (single demo artisan, no time for that)
- No real database (a Python dictionary acts as our "database" — resets when server restarts)
- AI calls are wrapped in one function (call_ai) so you only configure your API key ONCE

Run this with:  uvicorn main:app --reload
Then open:      http://127.0.0.1:8000/docs   <- test everything here first, before Flutter connects
"""

from fastapi import FastAPI, UploadFile, HTTPException
import os
import uuid
import json
from dotenv import load_dotenv
from google import genai

# ---- SETUP ----------------------------------------------------------------

load_dotenv(override=True)   # reads GEMINI_API_KEY from your .env file
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI(title="Artisan Backend")

os.makedirs("uploads", exist_ok=True)

# This is our "database" for the demo. Key = product_id, Value = all data about that product.
# Example after upload: {"abc123": {"filename": "vase.jpg", "path": "uploads/vase.jpg"}}
products = {}


# ---- AI HELPER --------------------------------------------------------------
# Every endpoint that needs AI calls THIS one function. Swap the inside of this
# function for OpenAI / Gemini / Claude — the rest of your code never changes.
#
# For now it returns FAKE data so you can build and test your whole backend
# and Flutter app WITHOUT needing an API key or internet call. Once everything
# works end-to-end, replace the inside of this function with a real API call.

def call_ai(prompt: str, image_path: str = None, json_shape: str = None) -> dict:
    """
    Real Gemini call using the new google-genai SDK.
    If image_path is given, Gemini looks at the actual photo.
    json_shape lets each endpoint ask for a different JSON structure back
    (e.g. a listing needs title/description/price, a translation just needs translated text).
    If json_shape isn't given, defaults to the full listing+pricing shape.
    """
    if json_shape is None:
        json_shape = """{
        "title": "...",
        "description": "...",
        "seo_tags": ["...", "...", "..."],
        "category": "...",
        "min_price": 0,
        "max_price": 0,
        "price_currency": "INR",
        "price_note": "one short sentence on why this price range"
    }"""

    full_prompt = f"""{prompt}

    Reply with ONLY valid JSON, no other text, no markdown formatting, in this exact shape:
    {json_shape}
    """

    if image_path:
        # Upload the image file to Gemini, then reference it alongside the text prompt
        uploaded_file = gemini_client.files.upload(file=image_path)
        contents = [full_prompt, uploaded_file]
    else:
        contents = full_prompt

    response = gemini_client.models.generate_content(
        model="gemini-3.6-flash",
        contents=contents
    )

    raw_text = response.text.strip()
    # Gemini sometimes wraps JSON in ```json ... ``` — strip that off if present
    raw_text = raw_text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        # If Gemini didn't return clean JSON, fail loudly so you can see the raw text
        raise HTTPException(status_code=500, detail=f"AI response wasn't valid JSON: {raw_text}")


# ---- ENDPOINT 1: Upload a photo --------------------------------------------

@app.post("/upload")
def upload_photo(file: UploadFile):
    """Artisan sends a photo. We save it and hand back a product_id to use in every other call."""
    product_id = str(uuid.uuid4())[:8]   # short random ID, e.g. "a1b2c3d4"
    save_path = f"uploads/{product_id}_{file.filename}"

    contents = file.file.read()
    with open(save_path, "wb") as f:
        f.write(contents)

    products[product_id] = {
        "filename": file.filename,
        "path": save_path,
        "listing": None,   # will be filled in by /generate-listing
        "price": None
    }

    return {"product_id": product_id, "message": "Photo received"}


# ---- ENDPOINT 2: Generate the AI listing -----------------------------------

@app.post("/generate-listing/{product_id}")
def generate_listing(product_id: str):
    """Send the saved photo to the AI, get back title/description/SEO tags/category/price."""
    if product_id not in products:
        raise HTTPException(status_code=404, detail="Product not found")

    image_path = products[product_id]["path"]
    ai_result = call_ai(
        prompt="Describe this handicraft product for an e-commerce listing, and suggest a fair price range in Indian Rupees (INR) for this as a handmade product sold to Indian B2C customers.",
        image_path=image_path
    )

    products[product_id]["listing"] = ai_result
    products[product_id]["price"] = {
        "min_price": ai_result.get("min_price"),
        "max_price": ai_result.get("max_price"),
        "currency": ai_result.get("price_currency", "INR"),
        "note": ai_result.get("price_note")
    }
    return ai_result


# ---- ENDPOINT 3: Fetch an already-generated listing ------------------------

@app.get("/listing/{product_id}")
def get_listing(product_id: str):
    """Re-fetch a listing already generated, without calling the AI again."""
    if product_id not in products:
        raise HTTPException(status_code=404, detail="Product not found")
    if products[product_id]["listing"] is None:
        raise HTTPException(status_code=400, detail="Listing not generated yet — call /generate-listing first")
    return products[product_id]["listing"]


# ---- ENDPOINT 4: Translate the listing -------------------------------------

@app.post("/translate/{product_id}")
def translate_listing(product_id: str, lang: str = "hi"):
    """Translate the existing description into another language (default Hindi)."""
    if product_id not in products or products[product_id]["listing"] is None:
        raise HTTPException(status_code=400, detail="Generate the listing first")

    original_text = products[product_id]["listing"]["description"]
    prompt = f"Translate this product description to language code '{lang}': {original_text}"
    translation_shape = '{"translated_description": "..."}'
    ai_result = call_ai(prompt=prompt, json_shape=translation_shape)

    return {"language": lang, "translated_description": ai_result["translated_description"]}


# ---- ENDPOINT 5: Pricing suggestion ----------------------------------------

@app.get("/pricing/{product_id}")
def get_pricing(product_id: str):
    """Return the AI-suggested price range. Generated together with the listing in /generate-listing."""
    if product_id not in products:
        raise HTTPException(status_code=404, detail="Product not found")
    if products[product_id]["price"] is None:
        raise HTTPException(status_code=400, detail="No price yet — call /generate-listing first")

    return products[product_id]["price"]


# ---- ENDPOINT 6: Business insights (mocked for demo) -----------------------

@app.get("/insights")
def get_insights():
    """Demand/growth insights. For the demo, mostly mocked — but a couple of stats are real (computed from our 'products' dict)."""
    total_products = len(products)
    return {
        "total_listings_created": total_products,   # this one is real, computed live
        "top_demand_category": "Home Decor / Pottery",
        "growth_tip": "Products with blue pottery tags are trending 20% higher this month",
        "recommended_focus": "Consider creating 2-3 more vase variations"
    }


# ---- ENDPOINT 7: Voice query (text-based, Flutter does the speech-to-text) --

@app.post("/voice-query")
def voice_query(question: str):
    """Flutter converts voice to text on-device and just sends us plain text."""
    answer_shape = '{"answer": "..."}'
    ai_result = call_ai(prompt=question, json_shape=answer_shape)
    return {"answer": ai_result["answer"]}


# ---- ENDPOINT 0: Health check -----------------------------------------------

@app.get("/")
def home():
    return {"message": "Artisan backend is alive"}