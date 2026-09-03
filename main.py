"""
Artisan Backend - SIH Demo
---------------------------
This is the ENTIRE backend for the demo. It's intentionally simple:
- No login/auth (single demo artisan, no time for that)
- No real database (a Python dictionary acts as our "database" — resets when server restarts)
- AI calls are wrapped in a couple of helper functions so you only configure your API key ONCE

Run this with:  uvicorn main:app --reload
Then open:      http://127.0.0.1:8000/docs   <- test everything here first, before Flutter connects
"""

from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import uuid
import json
import time
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from PIL import Image, ImageEnhance

# ---- SETUP ------------------------------------------------------------------

load_dotenv(override=True)   # reads GEMINI_API_KEY from your .env file (forces it to win over any system-level variable)
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI(title="Artisan Backend")

# Allows your Flutter app (running on a phone/emulator, a different "origin"
# than this backend) to actually call these endpoints. Without this, browsers
# and some HTTP clients block the request as a security measure.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # for the demo: allow any origin. Tighten later if needed.
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("uploads", exist_ok=True)

# Makes everything in uploads/ accessible via URL, e.g. /uploads/abc123_enhanced.jpg
# This is how Flutter will actually display uploaded/enhanced images.
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# This is our "database" for the demo. Key = product_id, Value = all data about that product.
# Example after upload: {"abc123": {"filename": "vase.jpg", "path": "uploads/vase.jpg", ...}}
products = {}


# ---- AI HELPER (text/JSON) ---------------------------------------------------
# Every endpoint that needs a text/JSON answer from AI calls THIS one function.
# It uploads the image (if given), sends the prompt to Gemini, and parses the
# JSON reply into a Python dict. Retries automatically if Gemini is temporarily
# overloaded (503), since that's common on the free tier.

def call_ai(prompt: str, image_path: str = None, json_shape: str = None) -> dict:
    """
    Real Gemini call using the google-genai SDK.
    If image_path is given, Gemini looks at the actual photo.
    json_shape lets each endpoint ask for a different JSON structure back
    (e.g. a listing needs title/description/price, a translation just needs
    translated text). If json_shape isn't given, defaults to the full
    listing+pricing shape.
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

    # Gemini's free tier occasionally returns 503 "high demand" errors that
    # usually clear up within a few seconds. Retry a few times before giving
    # up, instead of failing the whole request on the first hiccup.
    max_attempts = 4
    response = None
    for attempt in range(max_attempts):
        try:
            response = gemini_client.models.generate_content(
                model="gemini-3.6-flash",
                contents=contents
            )
            break   # success — stop retrying
        except genai_errors.ServerError:
            if attempt < max_attempts - 1:
                time.sleep(2 * (attempt + 1))   # wait a bit longer each retry: 2s, 4s, 6s
            continue

    if response is None:
        # All retries failed — tell the caller clearly instead of a raw crash
        raise HTTPException(
            status_code=503,
            detail="AI service is temporarily overloaded after several retries. Please try again shortly."
        )

    raw_text = response.text.strip()
    # Gemini sometimes wraps JSON in ```json ... ``` — strip that off if present
    raw_text = raw_text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        # If Gemini didn't return clean JSON, fail loudly so you can see the raw text
        raise HTTPException(status_code=500, detail=f"AI response wasn't valid JSON: {raw_text}")


# ---- AI HELPER (image editing) -----------------------------------------------
# Separate from call_ai() above because this one returns raw image bytes back
# from Gemini's image model, not JSON text. Same retry logic, same API key —
# just a different Gemini model that's actually capable of generating/editing
# images (gemini-3.6-flash, used above, is text-only).

def enhance_image_ai(image_path: str, instruction: str) -> bytes:
    """Sends the photo to Gemini's image model, returns the enhanced image as raw bytes. Retries on 503."""
    uploaded_file = gemini_client.files.upload(file=image_path)

    max_attempts = 4
    response = None
    for attempt in range(max_attempts):
        try:
            response = gemini_client.models.generate_content(
                model="gemini-2.5-flash-image",
                contents=[instruction, uploaded_file],
            )
            break
        except genai_errors.ServerError:
            if attempt < max_attempts - 1:
                time.sleep(2 * (attempt + 1))
            continue

    if response is None:
        raise HTTPException(status_code=503, detail="AI image service temporarily overloaded. Please try again shortly.")

    # The image comes back as one of the "parts" in the response — find it
    for part in response.candidates[0].content.parts:
        if part.inline_data:
            return part.inline_data.data

    raise HTTPException(status_code=500, detail="AI did not return an image")


# ---- ENDPOINT 1: Upload a photo ----------------------------------------------

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
        "listing": None,          # will be filled in by /generate-listing
        "price": None,            # will be filled in by /generate-listing
        "enhanced_path": None,    # will be filled in by /enhance-image (Pillow)
        "ai_enhanced_path": None  # will be filled in by /enhance-image-ai (Gemini)
    }

    return {"product_id": product_id, "message": "Photo received"}


# ---- ENDPOINT 2: Generate the AI listing -------------------------------------

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


# ---- ENDPOINT 3: Enhance the product photo (Pillow, no AI call needed) -------

@app.post("/enhance-image/{product_id}")
def enhance_image(product_id: str):
    """
    Auto-enhances the uploaded photo: boosts brightness, contrast, color, and
    sharpness so product photos look more professional. Uses Pillow, not AI —
    fast, free, and doesn't depend on Gemini being up. This is the reliable
    fallback if the AI version below ever fails during a demo.
    """
    if product_id not in products:
        raise HTTPException(status_code=404, detail="Product not found")

    original_path = products[product_id]["path"]
    image = Image.open(original_path).convert("RGB")

    # Each of these nudges the image up a bit — tweak the numbers to taste.
    # 1.0 = unchanged, >1.0 = more of that quality, <1.0 = less.
    image = ImageEnhance.Brightness(image).enhance(1.1)
    image = ImageEnhance.Contrast(image).enhance(1.15)
    image = ImageEnhance.Color(image).enhance(1.2)       # more vivid colors
    image = ImageEnhance.Sharpness(image).enhance(1.3)   # crisper edges

    enhanced_filename = f"{product_id}_enhanced.jpg"
    enhanced_path = f"uploads/{enhanced_filename}"
    image.save(enhanced_path, "JPEG", quality=90)

    products[product_id]["enhanced_path"] = enhanced_path

    return {
        "product_id": product_id,
        "enhanced_image_url": f"/uploads/{enhanced_filename}",
        "message": "Image enhanced"
    }


# ---- ENDPOINT 3b: Enhance the product photo using AI (Gemini image model) ---

@app.post("/enhance-image-ai/{product_id}")
def enhance_image_ai_endpoint(product_id: str):
    """
    AI-enhances the uploaded photo using Gemini's image model — can genuinely
    improve lighting/background, not just adjust sliders like the Pillow
    version above. Slower and depends on Gemini being up, so test this
    thoroughly before relying on it live; /enhance-image (Pillow) is the
    safer fallback if this fails during your demo.
    """
    if product_id not in products:
        raise HTTPException(status_code=404, detail="Product not found")

    original_path = products[product_id]["path"]
    instruction = (
        "Enhance this product photo for an e-commerce listing: improve lighting, "
        "sharpen details, and make the background clean and professional. "
        "Keep the product itself unchanged."
    )
    image_bytes = enhance_image_ai(original_path, instruction)

    enhanced_filename = f"{product_id}_ai_enhanced.jpg"
    enhanced_path = f"uploads/{enhanced_filename}"
    with open(enhanced_path, "wb") as f:
        f.write(image_bytes)

    products[product_id]["ai_enhanced_path"] = enhanced_path

    return {
        "product_id": product_id,
        "ai_enhanced_image_url": f"/uploads/{enhanced_filename}",
        "message": "Image enhanced with AI"
    }


# ---- ENDPOINT 4: Fetch an already-generated listing --------------------------

@app.get("/listing/{product_id}")
def get_listing(product_id: str):
    """Re-fetch a listing already generated, without calling the AI again."""
    if product_id not in products:
        raise HTTPException(status_code=404, detail="Product not found")
    if products[product_id]["listing"] is None:
        raise HTTPException(status_code=400, detail="Listing not generated yet — call /generate-listing first")
    return products[product_id]["listing"]


# ---- ENDPOINT 5: Translate the listing ---------------------------------------

@app.post("/translate/{product_id}")
def translate_listing(product_id: str, lang: str = "hi"):
    """Translate the existing description into another language (default Hindi). Works for any language, not just a fixed list — Gemini handles the translation itself."""
    if product_id not in products or products[product_id]["listing"] is None:
        raise HTTPException(status_code=400, detail="Generate the listing first")

    original_text = products[product_id]["listing"]["description"]
    prompt = f"Translate this product description to language code '{lang}': {original_text}"
    translation_shape = '{"translated_description": "..."}'
    ai_result = call_ai(prompt=prompt, json_shape=translation_shape)

    return {"language": lang, "translated_description": ai_result["translated_description"]}


# ---- ENDPOINT 6: Pricing suggestion ------------------------------------------

@app.get("/pricing/{product_id}")
def get_pricing(product_id: str):
    """Return the AI-suggested price range. Generated together with the listing in /generate-listing, so this is just re-reading saved data — no extra AI call."""
    if product_id not in products:
        raise HTTPException(status_code=404, detail="Product not found")
    if products[product_id]["price"] is None:
        raise HTTPException(status_code=400, detail="No price yet — call /generate-listing first")

    return products[product_id]["price"]


# ---- ENDPOINT 7: Product insights (compares THIS product to the market) -----

@app.get("/product-insights/{product_id}")
def get_product_insights(product_id: str, lang: str = "en"):
    """
    Compares this specific product's photo + listing to typical similar
    handmade products sold online, and gives concrete areas to improve
    (photography, presentation, description, pricing, uniqueness, etc.).
    lang controls the output language directly, e.g. lang=hi for Hindi,
    lang=ta for Tamil — same idea as /translate, just built in here.
    """
    if product_id not in products:
        raise HTTPException(status_code=404, detail="Product not found")
    if products[product_id]["listing"] is None:
        raise HTTPException(status_code=400, detail="Generate the listing first — call /generate-listing")

    listing = products[product_id]["listing"]
    image_path = products[product_id]["path"]

    prompt = f"""Here is a handmade product listing:
    Title: {listing.get('title')}
    Description: {listing.get('description')}
    Category: {listing.get('category')}
    Price range: {listing.get('min_price')}-{listing.get('max_price')} {listing.get('price_currency')}

    Looking at the product photo and this listing, compare it to typical
    successful listings of similar handmade products sold online (e.g. on
    Etsy, Amazon Handmade, or Indian marketplaces for regional crafts).
    Identify 2-3 genuine strengths and 2-3 concrete areas where this specific
    product or its listing could be improved (this could be about the
    photography, presentation, description clarity, pricing, or the craft
    itself). Be specific to what you actually see, not generic advice.

    Write your entire response in the language with code '{lang}'."""

    insight_shape = f"""{{
        "strengths": ["...", "..."],
        "areas_to_improve": ["...", "..."],
        "language": "{lang}"
    }}"""

    ai_result = call_ai(prompt=prompt, image_path=image_path, json_shape=insight_shape)
    return ai_result


# ---- ENDPOINT 8: Voice query (text-based, Flutter does the speech-to-text) --

@app.post("/voice-query")
def voice_query(question: str):
    """Flutter converts voice to text on-device and just sends us plain text — no audio upload needed here."""
    answer_shape = '{"answer": "..."}'
    ai_result = call_ai(prompt=question, json_shape=answer_shape)
    return {"answer": ai_result["answer"]}


# ---- ENDPOINT 0: Health check -------------------------------------------------

@app.get("/")
def home():
    return {"message": "Artisan backend is alive"}