import os
import uuid
import random
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import httpx
import websockets
import time
from PIL import Image
import logging

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

COMFY_SERVER = "http://127.0.0.1:8188"
OUTPUT_DIR = "c:/Users/User/Desktop/app/draw_api/output"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
COMFY_INPUT_DIR = "C:/ComfyUI_windows_portable_nvidia/ComfyUI_windows_portable/ComfyUI/input"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(COMFY_INPUT_DIR, exist_ok=True)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def generate_unique_filename(original_filename):
    """Generate a unique filename."""
    _, file_extension = os.path.splitext(original_filename)
    return f"{uuid.uuid4()}{file_extension}"

@app.post("/upload")
async def upload_image(image: UploadFile = File(...)):
    if not image.filename:
        raise HTTPException(status_code=400, detail="No selected file")

    try:
        unique_filename = generate_unique_filename(image.filename)
        filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
        
        with open(filepath, "wb") as buffer:
            content = await image.read()
            buffer.write(content)
        
        return JSONResponse(content={"message": "Image uploaded successfully", "filename": unique_filename})
    except Exception as e:
        logger.error(f"Error uploading image: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to upload image")

@app.post("/analyze")
async def analyze_image(filename: str):
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    if not os.path.exists(filepath):
        raise HTTPException(status_code=400, detail="Uploaded image not found")

    try:
        with Image.open(filepath) as input_image:
            input_image_copy = input_image.copy()
        result = await process_image_with_comfy(input_image_copy, filename)
        return {"result": result}
    except Exception as e:
        logger.error(f"Error analyzing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred during analysis: {str(e)}")

async def process_image_with_comfy(input_image, filename):
    try:
        json_path = os.path.join(BASE_DIR, "drawing_analysis_workflow.json")
        with open(json_path, "r", encoding="utf-8") as file_json:
            prompt = json.load(file_json)

        # For OllamaVision node, generate a random seed
        if "6" in prompt and "inputs" in prompt["6"]:
            prompt["6"]["inputs"]["seed"] = random.randint(1, 1000000)

    except FileNotFoundError:
        logger.error(f"JSON file not found at {json_path}")
        raise HTTPException(status_code=500, detail="Workflow JSON file not found")
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON file at {json_path}")
        raise HTTPException(status_code=500, detail="Invalid JSON in workflow file")
    
    width, height = input_image.size
    min_side = min(width, height)
    scale_factor = 512 / min_side
    new_size = (round(width * scale_factor), round(height * scale_factor))
    resized_image = input_image.resize(new_size)

    upload_path = os.path.join(COMFY_INPUT_DIR, filename)
    resized_image.convert('RGB').save(upload_path, 'JPEG', quality=95)

    for node in prompt.values():
        if node.get('class_type') == 'LoadImage':
            node['inputs']['image'] = upload_path

    client_id = f"draw_api_{int(time.time())}"
    prompt_id = await queue_prompt(prompt, client_id)
    if prompt_id:
        return await get_result(prompt_id, client_id)
    else:
        raise HTTPException(status_code=500, detail="Failed to queue prompt")

async def queue_prompt(workflow, client_id):
    p = {"prompt": workflow, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    url = f"{COMFY_SERVER}/prompt"

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, data=data)
            response.raise_for_status()
            result = response.json()
            logger.debug(f"Server response: {result}")
            if 'prompt_id' not in result:
                logger.error(f"Expected 'prompt_id' in response, but got: {result}")
                return None
            return result['prompt_id']
    except httpx.RequestError as e:
        logger.error(f"Request failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Request failed: {str(e)}")

async def get_result(prompt_id, client_id):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            async with websockets.connect(f"ws://{COMFY_SERVER.replace('http://', '')}/ws?clientId={client_id}") as ws:
                logger.info(f"WebSocket connected for prompt_id: {prompt_id}")
                
                while True:
                    out = await ws.recv()
                    message = json.loads(out)
                    logger.debug(f"Received message: {message}")
                    if message['type'] == 'executing':
                        data = message['data']
                        logger.info(f"Executing node: {data['node']}")
                        if data['node'] is None and data['prompt_id'] == prompt_id:
                            logger.info("Execution completed")
                            break

            history_url = f"{COMFY_SERVER}/history/{prompt_id}"
            async with httpx.AsyncClient() as client:
                response = await client.get(history_url)
                response.raise_for_status()
                history = response.json()
                logger.debug(f"History received: {history}")
                
                if prompt_id in history:
                    outputs = history[prompt_id]['outputs']
                    logger.info(f"Outputs for prompt_id {prompt_id}: {outputs}")
                    if '54' in outputs and 'text' in outputs['54']:
                        logger.info("Returning translated text from node 54")
                        return outputs['54']['text']
                    elif '53' in outputs and 'text' in outputs['53']:
                        logger.info("Returning original text from node 53")
                        return outputs['53']['text']
                else:
                    logger.warning(f"Prompt ID {prompt_id} not found in history")

            return "No result found in outputs"
        except (websockets.WebSocketException, TimeoutError) as e:
            logger.error(f"WebSocket error on attempt {attempt + 1}: {str(e)}")
            if attempt == max_retries - 1:
                raise

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5003, debug=True)
