import os
import uuid
import random
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import httpx
import websockets
import time
from PIL import Image
import logging
from typing import Dict
import asyncio

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 根據需要調整
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

COMFY_SERVER = "http://127.0.0.1:8188"  # 替換為您的 ComfyUI 服務地址
OUTPUT_DIR = "c:/Users/User/Desktop/app/draw_api/output"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
COMFY_INPUT_DIR = "C:/ComfyUI_windows_portable_nvidia/ComfyUI_windows_portable/ComfyUI/input"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(COMFY_INPUT_DIR, exist_ok=True)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# 用於存儲任務狀態
tasks: Dict[str, Dict] = {}

def generate_unique_filename(original_filename):
    """Generate a unique filename."""
    _, file_extension = os.path.splitext(original_filename)
    return f"{uuid.uuid4()}{file_extension}"

@app.post("/upload")
async def upload_image(image: UploadFile = File(...)):
    """
    上傳圖片並返回唯一的文件名。
    
    - **image**: 待上傳的圖片文件。
    """
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
async def analyze_image(
    background_tasks: BackgroundTasks,  # 移動到第一個參數
    filename: str = Query(...)
):
    """
    接受文件名，將圖片分析任務加入後台，並返回任務 ID。
    
    - **filename**: 上傳圖片後後端返回的唯一文件名，作為查詢參數傳遞。
    """
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    if not os.path.exists(filepath):
        raise HTTPException(status_code=400, detail="Uploaded image not found")

    # 生成唯一的任務 ID
    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "processing", "result": None}

    # 將分析任務加入後台
    background_tasks.add_task(process_image_with_comfy, input_image_path=filepath, filename=filename, task_id=task_id)

    return {"task_id": task_id}

@app.get("/result/{task_id}")
async def get_result_endpoint(task_id: str):
    """
    根據任務 ID 獲取分析結果。
    
    - **task_id**: 任務的唯一標識符。
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task ID not found")

    task = tasks[task_id]
    if task["status"] == "processing":
        return {"status": "processing", "result": None}
    elif task["status"] == "completed":
        return {"status": "completed", "result": task["result"]}
    elif task["status"] == "failed":
        return {"status": "failed", "result": task["result"]}
    else:
        return {"status": "unknown", "result": None}

async def process_image_with_comfy(input_image_path: str, filename: str, task_id: str):
    """
    後台處理圖片並分析，更新任務狀態。
    
    - **input_image_path**: 上傳圖片的路徑。
    - **filename**: 上傳圖片的文件名。
    - **task_id**: 任務的唯一標識符。
    """
    try:
        # 打開並處理圖片
        with Image.open(input_image_path) as input_image:
            input_image_copy = input_image.copy()

        # 加載 workflow JSON
        json_path = os.path.join(BASE_DIR, "drawing_analysis_workflow.json")
        with open(json_path, "r", encoding="utf-8") as file_json:
            prompt = json.load(file_json)

        # 為 OllamaVision 節點生成隨機 seed
        if "6" in prompt and "inputs" in prompt["6"]:
            prompt["6"]["inputs"]["seed"] = random.randint(1, 1000000)

        # 調整圖片大小
        width, height = input_image_copy.size
        min_side = min(width, height)
        scale_factor = 512 / min_side
        new_size = (round(width * scale_factor), round(height * scale_factor))
        resized_image = input_image_copy.resize(new_size)

        # 保存調整後的圖片到 COMFY_INPUT_DIR
        upload_path = os.path.join(COMFY_INPUT_DIR, filename)
        resized_image.convert('RGB').save(upload_path, 'JPEG', quality=95)

        # 更新 workflow 中的 LoadImage 節點
        for node in prompt.values():
            if node.get('class_type') == 'LoadImage':
                node['inputs']['image'] = upload_path

        # 生成 client_id
        client_id = f"draw_api_{int(time.time())}"

        # 將 workflow 加入 COMFY_SERVER 的提示隊列
        prompt_id = await queue_prompt(prompt, client_id)
        if not prompt_id:
            tasks[task_id]["status"] = "failed"
            tasks[task_id]["result"] = "Failed to queue prompt"
            return

        # 獲取分析結果
        result = await get_result(prompt_id, client_id)
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["result"] = result

    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["result"] = str(e)

async def queue_prompt(workflow, client_id):
    """
    將 workflow 加入 COMFY_SERVER 的提示隊列。
    
    - **workflow**: 要處理的 workflow。
    - **client_id**: 客戶端的唯一標識符。
    """
    p = {"prompt": workflow, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    url = f"{COMFY_SERVER}/prompt"

    try:
        async with httpx.AsyncClient(timeout=100) as client:
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
        return None

async def get_result(prompt_id, client_id):
    """
    通過 WebSocket 連接從 COMFY_SERVER 獲取分析結果。
    
    - **prompt_id**: 提示的唯一標識符。
    - **client_id**: 客戶端的唯一標識符。
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            async with websockets.connect(f"ws://{COMFY_SERVER.replace('http://', '')}/ws?clientId={client_id}", timeout=100) as ws:
                logger.info(f"WebSocket connected for prompt_id: {prompt_id}")
                
                while True:
                    out = await asyncio.wait_for(ws.recv(), timeout=100)
                    message = json.loads(out)
                    logger.debug(f"Received message: {message}")
                    if message['type'] == 'executing':
                        data = message['data']
                        logger.info(f"Executing node: {data['node']}")
                        if data['node'] is None and data['prompt_id'] == prompt_id:
                            logger.info("Execution completed")
                            break

            # 獲取歷史結果
            history_url = f"{COMFY_SERVER}/history/{prompt_id}"
            async with httpx.AsyncClient(timeout=100) as client:
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
        except (websockets.WebSocketException, asyncio.TimeoutError) as e:
            logger.error(f"WebSocket error on attempt {attempt + 1}: {str(e)}")
            if attempt == max_retries - 1:
                return "WebSocket connection failed after multiple attempts"
            else:
                await asyncio.sleep(1)  # 等待後重試

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 5003))  # 從環境變數讀取端口，默認為 5003
    uvicorn.run(app, host="0.0.0.0", port=port, debug=True)



