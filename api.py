import os
from dotenv import load_dotenv
import numpy as np
from datetime import datetime, date
from decimal import Decimal
import yaml

# โหลด environment ก่อนทุกอย่าง
load_dotenv(override=True)

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
from code.llm import AIDB
from langchain_core.messages import BaseMessage
import pandas as pd
from plotly.graph_objs import Figure
import plotly.io as pio
from bson import ObjectId
from uuid import UUID

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")

app = FastAPI()

app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")
app.mount("/code", StaticFiles(directory="code"), name="code")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    user_message: str
    thread_id: str
    owner_id: str

import base64
import numpy as np
import pandas as pd
from datetime import datetime, date

def decode_binary_data(data_dict):
    """Decode Plotly binary data format"""
    if not isinstance(data_dict, dict) or 'bdata' not in data_dict:
        return data_dict
    
    dtype = data_dict.get('dtype', 'f8')
    bdata = data_dict['bdata']
    
    try:
        decoded = base64.b64decode(bdata)
        
        dtype_map = {
            'i1': np.int8, 'i2': np.int16, 'i4': np.int32, 'i8': np.int64,
            'u1': np.uint8, 'u2': np.uint16, 'u4': np.uint32, 'u8': np.uint64,
            'f4': np.float32, 'f8': np.float64,
        }
        
        numpy_dtype = dtype_map.get(dtype, np.float64)
        arr = np.frombuffer(decoded, dtype=numpy_dtype)
        
        return arr.tolist()
    except Exception as e:
        print(f"Error decoding binary data: {e}")
        return []

def serialize_value(val):
    """แปลงค่าให้เป็น JSON serializable"""
    if val is None:
        return None
    elif isinstance(val, (str, int, float, bool)):
        if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
            return None
        return val
    elif isinstance(val, (pd.Timestamp, datetime, date)):
        return val.isoformat()
    elif isinstance(val, np.datetime64):
        return pd.Timestamp(val).isoformat()
    elif isinstance(val, (np.integer, np.floating)):
        item = val.item()
        if isinstance(item, float) and (np.isnan(item) or np.isinf(item)):
            return None
        return item
    elif isinstance(val, np.ndarray):
        return [serialize_value(v) for v in val.tolist()]
    elif isinstance(val, (list, tuple)):
        return [serialize_value(v) for v in val]
    elif isinstance(val, dict):
        # เช็คว่ามี bdata หรือไม่
        if 'bdata' in val:
            return decode_binary_data(val)
        return {k: serialize_value(v) for k, v in val.items()}
    else:
        return val

def convert_plotly_figure(fig):
    """แปลง Plotly Figure เป็น dict ที่ serialize ได้"""
    if isinstance(fig, Figure):
        # ใช้ to_json แล้ว parse
        fig_json = pio.to_json(fig, validate=False, remove_uids=True)
        fig_dict = json.loads(fig_json)
        
        print("##################")
        print("Original data type:", type(fig_dict['data'][0].get('x')))
        
        # แปลงทั้ง dict (จะจัดการทั้ง binary data และ numpy types)
        serialized = serialize_value(fig_dict)
        
        print("After serialize - x (first 5):", serialized['data'][0].get('x', [])[:5] if serialized.get('data') else None)
        print("######################")
        
        return {
            "type": "plotly_figure",
            "data": serialized.get("data", []),
            "layout": serialized.get("layout", {})
        }
    
    return fig

def convert_to_serializable(obj):
    """แปลง object เป็น JSON serializable recursively"""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    
    # MongoDB types
    elif isinstance(obj, ObjectId):
        return str(obj)
    
    elif isinstance(obj, BaseMessage):
        result = {
            "type": obj.__class__.__name__,
            "content": obj.content,
        }
        if hasattr(obj, 'tool_calls') and obj.tool_calls:
            result["tool_calls"] = [
                {
                    "name": tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None),
                    "args": tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {}),
                    "id": tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                }
                for tc in obj.tool_calls
            ]
        if hasattr(obj, 'tool_call_id'):
            result["tool_call_id"] = obj.tool_call_id
        return result
        
    elif isinstance(obj, pd.DataFrame):
        df_copy = obj.copy()
        datetime_cols = df_copy.select_dtypes(include=['datetime64', 'datetime']).columns
        for col in datetime_cols:
            df_copy[col] = df_copy[col].astype(str)
        
        # แปลง ObjectId columns
        for col in df_copy.columns:
            if df_copy[col].apply(lambda x: isinstance(x, ObjectId)).any():
                df_copy[col] = df_copy[col].apply(lambda x: str(x) if isinstance(x, ObjectId) else x)
        
        return {
            "type": "dataframe",
            "data": df_copy.to_dict(orient='records'),
            "columns": obj.columns.tolist(),
            "shape": list(obj.shape)
        }

    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, Figure):
        return convert_plotly_figure(obj)
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat()
    elif isinstance(obj, Decimal):
        return float(obj)
    
    # UUID
    elif isinstance(obj, UUID):
        return str(obj)
    
    # bytes
    elif isinstance(obj, bytes):
        return obj.decode('utf-8', errors='ignore')
    
    # set
    elif isinstance(obj, set):
        return list(obj)
    
    elif obj is None:
        return None
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    elif hasattr(obj, 'model_dump'):
        try:
            return convert_to_serializable(obj.model_dump())
        except:
            pass
    elif hasattr(obj, 'dict'):
        try:
            return convert_to_serializable(obj.dict())
        except:
            pass
    elif hasattr(obj, 'to_dict'):
        try:
            return convert_to_serializable(obj.to_dict())
        except:
            pass
    
    # ถ้าไม่รู้จัก object ให้ลอง convert เป็น string
    try:
        return str(obj)
    except:
        return f"<unserializable: {type(obj).__name__}>"
    
@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Stream chat response"""
    
    async def event_generator():
        try:
            async for output in AIDB(
                user_message=request.user_message,
                thread_id=request.thread_id,
                owner_id=request.owner_id
            ):
                # แปลงเป็น serializable
                serializable_output = convert_to_serializable(output)
                
                # Serialize to JSON
                json_str = json.dumps(
                    serializable_output,
                    ensure_ascii=False
                )
                
                yield f"data: {json_str}\n\n"
        
        except Exception as e:
            import traceback
            error_output = {
                "error": str(e),
                "type": "error",
                "traceback": traceback.format_exc()
            }
            print("Error in event_generator:")
            print(traceback.format_exc())
            yield f"data: {json.dumps(error_output, ensure_ascii=False)}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

@app.get("/")
async def index():
    try:
        with open("frontend/chat/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>AI Database API</h1><p>API is running. Use POST /api/chat to interact.</p>")



class PromptUpdate(BaseModel):
    prompt: str

@app.post("/api/update-prompt")
async def update_prompt(data: PromptUpdate):
    # อ่านไฟล์เดิม
    with open('code/prompt/prompt_mockdata.yaml', 'r', encoding='utf-8') as f:
        yaml_data = yaml.safe_load(f)
    
    # อัพเดท prompt
    yaml_data['prompt_database'] = data.prompt
    
    # เขียนกลับ
    with open('code/prompt/prompt_mockdata.yaml', 'w', encoding='utf-8') as f:
        f.write('prompt_mockdata: |\n')
        for line in data.prompt.split('\n'):
            f.write(f'  {line}\n')
    
    return {"status": "success"}


@app.get("/admin")
def admin_prompt():
    try:
        with open("frontend/update_prompt/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>AI Database API</h1><p>API is running. Use POST /api/chat to interact.</p>")

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "openrouter_key": "configured" if OPENROUTER_API_KEY else "missing",
        "mongodb_uri": "configured" if MONGODB_URI else "missing"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)