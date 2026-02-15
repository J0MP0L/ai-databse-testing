from langchain.tools import tool, InjectedToolArg
from typing import List, Dict, Any, Optional, Annotated, Union, Literal, TypedDict, Tuple
import os
from dotenv import load_dotenv

import boto3
from pymongo import MongoClient
from pymongo.errors import PyMongoError
import pandas as pd
import io
from datetime import datetime, timedelta
import uuid
import traceback
import numpy as np
from plotly.graph_objs import Figure
import sys
import plotly.express as px

load_dotenv(override=True)
MONGODB_URI = os.getenv("MONGODB_URI")

BUCKET_NAME = "agenticai6"
LIMIT_DATA = 1000

## tool สำหรับ mongodb_agent ในการ query มาตอบ
@tool
def aggregate_sendingURL_tool(
    db_name: str,
    collection_name: str,
    pipeline: List[Any],  
    url_sending: Optional[bool] = False,       
    hint: Optional[Any] = None 
) -> Union[List, str, Tuple[str, List]]:
    """
    Execute a MongoDB aggregation pipeline. Use for sending url for download data to the user when user want to download data or the user want data with many row

    Args:
        db_name (str): The name of the database to query. AI must only use the database specified for its task.
        
        collection_name (str): The name of the collection within the database to query. AI must choose the correct collection
            relevant to the user's question.
        
        pipeline (List[Dict]): A list of dictionaries representing the MongoDB aggregation pipeline. 
            - AI is required to construct this pipeline according to the calculation or analysis needed.
            - The pipeline must include all stages explicitly:
                - $match: filter documents
                - $group: calculate count, sum, avg, min, max or group by fields
                - $sort: order the results
                - $skip: skip documents for pagination
                - $limit: limit the number of results
            - Example:
              [
                  {"$match": {"status": "active"}},
                  {"$group": {"_id": "$owner_id", "total_amount": {"$sum": "$amount"}}},
                  {"$sort": {"total_amount": -1}},
                  {"$limit": 10}
              ]
            - AI must not leave the pipeline empty and must not use find() style filter parameters outside the pipeline.

        url_sending (bool, optional): Set to True if the user wants a URL to download the data **but** if user doesn't ask to download set it to False (default value).
        
        hint (dict, optional): Optional index hint to optimize the aggregation query. 
            - Must be a dictionary or string recognized by MongoDB. 
            - Example: {"owner_id": 1} or "index_name".
            - If not needed, set to None.
    Returns:
        str: Link URL returned by MongoDB according to the pipeline.
    """

    if not isinstance(pipeline, list) or len(pipeline) == 0:
        return "You must provide a non-empty aggregation pipeline."
    # ใส่ LIMIT ไว้เสมอ
    last_stage = pipeline[-1]
    client = MongoClient(MONGODB_URI)
    db = client[db_name]
    collection = db[collection_name]
    try:
        # Execute aggregation
        cursor_args = {}
        if hint is not None: 
            cursor_args['hint'] = hint
        #ถ้า pipeline ใส่ limit เข้าไปเพื่อป้องกันการดึงข้อมูลจำนวนมากเกินไป
        pipeline.append({"$limit": LIMIT_DATA})
        cursor = collection.aggregate(pipeline, **cursor_args)
        ## กรณีไม่ต้องส่ง url
        if not url_sending:
            df_list = list(cursor)
            return df_list if df_list else "Data not found. Check pipeline and use this tool again if there are bugs."
        ## กรณีส่ง url
        df_list = list(cursor)
        df = pd.DataFrame(df_list)
        if df.empty:
            return "Data not found"
        stream = io.BytesIO()
        with pd.ExcelWriter(stream, engine = "xlsxwriter") as writer:
            df.to_excel(writer, index = False)
        stream.seek(0)
        # Upload ไปยัง S3 
        s3 = boto3.client("s3")
        bucket_name = BUCKET_NAME
        # ใช้ folder prefix สำหรับไฟล์ชั่วคราว
        filename = f"aiDb/{datetime.now().strftime('%Y%m%d')}/{uuid.uuid4()}.xlsx"
        # Upload with metadata
        s3.upload_fileobj(
            stream, 
            bucket_name, 
            filename,
        )
        # สร้าง presigned URL 
        url = s3.generate_presigned_url(
            ClientMethod="get_object",
            Params={
                "Bucket": bucket_name, 
                "Key": filename,
            },
            ExpiresIn=300  # 5 นาที
        )
        ## ใช้เพื่อ extract file name เอาไว้ลบทีหลัง
        return (
            f"[filename]:{filename}///Download your data here (link expires in 10 minutes): {url}", 
            df_list
        )
    except PyMongoError as e:
        return f"MongoDB pipeline error, USE THE ERROR TO IMPROVE YOU CODE AND TRY TO USE THIS TOOL AGAIN: {str(e)}"
    except Exception as e:
        return f"Error, USE THE ERROR TO IMPROVE YOU CODE AND TRY TO USE THIS TOOL AGAIN: {str(e)}"
    finally:
        client.close()


# ใช้เพื่อให้ llm มองไม่เห็น argument พวกนี้ใน tool llm จะได้ไม่ต้องใส่มา
ToolRuntime = Annotated[object, InjectedToolArg]
## fucntion เพื่อ execute code จาก llm
@tool
def python_execute(
    code: str,
    df: ToolRuntime, ## llm มองไม่เห็น
) -> Figure | str:
    """
    Execute the python code to generate graph.
    
    Args:
        code: The Python code to be executed. Must create a Plotly figure named 'fig'.
    """
    
    SAFE_GLOBAL = {
        "pd": pd,
        "px": px,
        "np": np
    }
    SAFE_LOCAL = {
        "df": df
    }
    python_code = code
    # Capture stdout from the executed code
    _stdout_buf, _old_stdout = io.StringIO(), sys.stdout
    sys.stdout = _stdout_buf
    err_text = None
    try:
        exec(python_code, SAFE_GLOBAL, SAFE_LOCAL)
    except Exception:
        err_text = traceback.format_exc()
    finally:
        sys.stdout = _old_stdout
    printed = _stdout_buf.getvalue().strip()
    fig = SAFE_LOCAL.get("fig", None)
    # ลำดับการ return: fig → err_text → message
    return (
        fig 
        if fig is not None 
        else (err_text or "[Error]: You must name the output of the graph to be 'fig' don't use other name")
    ) # ถ้ามี fig จะ return fig ถ้ามี error จะ return error ก่อน เเละให้ message อันดับสุดท้าย


## สร้าง schema_block ให้กับ prompt
def create_schema(
    df: pd.DataFrame
    ) -> str:
    columns_name = list(df.columns)
    nunique = {}
    for col in df.columns:
        try:
            nunique[col] = df[col].nunique()
        except:
            nunique[col] = "non-hashable"

    context = f"DASET has {df.shape[0]} rows, {df.shape[1]} columns\nColumns: {columns_name}\n\n"

    for col in columns_name:
        context += f" COLUMN: {col} has nunique: {nunique[col]}, "
        if pd.api.types.is_numeric_dtype(df[col]):
            context += f"range: {df[col].min():.2f} to {df[col].max():.2f}, mean: {df[col].mean():.2f}"
        else:
            text = "\n".join(
                [f"{idx}: {count}" for idx, count in df[col].value_counts().head(2).items()]
            )
            context += f"mode: {text}"
        sample = "\n ".join(
            [f"{idx}: {count}" for idx, count in df[col].head(3).items()]
        )
        context += f" sample data first 3 rows {sample}"

    return context
