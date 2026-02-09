from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, add_messages
from langchain.tools import tool, InjectedToolArg
from langchain.messages import AnyMessage, ToolMessage
from langgraph.graph.state import CompiledStateGraph
from langchain_core.messages import BaseMessage

import boto3
import base64
import re
from pymongo import MongoClient
from pymongo.errors import PyMongoError
import pandas as pd
import io
import sys
import plotly.express as px
from plotly.graph_objs import Figure
import numpy as np
import traceback
from datetime import datetime, timedelta
import uuid
import asyncio
import plotly.graph_objects as go

from typing import List, Dict, Any, Optional, Annotated, Union, Literal, TypedDict
from pydantic import BaseModel, Field
import operator
import os
from IPython.display import Markdown, display, clear_output, HTML, Image
from dotenv import load_dotenv

from code.prompt import prompt_mockdata

load_dotenv(override=True)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")

LIMIT = 10
MAX_TOOL_CALLS = 6
MAX_EVAL_CALLS = 3
MAX_QUERYDB = 3
WINDOW_SIZE = 10
BUCKET_NAME = "agenticai6"

def create_llm(
    model: str, 
    temperature: float = 0.3, 
    max_retries: int = 6, 
    reasoning_effort: str | None = None,
    tags: str | None = None
) -> CompiledStateGraph:
    llm = ChatOpenAI(model=model,
            openai_api_key = OPENROUTER_API_KEY,
            openai_api_base = "https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            temperature = temperature,
            tags = tags,
            reasoning_effort = reasoning_effort)
    llm.with_retry(stop_after_attempt = max_retries)
    return llm

## tool สำหรับ mongodb_agent ในการ query มาตอบ
@tool
def aggregate_sendingURL_tool(
    db_name: str,
    collection_name: str,
    pipeline: List[Any],  
    url_sending: Optional[bool] = False,       
    hint: Optional[Any] = None 
) -> Union[List, str]:
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
        cursor = collection.aggregate(pipeline, **cursor_args)
        ## กรณีไม่ต้องส่ง url
        if not url_sending:
            df_list = list(cursor)
            return df_list if df_list else "Data not found"
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
        return f"MongoDB pipeline error: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"
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

### prompt database schema
prompt_database = prompt_mockdata


### prompt supervisor agent
propmt_supervised_agent = """You are a supervisor agent. Your responsibility is to gather required information by asking follow-up questions to the user
and to delegate tasks to other agents appropriately.

DATABASES, COLLECTIONS, AND FIELDS:
{prompt_database}

RESPONSIBILITIES:
- Based on the user's question, determine whether a start date and end date are required
  in order to answer the question correctly.
  - If needed, ask the user to specify the desired start and end dates. BUT USER CAN ASK TO SEE ALL THE DATA AVAILABLE.
  - DO NOT ASK ANYTHING OTHER THAN THE DATE. TRY TO ANSWER USER YOUR SELF.
- Once all required information is available, write detailed and explicit instructions
  describing how to query the database and send them to the mongodb_agent,
  so that the agent can retrieve the correct data.
- Only if necessary — or if the user explicitly requests a graph only —
  send the database query steps to the code_agent so that it can generate a graph.

AGENTS:
1. mongodb_agent:
   Can query the MongoDB database and use the retrieved data to respond to the user
   with clear and easy-to-understand explanations.
   It can send URL to user so that the user can download the data in xlsx format.
   YOU MUST TELL mongodb_agent THAT DO NOT INCLUDE ID OR TECHNICAL WORD IN ITS ANSWER
2. code_agent:
   Can query the MongoDB database and use the retrieved data to write Python code
   that plots graphs for the user. It only provide a graph, can not generate text.
Note: - The mongodb_agent agents can only use 'aggregate' function to query MongoDB database.
      - code_agent will use the same data from mongodb_agent to generate graph.

DATE TODAY:
{datetime}

RULES:
- Instructions sent to other agents must be detailed, clearly describing how to query
  the data correctly and what kind of output or response is expected.
- For mongodb_agent instead of describe of to query you should tell them the aggregate code use for query database in python.
- Call the code_agent only when it is necessary to generate a graph.
- Use code_agent to plot only one graph. Select the graph that most important for the user.
- If user ask for download data tell mongodb_agent to send URL otherwise do not tell mongodb_agent.


OUTPUT FORMAT (important):
- When you want to call mongodb_agent or code_agent you must response in this format: <agent_name> it's task 
  e.g.
    <mongodb_agent> His tasks.  <code_agent> His task.
    if you want to call only mongodb_agent you can output in this format:
    <mongodb_agent> His tasks.
  YOU ARE NOT ALLOW TO CALL ONE AGENT MANY TIME e.g. <mongodb_agent> His tasks. <mongodb_agent> His tasks.
- If you want to ask the user you can ask without any output format.

YOU MUST USE ONLY owner_id: {owner_id} FOR QUERY DATABASE.
"""

## prompt mongodb agent
prompt_mongodb_agent = """You are a MongoDB expert for querying data to generate report to answer the user's question.

DATABASES & COLLECTIONS & FIELD:
  {prompt_database}

RESPONSE:
- Response in Thai.
- Use the data obtained from the tool to generate a report.
- The report must include three sections:
    1. An introduction to the report.
    2. The main points.
    3. A conclusion.
- The report must be relevant to the user's query.
- Query more fields than the user explicitly requests, provided they are relevant to the user’s query.
- THE REPORT MUST EASY TO UNDERSTAND CONVERT NAME OF THE FIELDS SO THE USER KNOW WHAT IS IT AND DON'T USE ANY TECHNICAL TERM.

DATE TODAY:
  {datetime}

LIMITATION:
- You can only query the data. You can't change or delete the data.
- The data you query will Limit at {LIMIT} 

RULES:
- DO NOT QUERY THE FIELD THAT NOT APPEAR IN THE COLLECTION.
- If you cannot answer the user's question because required data is unclear, ask a clarification question using ONLY human-readable descriptions.
  e.g. [WRONG]"คุณต้องการหา field is_deleted ใช่หรือไม่"
       [RIGHT]"คุณต้องการหาว่าบิลนี้ถูกลบไปเเล้วหรือยังใช่ไหม"
- DO NOT INCLUDE ID, NAME OF DATABASE, COLLECTION, FIELD OR ANY TECHNICAL TERM AND SECRET NAME IN YOUR ANSWER.

YOU MUST USE ONLY owner_id: {owner_id} FOR QUERY DATABASE.
"""


## prompt ที่เห็น schema แบบ dynamic
prompt_code_agent = """You are a data visualization expert. PLAN BY WRITING PYTHON CODE USING PANDAS AND PLOTLY.

Database Schema & Samples (read-only):
{schema_block}

Execution Environment (already imported/provided):
- Variables: df # List[pd.DataFrame] (The df is already in you environment)
- Helpers: pd -> pandas library, px -> plotly library, np -> numpy library (These function is in you environment you don't need to import)
- You can not import any module. You must use only pd, np, px.

RULES:
- You must should index of the dataframe to plot like df[0] or df[1] in your code.
- Always make your graph look beautiful.
- Do not include any comment in your code.

HUMAN RESPONSE REQUIREMENT (hard):
- You must name the graph to be 'fig' 
    e.g. fig = px.bar(df, x, y)
    When you write this code your job is done. Your must **don't include** fig.show() or save any file
- If x-axis is timeseries data YOU MUST SORT TIMESERIES DATA before ploting.
- You shiuld give the name to x-axis, y-axis and the title of the graph.
- If the graph have so many category, you should choose only first 15 categories to plot graph (sorted by some variable before choose first 15 categories)
"""

## prompt eval graph
prompt_eval_graph = """You are eval graph agent. YOU HAVE TO PROVIDE FEEDBACK TO THE GRAPH AND DECIDE THE GRAPH IS PASS OR NOT.

This is the Database Schema use to plot the graph:
{schema_block}

CRITERIA FOR THE GRAPH TO PASS:
- The graph is readable. Example: when using pie chart it should contain resonable amount of category.
- The graph is suitable for the x-axis and y-axis.
- If x-axis is time series data it should be sorted.

FEEDBACK MESSAGE:
- Include why the graph is pass or not.
- Include How to improve the graph if it not pass.

DATA USABILITY CHECK:
- If the data used to generate the graph is incomplete, incorrect,
  irrelevant, or unsuitable for visualization, set `data_usable` to False.
  e.g. it contain only one row or it too hard to code to process the data -> set `data_usable` to False.
- Otherwise, set `data_usable` to True.
"""

class DbState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    tool_count: int
    used_tool: bool
    owner_id: str
    dfs: Annotated[List[pd.DataFrame], operator.add]

async def mongodb_agent(dbstate: DbState) -> DbState:
    model = create_llm(model = "openai/gpt-4o", reasoning_effort="high", tags = ["mongodb"])
    used_tool = dbstate.get("used_tool", False)

    if not used_tool:
        model_with_tools = model.bind_tools([aggregate_sendingURL_tool], tool_choice="required")
        used_tool = True 
    else:
        model_with_tools = model.bind_tools([aggregate_sendingURL_tool])

    PROMPT_MONGODB_AGENT = SystemMessage(content = prompt_mongodb_agent.format(prompt_database=prompt_database, LIMIT=LIMIT, 
                                            datetime=datetime.now(), owner_id = dbstate['owner_id']))   
    messages = [PROMPT_MONGODB_AGENT] + dbstate['messages']
    response = await model_with_tools.ainvoke(messages)

    tool_count = dbstate.get("tool_count", 0)
    if response.tool_calls:
        tool_count += 1

    new_state = {'messages': response ,'tool_count': tool_count, "used_tool": used_tool}
    return new_state

async def aggregate_node(dbstate: DbState) -> DbState:
    """Query mongodb database node"""

    tasks = []
    last_message = dbstate["messages"][-1]

    for tool_call in last_message.tool_calls:
        tasks.append(
            aggregate_sendingURL_tool.ainvoke(tool_call["args"])
        )
    results = await asyncio.gather(*tasks)

    # แปลงผลลัพธ์ให้เป็น ToolMessage
    dfs = []
    tool_messages = []
    for result, tool_call in zip(results, last_message.tool_calls):
        if isinstance(result, list):
            content = str(result[:LIMIT]) ## LIMIT ไว้สำหรับ llm ในการตอบ เเต่ไม่ limit สำหรับ plot graph
            dfs.append(pd.DataFrame(result))
        elif isinstance(result, tuple):
            link = result[0]
            df = result[1]
            content = link + f"\n\n[DATA]\n {df[:LIMIT]}"
            dfs.append(pd.DataFrame(df))
        else: ## ถ้าไม่ใช่ list หรือ tuple จะเป็น str ของ error หรือ data not dound เสมอ
            content = result
        tool_messages.append(
            ToolMessage(
                content=content,
                tool_call_id=tool_call["id"]
            )
        )

    new_state = {
        "messages": tool_messages,
        "dfs": dfs
    }

    return new_state


def after_mongodb_agent(dbstate: DbState) -> Literal["aggregate_node", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = dbstate["messages"]
    last_message = messages[-1]

    if last_message.tool_calls and dbstate["tool_count"] <= MAX_TOOL_CALLS:
        return "aggregate_node"

    return END

db_graph = StateGraph(DbState)
db_graph.add_node("mongodb_agent", mongodb_agent)
db_graph.add_node("aggregate_node", aggregate_node)
db_graph.add_edge(START, "mongodb_agent")
db_graph.add_conditional_edges("mongodb_agent", after_mongodb_agent, ["aggregate_node", END])
db_graph.add_edge("aggregate_node", "mongodb_agent")
db_graph = db_graph.compile()

class SubState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages] ## มี user query เเละ graph กับ eval ไม่รวม querydb_agent
    dfs: List[pd.DataFrame] # DataFrame สำหรับเอาไว้ใช้ใน python_executor
    schema_block: str
    code: str
    html_fig: Figure # rusult กราฟ
    is_pass: bool # is_pass จาก eval
    data_usable: bool # data ที่ได้มาสามารถ ใช้ plot graph ได้จริงไหมจาก eval
    tool_count: int # นับจำนวนครั้งที่ใช้ tool 
    eval_count: int # นับจำนวนครั้งที่ทำงานไม่ผ่าน 

class eval_response_format(BaseModel):
    is_pass: bool = Field(description="Wheter the graph from the code_agent is pass or not")
    feedback: str = Field(description="Feedback to coder agent")
    data_usable: bool = Field(description="Data is usable to plot a graph")

async def call_code_agent(substate: SubState) -> SubState:
    """Agent that can execute python code to genrate graph using plotly"""
    
    schema_block = substate.get("schema_block", None)
    if schema_block is None:
        schema_block = ""
        for i in range(len(substate["dfs"])):
            schema_block += f"""[DATABASE INDEX {i}]: {create_schema(df = substate["dfs"][i])} \n\n"""
        
    messages = [SystemMessage(content = prompt_code_agent.format(schema_block=schema_block))] + substate["messages"]
    code_agent = create_llm(model = "anthropic/claude-sonnet-4.5", tags = ["code"]) 
    ## ถ้าไม่ผ่านการ evaluate หรือเป็นครั้งเเรกที่รัน ครั้งเเรก is_pass ยังไม่ define
    if not substate.get("is_pass", False):
        code_agent_with_tool = code_agent.bind_tools([python_execute], tool_choice="required")
        substate["is_pass"] = True ## รอบหน้าที่เรียกหลังจากใช้ tool มันจะไม่ required tool เเละถ้า eval ไม่ให้ผ่าน มันจะ required tool อีกครั้ง 
    else:
        code_agent_with_tool = code_agent.bind_tools([python_execute])
    response = await code_agent_with_tool.ainvoke(messages)
    ## นับจำนวนการเรียกใช้ tool call
    tool_count = substate.get("tool_count", 0)
    if response.tool_calls:
        tool_count += 1
    new_state = {"messages": response, "is_pass": substate["is_pass"], "tool_count": tool_count, "schema_block": schema_block}
    ## ถ้า tool call มากกว่า MAX_TOOL_CALLS เเปลว่าข้อมูลใช้งานไม่ได้
    if tool_count >= MAX_TOOL_CALLS:
        new_state["data_usable"] = False
    return new_state
    
## เอาไว้ execute tool เนื่องจากมี tool เดียวเลยไม่ต้องใช้ for loop
async def python_execute_node(substate: SubState):
    """Performs the python exec to plot graph"""
    last_message = substate["messages"][-1]
    tasks = []
    code = ""
    for i, tool_call in enumerate(last_message.tool_calls):
        tasks.append(
                    python_execute.ainvoke({
                    "code": tool_call["args"]["code"],
                    "df": substate["dfs"]
                }
            )
        )
        code += f"""[CODE {i}] {tool_call["args"]["code"]}"""
    results = await asyncio.gather(*tasks)
    tool_messages = []
    html_fig = None
    for result, tool_call in zip(results, last_message.tool_calls):
        if isinstance(result, str):
            tool_messages.append(
                        ToolMessage(
                            content=result,
                            tool_call_id=tool_call["id"]
                        )
                    )
        else: ## เเปลว่า plot graph สำเร็จ
            html_fig = result ## เอาเเค่ fig อันเดียว
            tool_messages.append(
                        ToolMessage(
                            content="You have successfully created the graph. [DON'T ANSWER ANYTHING JUST SAY 'OK' TO SPEED UP RUNTIME]",
                            tool_call_id=tool_call["id"]
                        )
                    )
    new_state = {"messages": tool_messages, "code": code}
    if html_fig is not None:
        new_state["html_fig"] = html_fig
    return new_state

## node ที่เอาไว้ตัดสินว่าจะไปเข้า tool หรือจะ end แค่นี้
def should_continue(substate: SubState) -> Literal["tool_node", "call_eval_graph"]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = substate["messages"]
    last_message = messages[-1]

    ## หากมีการเรียก tool_call มากกว่า 10 ครั้งในการรันครั้งเดียวจะบังคับจบทันที
    if last_message.tool_calls and substate["tool_count"]<=MAX_TOOL_CALLS:
        return "python_execute_node"

    return "call_eval_graph"

async def call_eval_graph(substate: SubState) -> SubState:
    """Eval agent that evaluate the graph ploted by code_agent"""

    schema_block = substate["schema_block"]
    eval_graph = create_llm(model="openai/gpt-4o")
    eval_graph = eval_graph.with_structured_output(eval_response_format)

    try:
        img_bytes = substate['html_fig'].to_image(format="png")
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    except Exception as e:
        print(f"[call_eval_error]: No html_fig -> {e}")
        new_state = {"data_usable": False}
        return new_state

    code_message = f"""This is the code AI use to generate this graph:\n\n{substate["code"]}"""
    message = HumanMessage(content_blocks=[
        {"type": "text", "text": code_message  + "\n\nPlease evaluate this graph."},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_b64}},
    ])
    response = await eval_graph.ainvoke(
        [SystemMessage(prompt_eval_graph.format(schema_block=schema_block))]+[message]
    )
    eval_count = substate.get("eval_count", 0)
    eval_count += 1
    new_state = {"messages":HumanMessage("The feedback from the graph is: "+response.feedback), "is_pass":response.is_pass, "data_usable":response.data_usable, "eval_count":eval_count}
    ## ถ้า eval_count >= MAX_EVAL_CALLS เเละยังไม่ผ่าน ยังไม่ผ่านเเปลว่าข้อมูลใช้ไม่ได้
    if (eval_count >= MAX_EVAL_CALLS) and (not response.is_pass):
        new_state["data_usable"] = False
    return new_state

def should_end(substate: SubState) -> Literal[END, "call_code_agent"]:
    """Decide whether itshould call code_agent again or end"""
    
    if (substate["is_pass"]) or (not substate["data_usable"]) or (substate["eval_count"]>=MAX_EVAL_CALLS):
        return END     
    else:
        return "call_code_agent"

code_graph = StateGraph(SubState)
code_graph.add_node("call_code_agent", call_code_agent)
code_graph.add_node("python_execute_node", python_execute_node)
code_graph.add_node("call_eval_graph", call_eval_graph)
code_graph.add_edge(START, "call_code_agent")
code_graph.add_conditional_edges(
    "call_code_agent",
    should_continue,
    ["call_eval_graph", "python_execute_node"]
)
code_graph.add_edge("python_execute_node", "call_code_agent")
code_graph.add_conditional_edges(
    "call_eval_graph",
    should_end,
    ["call_code_agent", END]
)
code_graph = code_graph.compile()

class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    within_messages: Annotated[List[AnyMessage], add_messages]
    owner_id: str 
    dfs: List[pd.DataFrame] | None
    db_message: str | None
    code_message: str | None
    html_fig: Figure | None

def extract_output_supervised(output: str) -> List[Dict]:
    """Parse supervised output"""
    pattern = r'<(\w+)>\s*(.*?)(?=<\w+>|$)'
    matches = re.findall(pattern, output, re.DOTALL)

    results = []
    for agent_name, task in matches:
        if agent_name.strip() in ["mongodb_agent", "code_agent"]:
            results.append({
                'agent': agent_name.strip(),
                'task': task.strip()
            })
    
    return results
    
async def call_supervisor_agent(state: State) -> State:
    prompt = SystemMessage(content = propmt_supervised_agent.format(
        prompt_database=prompt_database, 
        datetime=datetime.now(),
        owner_id = state["owner_id"]) )
    messages = [prompt] + state["messages"] + state.get("within_messages", "")
    supervisor_agent = create_llm(model = "openai/gpt-5.2", temperature=0.0, reasoning_effort="low", tags=["supervisor"])
    response = await supervisor_agent.ainvoke(messages)
    new_state = {"within_messages": response}
    return new_state

def after_supervisor(state: State) -> State:
    results = extract_output_supervised(state["within_messages"][-1].content)
    if len(results) == 0: ## เเปลว่า AI ถามกลับ หรืออาจตอบเอง
        new_state = {"messages": state["within_messages"][-1]}
        return new_state
    new_state = {}
    for result in results:
        if result["agent"] == "mongodb_agent":
            new_state["db_message"] = result["task"]
        elif result["agent"] == "code_agent":
            new_state["code_message"] = result["task"]
        else:
            print("Supervisor provided wrong name")
    return new_state

def willcall_mongodb_agent(state: State) -> Literal[END, "call_mongodb_agent"]:
    if state.get("db_message", None) is not None:
        return "call_mongodb_agent"
    return END

async def call_mongodb_agent(state: State) -> State:
    message = HumanMessage(state["db_message"])
    response = await db_graph.ainvoke({
        "messages": message,
        "owner_id": state["owner_id"]})
    new_state = {"messages": response["messages"][-1], "dfs": response["dfs"]}
    return new_state

async def call_graph_agent(state: State) -> State:
    message = HumanMessage(state["code_message"])
    response = await code_graph.ainvoke({
        "messages": message,
        "dfs":  state["dfs"]})
    new_state = {"html_fig": response.get("html_fig", None)}
    return new_state

def willcall_graph_agent(state: State) -> Literal[END, "call_graph_agent"]:
    max_row = 0
    for i in range(len(state["dfs"])):
        if len(state["dfs"][i]) > max_row:
            max_row = len(state["dfs"][i])
    if (state.get("code_message", None) is not None) and (max_row > 1):
        return "call_graph_agent"
    return END

graph = StateGraph(State)
graph.add_node("call_supervisor_agent", call_supervisor_agent)
graph.add_node("after_supervisor", after_supervisor)
graph.add_node("call_mongodb_agent", call_mongodb_agent)
graph.add_node("call_graph_agent", call_graph_agent)
graph.add_edge(START, "call_supervisor_agent")
graph.add_edge("call_supervisor_agent", "after_supervisor")
graph.add_conditional_edges(
    "after_supervisor", 
    willcall_mongodb_agent,
    ["call_mongodb_agent", END]
)
graph.add_conditional_edges(
    "call_mongodb_agent",
    willcall_graph_agent,
    ["call_graph_agent", END]
)
graph.add_edge("call_graph_agent", END)
graph = graph.compile()

loading_code = """
<style>
@keyframes glitch {
  0%, 100% { clip-path: inset(0 0 0 0); }
  20% { clip-path: inset(10% 0 85% 0); }
  40% { clip-path: inset(50% 0 30% 0); }
  60% { clip-path: inset(30% 0 50% 0); }
  80% { clip-path: inset(85% 0 10% 0); }
}

@keyframes scan {
  0% { transform: translateY(-100%); }
  100% { transform: translateY(100%); }
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

.loading-container {
  position: relative;
  padding: 15px 20px;  
  background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
  border: 1px solid #00ffff;  
  border-radius: 8px;  
  box-shadow: 0 0 15px rgba(0, 255, 255, 0.3), inset 0 0 15px rgba(0, 255, 255, 0.1); 
  overflow: hidden;
  max-width: 400px;  
  margin: 0 auto; 
}

.loading-container::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 1px; 
  background: linear-gradient(90deg, transparent, #00ffff, transparent);
  animation: scan 2s linear infinite;
}

.spinner-container {
  display: flex;
  align-items: center;
  gap: 12px; 
}

.hexagon-spinner {
  position: relative;
  width: 30px; 
  height: 30px;  
  flex-shrink: 0;  
}

.hexagon {
  position: absolute;
  width: 30px; 
  height: 30px;  
  border: 2px solid transparent;  
  border-top-color: #00ffff;
  border-bottom-color: #ff00ff;
  clip-path: polygon(30% 0%, 70% 0%, 100% 50%, 70% 100%, 30% 100%, 0% 50%);
  animation: rotate 2s linear infinite;
}

.hexagon:nth-child(2) {
  animation-delay: -0.5s;
  opacity: 0.6;
}

.hexagon:nth-child(3) {
  animation-delay: -1s;
  opacity: 0.3;
}

@keyframes rotate {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.loading-text {
  font-family: 'Courier New', monospace;
  font-size: 14px;  
  font-weight: bold;
  color: #00ffff;
  text-shadow: 0 0 5px rgba(0, 255, 255, 0.8), 0 0 10px rgba(0, 255, 255, 0.6);  
  animation: glitch 3s infinite, pulse 2s infinite;
}

.loading-dots {
  display: inline-block;
}

.loading-dots::after {
  content: '';
  animation: dots 1.5s steps(4, end) infinite;
}

@keyframes dots {
  0%, 20% { content: ''; }
  40% { content: '.'; }
  60% { content: '..'; }
  80%, 100% { content: '...'; }
}

.progress-bar {
  width: 100%;
  height: 2px;  
  background: rgba(0, 255, 255, 0.1);
  border-radius: 1px;
  margin-top: 10px; 
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #00ffff, #ff00ff, #00ffff);
  background-size: 200% 100%;
  animation: progress 2s linear infinite;
}

@keyframes progress {
  0% { background-position: 0% 0%; }
  100% { background-position: 200% 0%; }
}
</style>

<div class="loading-container">
  <div class="spinner-container">
    <div class="hexagon-spinner">
      <div class="hexagon"></div>
      <div class="hexagon"></div>
      <div class="hexagon"></div>
    </div>
    <div class="loading-text">
      Generating Graph<span class="loading-dots"></span>
    </div>
  </div>
  <div class="progress-bar">
    <div class="progress-fill"></div>
  </div>
</div>
"""



class WindowedMemoryManager:
    """Memory manager with sliding window (keep last N messages)"""
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.conversations: Dict[str, List[BaseMessage]] = {}
    
    def get_messages(self, thread_id: str) -> List[BaseMessage]:
        """Get conversation history with sliding window"""
        messages = self.conversations.get(thread_id, [])
        # เก็บแค่ window_size messages สุดท้าย
        return messages[-self.window_size:] if len(messages) > self.window_size else messages
    
    def add_user_message(self, thread_id: str, message: str | HumanMessage):
        """Add user message"""
        if isinstance(message, str):
            message = HumanMessage(content=message)
        
        if thread_id not in self.conversations:
            self.conversations[thread_id] = []
        
        self.conversations[thread_id].append(message)
        
        self._trim_if_needed(thread_id)
    
    def add_ai_message(self, thread_id: str, message: str | AIMessage):
        """Add AI response"""
        if isinstance(message, str):
            message = AIMessage(content=message)
        
        if thread_id not in self.conversations:
            self.conversations[thread_id] = []
        
        self.conversations[thread_id].append(message)
        
        self._trim_if_needed(thread_id)
    
    def _trim_if_needed(self, thread_id: str):
        """Trim messages if exceeds window size"""
        messages = self.conversations[thread_id]
        if len(messages) > self.window_size:
            # เก็บแค่ window_size messages สุดท้าย
            self.conversations[thread_id] = messages[-self.window_size:]
    
    def clear_thread(self, thread_id: str):
        """Clear conversation"""
        if thread_id in self.conversations:
            del self.conversations[thread_id]
        if thread_id in self.metadata:
            del self.metadata[thread_id]

# สร้าง memory manager
memory = WindowedMemoryManager(window_size=WINDOW_SIZE)

async def run_with_memory(user_message: str, thread_id: str, owner_id: str):
    """Run graph with memory"""
    
    history = memory.get_messages(thread_id)
    
    new_message = HumanMessage(content=user_message)
    
    all_messages = history + [new_message]
    
    # เรียก graph
    result = await graph.ainvoke({
        "messages": all_messages,
        "owner_id": owner_id,
    })
    
    memory.add_user_message(thread_id, new_message)
    
    if result["messages"]:
        last_ai_message = result["messages"][-1]
        memory.add_ai_message(thread_id, last_ai_message)
    
    return result

async def AIDB(user_message: str, thread_id: str, owner_id: str):
    history = memory.get_messages(thread_id)
    new_message = HumanMessage(content=user_message)
    memory.add_user_message(thread_id, new_message)
    all_messages = history + [new_message]

    supervisor_output = ""
    mongodb_output = ""

    final_output = None
    is_never = True
    async for _, mode, metadata in graph.astream(
        {
            "messages": all_messages,
            "owner_id": owner_id
        },
        subgraphs=True,
        stream_mode=["messages", "values"],  
    ):
        if mode == "messages":
            if metadata[1].get("tags", None) == ["supervisor"]:
                supervisor_output += metadata[0].content
                yield {"supervisor_output": supervisor_output}

            if metadata[1].get("tags", None) == ["mongodb"]:
                mongodb_output += metadata[0].content
                yield {"mongodb_output": mongodb_output}

            if (metadata[1].get("tags", None) == ["code"]) and is_never:
                is_never = False
                yield {"loading_code": [mongodb_output, loading_code]}

        elif mode == "values":
            final_output = metadata

    yield {"final_output": final_output}

    if final_output["messages"]:
        last_ai_message = final_output["messages"][-1]
        memory.add_ai_message(thread_id, last_ai_message)

