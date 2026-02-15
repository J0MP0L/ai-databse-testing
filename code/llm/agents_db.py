from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, add_messages
from langchain.messages import AnyMessage, ToolMessage
from langgraph.graph.state import CompiledStateGraph
from langchain_core.messages import BaseMessage

import base64
import re
import pandas as pd
from plotly.graph_objs import Figure
from datetime import datetime, timedelta
import asyncio
import plotly.graph_objects as go

from typing import List, Dict, Any, Optional, Annotated, Union, Literal, TypedDict
from pydantic import BaseModel, Field
import operator
import os
from dotenv import load_dotenv

from ..prompt import prompt_mockdata, prompt_code_agent, prompt_eval_graph, prompt_mongodb_agent, prompt_supervised_agent
from .tools import aggregate_sendingURL_tool, python_execute, create_schema
from .others import loading_code
### prompt database schema
prompt_database = prompt_mockdata

load_dotenv(override=True)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

LIMIT = 10
MAX_TOOL_CALLS = 6
MAX_EVAL_CALLS = 3
MAX_QUERYDB = 3
WINDOW_SIZE = 10

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

class DbState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    tool_count: int
    used_tool: bool
    owner_id: str
    dfs: Annotated[List[pd.DataFrame], operator.add]

async def mongodb_agent(dbstate: DbState) -> DbState:
    model = create_llm(model = "anthropic/claude-sonnet-4.5", reasoning_effort="high", tags = ["mongodb"])
    used_tool = dbstate.get("used_tool", False)

    if not used_tool:
        model_with_tools = model.bind_tools([aggregate_sendingURL_tool], tool_choice="required")
        used_tool = True 
    elif dbstate["tool_count"] <= MAX_TOOL_CALLS:
        model_with_tools = model.bind_tools([aggregate_sendingURL_tool])
    else:
        model_with_tools = model

    PROMPT_MONGODB_AGENT = SystemMessage(content = prompt_mongodb_agent.format(prompt_database=prompt_database, LIMIT=LIMIT, 
                                            datetime=datetime.now(), owner_id = dbstate['owner_id'], MAX_TOOL_CALLS=MAX_TOOL_CALLS))   
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

        if dbstate["tool_count"] >= MAX_TOOL_CALLS:
            content += "\n\nYou have reached the maximum number of tool calls. You must answer the user's question based on the data you have retrieved."
        
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

    if last_message.tool_calls:
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
    prompt = SystemMessage(content = prompt_supervised_agent.format(
        prompt_database=prompt_database, 
        datetime=datetime.now(),
        owner_id = state["owner_id"]) )
    messages = [prompt] + state["messages"] + state.get("within_messages", "")
    supervisor_agent = create_llm(model = "openai/gpt-5.2-codex", temperature=0.0, reasoning_effort="low", tags=["supervisor"])
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
    message = state["messages"] + [AIMessage(content=state["db_message"])]
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

