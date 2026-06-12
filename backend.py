from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langgraph.graph import StateGraph,START,END
from typing import TypedDict,Annotated,Literal
from langchain_core.messages import BaseMessage,HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field
from dotenv import load_dotenv
import os
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-235B-A22B-Instruct-2507",
    task="text-generation",
   
    temperature=2,
    max_new_tokens=100
)


model=ChatHuggingFace(llm=llm)
from langgraph.graph.message import add_messages
class ChatState(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]

def chat_node(state:ChatState):
    msg=state['messages']
    response=model.invoke(msg)
    return {'messages':[response]}

checkpointer=InMemorySaver()
graph=StateGraph(ChatState)
graph.add_node('chat_node',chat_node)
graph.add_edge(START,'chat_node')
graph.add_edge('chat_node',END)
workflow=graph.compile(checkpointer=checkpointer)

