# main.py
import gradio as gr
from typing import Dict, List
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from agent.planning_agent import PlanningAgent
from dotenv import load_dotenv
import os

api_key = os.environ['OA_API']           
os.environ['OPENAI_API_KEY'] = api_key

class Orchestrator:
    def __init__(self):
        load_dotenv()
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo-instruct",
            temperature=0
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.planning_agent = PlanningAgent(self.llm, self.memory)

    async def process_query(self, query: str, history: List[List[str]]) -> str:
        try:
            if history:
                for human_msg, ai_msg in history:
                    self.memory.chat_memory.add_message(HumanMessage(content=human_msg))
                    self.memory.chat_memory.add_message(AIMessage(content=ai_msg))
            
            response = await self.planning_agent.execute(query)
            
            self.memory.chat_memory.add_message(HumanMessage(content=query))
            self.memory.chat_memory.add_message(AIMessage(content=response))
            
            return response
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            self.memory.chat_memory.add_message(HumanMessage(content=query))
            self.memory.chat_memory.add_message(AIMessage(content=error_msg))
            return error_msg

    def clear_context(self):
        self.planning_agent.clear_context()
        self.memory.clear()
        return [], []

def create_gradio_app(orchestrator: Orchestrator) -> gr.Blocks:
    from app import create_interface
    return create_interface(orchestrator)

def main():
    orchestrator = Orchestrator()
    app = create_gradio_app(orchestrator)
    app.queue()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)

if __name__ == "__main__":
    main()