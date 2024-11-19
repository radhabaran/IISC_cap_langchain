import gradio as gr
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from dotenv import load_dotenv
import os
import agent.planning_agent as planning_agent

# Global variables
llm = None
memory = None

def initialize_components():
    global llm, memory
    load_dotenv()
    api_key = os.environ['OA_API']           
    os.environ['OPENAI_API_KEY'] = api_key
    
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0,
        api_key=api_key
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    planning_agent.initialize_planning_agent(llm, memory)

def process_query(query, history):
    try:
        if history:
            for human_msg, ai_msg in history:
                memory.chat_memory.add_message(HumanMessage(content=human_msg))
                memory.chat_memory.add_message(AIMessage(content=ai_msg))
        
        response = planning_agent.execute(query)
        
        memory.chat_memory.add_message(HumanMessage(content=query))
        memory.chat_memory.add_message(AIMessage(content=response))
        
        return response
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        print(f"Error details: {str(e)}") 
        memory.chat_memory.add_message(HumanMessage(content=query))
        memory.chat_memory.add_message(AIMessage(content=error_msg))
        return error_msg

def clear_context():
    planning_agent.clear_context()
    memory.clear()
    return [], []

def create_gradio_app():
    from app import create_interface
    return create_interface(process_query, clear_context)

def main():
    initialize_components()
    app = create_gradio_app()
    app.queue()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)

if __name__ == "__main__":
    main()