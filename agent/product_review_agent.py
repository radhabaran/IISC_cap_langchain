from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
import faiss
import warnings
import os

warnings.filterwarnings("ignore")
import pickle

# Global variables
llm = None
memory = None
# vectorstore = None

def initialize_product_review_agent(llm_instance, memory_instance):
    global llm, memory #, vectorstore
    llm = llm_instance
    memory = memory_instance
    # vectorstore = Chroma(
    #     collection_name="products",
    #     persist_directory="./data/chroma"
    # )

def process(query):
    # results = vectorstore.similarity_search(query)
    # Process results and return formatted response
    return format_response(results)

def format_response(results):
    # Format the response based on vector search results
    return "\n".join([doc.page_content for doc in results[:2]])

def clear_context():
    memory.clear()