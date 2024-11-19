from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory

# Global variables
llm = None
memory = None
prompt = None

def initialize_router_agent(llm_instance, memory_instance):
    global llm, memory, prompt
    llm = llm_instance
    memory = memory_instance
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Classify the query into one of two categories:
        1. product_review: Queries about product features, prices, availability, or reviews
        2. generic: All other queries
        Respond with only the category name."""),
        ("human", "{query}")
    ])