from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
llm = None
memory = None
prompt = None

def initialize_composer_agent(llm_instance, memory_instance):
    global llm, memory, prompt
    llm = llm_instance
    memory = memory_instance
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Format the response in a clear and concise manner."),
        ("human", "{response}")
    ])
    logger.info("composer agent initialized successfully")

def compose_response(response):
    chain = prompt | llm
    formatted = chain.invoke({"response": response})
    return formatted.content