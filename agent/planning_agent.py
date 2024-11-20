from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory, SimpleMemory
import agent.router_agent as router_agent
import agent.product_review_agent as product_review_agent
import agent.generic_agent as generic_agent
import agent.composer_agent as composer_agent
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
llm = None
chat_memory = None
query_memory = None
agent = None

def initialize_planning_agent(llm_instance, chat_memory_instance, query_memory_instance):
    global llm, chat_memory, query_memory, agent
    
    llm = llm_instance
    chat_memory = chat_memory_instance
    query_memory = query_memory_instance
    
    # Initialize agents
    router_agent.initialize_router_agent(llm, chat_memory)
    product_review_agent.initialize_product_review_agent(llm, chat_memory)
    generic_agent.initialize_generic_agent(llm, chat_memory)
    # composer_agent.initialize_composer_agent(llm, memory)
    
    tools = [
        Tool(
            name="route_query",
            func=route_query,
            description="First step: Determine query type. Returns either 'product_review' or 'generic'"
        ),
        Tool(
            name="get_product_info",
            func=get_product_info,
            description="Use this for product-related queries about features, prices, availability, or reviews"
        ),
        Tool(
            name="handle_generic_query",
            func=handle_generic_query,
            description="Use this for general queries not related to products"
        ),
        Tool(
            name="compose_response",
            func=compose_response,
            description="Final step: Use this to format and enhance the response. After this step, return the response to main.py"
        )
    ]
    
    system_prompt = """You are a planning agent that processes user queries efficiently. Follow these steps:

    1. Always start by using route_query to determine the query type
    2. Based on the route_query result:
       - If 'product_review': use get_product_info
       - If 'generic': use handle_generic_query
    3. Always use compose_response as the next step after getting the response from either pget_product_info or gehandle_generic_query
    4. IMPORTANT: After compose_response returns its result, return that result immediately. 
       Do not perform any additional processing or tool calls.
    
    Important: 
    - Follow the sequence: route -> process -> compose
    - Only use one processing tool (get_product_info OR handle_generic_query) after routing
    - Always end with compose_response

    Remember: The sequence must be route -> process -> compose -> return
    """

    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        memory=chat_memory,
        system_message=system_prompt
    )
    logger.info("Planning agent initialized successfully")

def route_query(query):
    # Get original query from memory if needed
    original_query = query_memory.memories.get('original_query', query)
    return router_agent.classify_query(original_query)

def get_product_info(query):
    # Get original query from memory if needed
    original_query = query_memory.memories.get('original_query', query)
    return product_review_agent.process(original_query)

def handle_generic_query(query):
    # Get original query from memory if needed
    original_query = query_memory.memories.get('original_query', query)
    return generic_agent.process(original_query)

def compose_response(response):
    return composer_agent.compose_response(response)

def execute(query):
    try:
        # Store original query
        query_memory.memories['original_query'] = query
        return agent.run(
            f"Process this user query: {query}"
        )
    except Exception as e:
        logger.error(f"Error in planning agent: {str(e)}")
        return f"Error in planning agent: {str(e)}"

def clear_context():
    if chat_memory:
        chat_memory.clear()
    if query_memory:
        query_memory.memories.clear()
    product_review_agent.clear_context()
    generic_agent.clear_context()