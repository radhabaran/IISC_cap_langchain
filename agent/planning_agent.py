from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory, SimpleMemory
import agent.router_agent as router_agent
import agent.product_review_agent as product_review_agent
import agent.generic_agent as generic_agent
import agent.composer_agent as composer_agent
import logging

# Set httpx (HTTP request) logging to WARNING or ERROR level
# This will hide the HTTP request logs while keeping agent thoughts visible
logging.getLogger("httpx").setLevel(logging.WARNING)   # added on 23-Nob

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
            description="Determine query type. Returns either 'product_review' or 'generic'"
        ),
        Tool(
            name="get_product_info",
            func=get_product_info,
            description="Use this to get product-related data such as features, prices, availability, or reviews"
        ),
        Tool(
            name="handle_generic_query",
            func=handle_generic_query,
            description="Use this to get response to user queries which are generic and where the retrieval of product details are not required"
        ),
        Tool(
            name="compose_response",
            func=compose_response,
            description="Use this to only format the response. After this step, return the formatted response to main.py"
        )
    ]
    
    
    system_prompt = """You are an efficient AI planning agent. Follow these rules strictly:

    CRITICAL INSTRUCTION:
    For simple queries listed below, skip the route_query and directly go to handle_generic_query.

    SIMPLE QUERIES (NEVER use tools):
    1. Greetings: "hi", "hello", "hey", "good morning", "good evening", "good afternoon"
    2. Farewells: "bye", "goodbye", "see you", "take care"
    3. Thank you messages: "thanks", "thank you", "thanks a lot", "appreciate it"
    4. Simple confirmations: "okay", "yes", "no", "sure", "alright"
    5. Basic courtesy: "how are you?", "how are you doing?", "what's up?", "what are you doing?"
    6. Simple acknowledgments: "got it", "understood", "I see"

    FOR ALL OTHER QUERIES:
    1. Use route_query to determine if query is product_review or generic
    2. If route_query returns 'generic', use handle_generic_query and STOP
    3. If route_query returns 'product_review', use get_product_info and STOP

    EXAMPLES:
    User: "Hi"
    Thought: Simple greeting, respond directly
    Action: handle_generic_query
    Observation : "Hi! How can I help you today?"
    Thought : I have got the final answer
    Action  : compose_responses
    Final Answer: "Hi! How can I help you today?"

    User: "Tell me about smartphone batteries"
    Thought: Need to determine query type
    Action: route_query
    Action Input: "Tell me about smartphone batteries"

    Remember: For simple queries listed above, respond immediately with Final Answer WITHOUT using tools.
    """

    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        memory=chat_memory,
        system_message=system_prompt,
        early_stopping_method="generate",
        max_iterations=2
    )
    logger.info("Planning agent initialized successfully")

def route_query(query):
    # Get original query from memory if needed
    original_query = query_memory.memories.get('original_query', query)
    return router_agent.classify_query(original_query)

def get_product_info(query):
    # Get original query from memory if needed
    original_query = query_memory.memories.get('original_query', query)
    response = product_review_agent.process(original_query)

    return {
        "intermediate_steps": [],
        "output": response,
        "action": "Final Answer",
        "action_input": response
    }

def handle_generic_query(query):
    # Get original query from memory if needed
    original_query = query_memory.memories.get('original_query', query)
    response = generic_agent.process(original_query)
    return {
        "intermediate_steps": [],
        "output": response,
        "action": "Final Answer",
        "action_input": response
    }

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