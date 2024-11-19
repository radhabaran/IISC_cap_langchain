from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
import agent.router_agent as router_agent
import agent.product_review_agent as product_review_agent
import agent.generic_agent as generic_agent
import agent.composer_agent as composer_agent

# Global variables
llm = None
memory = None
agent = None

def initialize_planning_agent(llm_instance, memory_instance):
    global llm, memory, agent
    
    llm = llm_instance
    memory = memory_instance
    
    # Initialize agents
    router_agent.initialize_router_agent(llm, memory)
    product_review_agent.initialize_product_review_agent(llm, memory)
    generic_agent.initialize_generic_agent(llm, memory)
    composer_agent.initialize_composer_agent(llm, memory)
    
    tools = [
        Tool(
            name="route_query",
            func=route_query,
            description="Routes the query to appropriate agent based on query type"
        ),
        Tool(
            name="get_product_info",
            func=get_product_info,
            description="Get product related information including features, prices, and reviews"
        ),
        Tool(
            name="handle_generic_query",
            func=handle_generic_query,
            description="Handle general queries not related to products"
        )
    ]
    
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        memory=memory
    )

def route_query(query):
    return router_agent.classify_query(query)

def get_product_info(query):
    return product_review_agent.process(query)

def handle_generic_query(query):
    return generic_agent.process(query)

def execute(query):
    try:
        response = agent.run(
            f"Process this query: {query}"
        )
        print("***************in planning_agent**************")
        print("response :", response)
        return composer_agent.compose_response(response)
    except Exception as e:
        return f"Error in planning agent: {str(e)}"

def clear_context():
    memory.clear()
    product_review_agent.clear_context()
    generic_agent.clear_context()