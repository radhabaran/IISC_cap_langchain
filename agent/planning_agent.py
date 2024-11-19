# planning_agent.py
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from agent.router_agent import RouterAgent
from agent.product_review_agent import ProductReviewAgent
from agent.generic_agent import GenericAgent
from agent.composer_agent import ComposerAgent

class PlanningAgent:
    def __init__(self, llm: ChatOpenAI, memory: ConversationBufferMemory):
        self.llm = llm
        self.memory = memory
        self.router = RouterAgent(llm, memory)
        self.product_review_agent = ProductReviewAgent(llm, memory)
        self.generic_agent = GenericAgent(llm, memory)
        self.composer = ComposerAgent(llm, memory)
        
        self.tools = [
            Tool(
                name="route_query",
                func=self.route_query,
                description="Routes the query to appropriate agent based on query type"
            ),
            Tool(
                name="get_product_info",
                func=self.get_product_info,
                description="Get product related information including features, prices, and reviews"
            ),
            Tool(
                name="handle_generic_query",
                func=self.handle_generic_query,
                description="Handle general queries not related to products"
            )
        ]
        
        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            memory=self.memory
        )

    async def route_query(self, query: str) -> str:
        return await self.router.classify_query(query)

    async def get_product_info(self, query: str) -> str:
        return await self.product_review_agent.process(query)

    async def handle_generic_query(self, query: str) -> str:
        return await self.generic_agent.process(query)

    async def execute(self, query: str) -> str:
        try:
            response = await self.agent.arun(
                f"Process this query: {query}"
            )
            return await self.composer.compose_response(response)
        except Exception as e:
            return f"Error in planning agent: {str(e)}"

    def clear_context(self):
        self.memory.clear()
        self.product_review_agent.clear_context()
        self.generic_agent.clear_context()