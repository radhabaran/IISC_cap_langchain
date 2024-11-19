# router_agent.py
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory

class RouterAgent:
    def __init__(self, llm: ChatOpenAI, memory: ConversationBufferMemory):
        self.llm = llm
        self.memory = memory
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Classify the query into one of two categories:
            1. product_review: Queries about product features, prices, availability, or reviews
            2. generic: All other queries
            Respond with only the category name."""),
            ("human", "{query}")
        ])

    async def classify_query(self, query: str) -> str:
        chain = self.prompt | self.llm
        response = await chain.ainvoke({"query": query})
        return response.content.strip().lower()