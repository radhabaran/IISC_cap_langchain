# generic_agent.py
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory

class GenericAgent:
    def __init__(self, llm: ChatOpenAI, memory: ConversationBufferMemory):
        self.llm = llm
        self.memory = memory
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant handling general queries."),
            ("human", "{query}")
        ])

    async def process(self, query: str) -> str:
        chain = self.prompt | self.llm
        response = await chain.ainvoke({"query": query})
        return response.content

    def clear_context(self):
        self.memory.clear()