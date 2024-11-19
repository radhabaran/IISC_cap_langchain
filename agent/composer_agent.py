# composer_agent.py
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory

class ComposerAgent:
    def __init__(self, llm: ChatOpenAI, memory: ConversationBufferMemory):
        self.llm = llm
        self.memory = memory
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "Format the response in a clear and concise manner."),
            ("human", "{response}")
        ])

    async def compose_response(self, response: str) -> str:
        chain = self.prompt | self.llm
        formatted = await chain.ainvoke({"response": response})
        return formatted.content