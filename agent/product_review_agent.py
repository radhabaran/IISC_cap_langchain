# product_review_agent.py
from langchain_openai import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory

class ProductReviewAgent:
    def __init__(self, llm: ChatOpenAI, memory: ConversationBufferMemory):
        self.llm = llm
        self.memory = memory
        self.vectorstore = Chroma(
            collection_name="products",
            persist_directory="./data/chroma"
        )

    async def process(self, query: str) -> str:
        results = self.vectorstore.similarity_search(query)
        # Process results and return formatted response
        return self._format_response(results)

    def _format_response(self, results):
        # Format the response based on vector search results
        return "\n".join([doc.page_content for doc in results[:2]])

    def clear_context(self):
        self.memory.clear()