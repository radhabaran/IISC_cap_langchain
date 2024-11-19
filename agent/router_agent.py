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
        ("human", "{input}")
    ])


def classify_query(query):
    try:
        chain = prompt | llm
        response = chain.invoke({"input": query})
        category = response.content.strip().lower()
        
        if category not in ["product_review", "generic"]:
            return "generic"  # Default fallback
        print("**** in router agent****")
        print("query :", query)
        print("category :", category)
        return category
    except Exception as e:
        print(f"Error in routing: {str(e)}")
        return "generic"  # Default fallback on error

def clear_context():
    if memory:
        memory.clear()