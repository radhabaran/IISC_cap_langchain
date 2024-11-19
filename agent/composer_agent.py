from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory

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

def compose_response(response):
    chain = prompt | llm
    formatted = chain.invoke({"response": response})
    return formatted.content