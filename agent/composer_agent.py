# from langchain_openai import ChatOpenAI
# from langchain.prompts import ChatPromptTemplate
# from langchain.memory import ConversationBufferMemory
# import logging

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Global variables
# llm = None
# memory = None
# prompt = None

# def initialize_composer_agent(llm_instance, memory_instance):
#     global llm, memory, prompt
#     llm = llm_instance
#     memory = memory_instance
#     prompt = ChatPromptTemplate.from_messages([
#         ("system", "Format the response only. Do not change any wording in the response received"),
#         ("user", "{response}")
#     ])
#     logger.info("composer agent initialized successfully")

# def compose_response(response):
#     chain = prompt | llm
#     formatted = chain.invoke({"response": response})
#     print("************ In composer agent ************")
#     print("response:", response)
#     return formatted.content

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compose_response(response: str) -> str:
    """
    Process and enhance the final response
    """
    try:
        # Remove any system artifacts or unwanted patterns
        response = remove_system_artifacts(response)
        
        # Apply standard formatting
        response = format_response(response)
        
        return response
        
    except Exception as e:
        logger.error(f"Error in composition: {str(e)}")
        return response  # Fallback to original

def remove_system_artifacts(text: str) -> str:
    """Remove any system artifacts or unwanted patterns"""
    artifacts = ["Assistant:", "AI:", "Human:", "User:"]
    cleaned = text
    for artifact in artifacts:
        cleaned = cleaned.replace(artifact, "")
    return cleaned.strip()

def format_response(text: str) -> str:
    """Apply standard formatting"""
    # Add proper spacing
    formatted = text.replace("\n\n\n", "\n\n")
    
    # Ensure proper capitalization
    formatted = ". ".join(s.strip().capitalize() for s in formatted.split(". "))
    
    # Ensure proper ending punctuation
    if formatted and not formatted[-1] in ['.', '!', '?']:
        formatted += '.'
        
    return formatted