"""
================================================================================
MODULE: Chatbot with Memory using LangChain
================================================================================

DESCRIPTION:
    This module implements a simple conversational chatbot powered by LangChain that
    maintains conversation history and context across multiple interactions.
    The chatbot leverages language models with memory management capabilities
    to provide coherent and context-aware responses.

AUTHOR: 
    Created for LangChain Agents Project

DATE CREATED:
    January 2026

DEPENDENCIES:
    - langchain: LLM framework and tooling
    - python-dotenv: Environment variable management
    Additional dependencies listed in requirements.txt

USAGE:
    Basic usage pattern:
    >>> from chatbot_with_memory import Chatbot
    >>> bot = Chatbot()
    >>> response = bot.chat("Hello, how are you?")

VERSION: 1.0
================================================================================

The implementation follows the structure outlined:
1. Import necessary libraries and modules.
2. Set up a language model.
3. Set up a conversation chat with memory.
4. Implement a simple chat interface.
5. Test the chatbot functionality with a series of example interactions.
6. Examine how the chatbot history stored and accessed.
"""

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_classic.memory import ConversationSummaryMemory
from langchain_classic.chains import ConversationChain
from langchain_core.messages import HumanMessage, AIMessage
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
import torch

#Chatbot class definition
class Chatbot:
    def __init__(self):
        self.conversation = conversation

    def chat(self, user_input: str) -> str:
        response = self.conversation.invoke(input=user_input)
        return response
    
# Load tiny Llama model for casual language modeling
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

def load_model():
    # Check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load model with GPU and low CPU memory optimizations
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",  # Automatically map model to available GPU(s)
        #load_in_8bit=True,  # Use 8-bit quantization to reduce memory usage by ~75%
        #dtype=torch.float16,  # Use half precision for memory efficiency
    )


    print(f"Model loaded: {model_name}")
    print(f"Model size: {model.num_parameters():,} parameters")

    # Create a Hugging Face pipeline from the loaded model and tokenizer
    text_generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=100,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.7,  # Controls randomness: lower (0.1-0.3) = deterministic, higher (0.7-1.0) = creative
        top_p=0.9,  # Nucleus sampling: only consider top 90% probability mass
        top_k=50  # Only consider top 50 tokens
    )

    # Wrap the pipeline with LangChain's HuggingFacePipeline
    llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

    print("HuggingFacePipeline LLM created successfully.")

    return llm

def setup_chat(llm):
    #Set up initial chat history
    chat_history = ChatMessageHistory()
    chat_history.add_user_message("Hello, I am Naz.")
    chat_history.add_ai_message("Hi Naz! How can I assist you today?")
    print("Initial chat history set up. History: ", chat_history.messages)

    # Set up conversation memory
    memory = ConversationSummaryMemory(llm=llm, max_token_limit=500)
    conversation = ConversationChain(llm=llm, memory=memory)
    print("ConversationChain with memory created successfully.")   

    return conversation 

# Load model and set up chat
if __name__ == "__main__":
    llm = load_model()
    conversation = setup_chat(llm)
    bot = Chatbot()

    # Example interactions
    user_inputs = [
        "Can you tell me a joke?",
        "What's the weather like today?",
        "Who won the World Series in 2020?"
    ]

    for user_input in user_inputs:
        response = bot.chat(user_input)
        print(f"User: {user_input}\nBot: {response}\n")

    # Examine chat history
    print("Final chat history:")
    for message in conversation.memory.chat_memory.messages:
        print(f"{message.type}: {message.content}")

    print("Conversation summary:")
    print(conversation.memory.summary)

