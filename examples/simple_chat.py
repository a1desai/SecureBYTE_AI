"""
Simple Chat Example
Shows how to use the LLM Manager for a basic chat application
"""

import os
import sys
import time
from dotenv import load_dotenv

# Add parent directory to path to import from main
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import LLMManager

def simple_chat():
    """Run a simple chat loop with the current provider"""
    load_dotenv()
    
    # Initialize with default provider from config.py
    llm = LLMManager()
    
    print(f"🤖 Simple Chat with {llm.current_provider.capitalize()}")
    print(f"Model: {llm.get_model_config().get('model')}")
    print("=" * 50)
    print("Type 'exit' to quit, 'switch:<provider>' to change providers")
    print("=" * 50)
    
    # Set a custom system prompt
    system_prompt = """You are a friendly and helpful AI assistant. 
You provide concise, accurate, and helpful responses to user questions.
You have a friendly personality and occasionally use emojis to express yourself."""
    
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()
        
        # Check for exit command
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("\n👋 Goodbye!")
            break
        
        # Check for switch command
        if user_input.lower().startswith('switch:'):
            provider = user_input.split(':', 1)[1].strip()
            try:
                llm.switch_provider(provider)
                print(f"✅ Switched to {provider}")
                print(f"Model: {llm.get_model_config().get('model')}")
                continue
            except Exception as e:
                print(f"❌ Error switching provider: {e}")
                continue
        
        # Skip empty inputs
        if not user_input:
            continue
        
        # Get response
        print("\nAI: ", end="", flush=True)
        
        start_time = time.time()
        
        # Use streaming for a better user experience
        full_response = ""
        for chunk in llm.stream_response(user_input, system_prompt=system_prompt):
            print(chunk, end="", flush=True)
            full_response += chunk
        
        end_time = time.time()
        
        # Print timing information
        print(f"\n[Response time: {end_time - start_time:.2f}s, {len(full_response)} chars]")

if __name__ == "__main__":
    simple_chat()