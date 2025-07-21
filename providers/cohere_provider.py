"""
Cohere Provider
Documentation: https://docs.cohere.com/docs/models
API Reference: https://docs.cohere.com/reference/about

Supported Models:
- command (Latest Command model)
- command-light (Faster, more efficient version)
- command-nightly (Experimental version with latest improvements)
- command-r (Enhanced reasoning capabilities)
- command-r-plus (Most capable reasoning model)
- embed-english-v3.0 (Embedding model)
- embed-multilingual-v3.0 (Multilingual embedding model)

Default Parameters:
- temperature: 0.7 (0.0-5.0, higher = more creative)
- max_tokens: 2000 (1-4096)
- p: 1.0 (0.0-1.0, nucleus sampling)
- k: 0 (0-500, top-k sampling)
- frequency_penalty: 0.0 (0.0-1.0)
- presence_penalty: 0.0 (0.0-1.0)
"""

import os
import cohere
from typing import Dict, Any

class CohereProvider:
    def __init__(self):
        self.client = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))
    
    def generate_response(self, 
                         system_prompt: str, 
                         user_prompt: str, 
                         model_config: Dict[str, Any]) -> str:
        """Generate response using Cohere API"""
        try:
            # Combine system and user prompts for Cohere
            full_prompt = f"{system_prompt}\n\nUser: {user_prompt}\nAssistant:"
            
            response = self.client.generate(
                model=model_config.get("model", "command"),
                prompt=full_prompt,
                temperature=model_config.get("temperature", 0.7),
                max_tokens=model_config.get("max_tokens", 2000),
                p=model_config.get("p", 1.0),
                k=model_config.get("k", 0)
            )
            return response.generations[0].text.strip()
        except Exception as e:
            return f"Error with Cohere: {str(e)}"
    
    def stream_response(self, 
                       system_prompt: str, 
                       user_prompt: str, 
                       model_config: Dict[str, Any]):
        """Stream response using Cohere API"""
        try:
            full_prompt = f"{system_prompt}\n\nUser: {user_prompt}\nAssistant:"
            
            response = self.client.generate(
                model=model_config.get("model", "command"),
                prompt=full_prompt,
                temperature=model_config.get("temperature", 0.7),
                max_tokens=model_config.get("max_tokens", 2000),
                p=model_config.get("p", 1.0),
                k=model_config.get("k", 0),
                stream=True
            )
            
            for token in response:
                if hasattr(token, 'text'):
                    yield token.text
        except Exception as e:
            yield f"Error with Cohere streaming: {str(e)}"

# Test this provider individually
if __name__ == "__main__":
    import sys
    import time
    from dotenv import load_dotenv
    
    load_dotenv()
    
    print("🤖 Testing Cohere Provider...")
    print("=" * 50)
    
    # Test configuration
    test_config = {
        "model": "command",
        "temperature": 0.7,
        "max_tokens": 100,
        "p": 1.0,
        "k": 0
    }
    
    system_prompt = "You are an AI assistant powered by Cohere. Be helpful and concise."
    user_prompt = "What makes large language models effective for text generation?"
    
    print(f"Model: {test_config['model']}")
    print(f"System: {system_prompt}")
    print(f"User: {user_prompt}")
    print("-" * 50)
    
    try:
        provider = CohereProvider()
        
        # Test regular response
        start_time = time.time()
        response = provider.generate_response(system_prompt, user_prompt, test_config)
        end_time = time.time()
        
        print(f"Response: {response}")
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        print(f"Characters: {len(response)}")
        
        # Test streaming response
        print("\n🔄 Testing streaming...")
        start_time = time.time()
        full_response = ""
        for chunk in provider.stream_response(system_prompt, user_prompt, test_config):
            print(chunk, end="", flush=True)
            full_response += chunk
        end_time = time.time()
        
        print(f"\nStreaming time: {end_time - start_time:.2f} seconds")
        print("✅ Cohere test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("Make sure your COHERE_API_KEY is set in the .env file")
        sys.exit(1)