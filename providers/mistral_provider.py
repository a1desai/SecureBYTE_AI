"""
Mistral AI Provider
Documentation: https://docs.mistral.ai/
API Reference: https://docs.mistral.ai/api/

Supported Models:
- mistral-large-latest (Most capable model)
- mistral-large-2402 (Specific version of Large)
- mistral-medium-latest (Balanced performance and cost)
- mistral-medium-2312 (Specific version of Medium)
- mistral-small-latest (Fast and cost-effective)
- mistral-small-2402 (Specific version of Small)
- open-mistral-7b (Open-source 7B model)
- open-mixtral-8x7b (Open-source mixture of experts)

Default Parameters:
- temperature: 0.7 (0.0-1.0, higher = more creative)
- max_tokens: 2000 (1-32768 depending on model)
- top_p: 1.0 (0.0-1.0, nucleus sampling)
- safe_prompt: false (Enable/disable content filtering)
- random_seed: null (Integer for deterministic outputs)
"""

import os
from mistralai import Mistral, UserMessage, SystemMessage
from typing import Dict, Any

class MistralProvider:
    def __init__(self):
        self.client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
    
    def generate_response(self, 
                         system_prompt: str, 
                         user_prompt: str, 
                         model_config: Dict[str, Any]) -> str:
        """Generate response using Mistral API"""
        try:
            messages = [
                SystemMessage(content=system_prompt),
                UserMessage(content=user_prompt)
            ]
            
            response = self.client.chat.complete(
                model=model_config.get("model", "mistral-large-latest"),
                messages=messages,
                temperature=model_config.get("temperature", 0.7),
                max_tokens=model_config.get("max_tokens", 2000),
                top_p=model_config.get("top_p", 1.0)
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error with Mistral: {str(e)}"
    
    def stream_response(self, 
                       system_prompt: str, 
                       user_prompt: str, 
                       model_config: Dict[str, Any]):
        """Stream response using Mistral API"""
        try:
            messages = [
                SystemMessage(content=system_prompt),
                UserMessage(content=user_prompt)
            ]
            
            response = self.client.chat.stream(
                model=model_config.get("model", "mistral-large-latest"),
                messages=messages,
                temperature=model_config.get("temperature", 0.7),
                max_tokens=model_config.get("max_tokens", 2000),
                top_p=model_config.get("top_p", 1.0)
            )
            
            for chunk in response:
                if chunk.data.choices[0].delta.content is not None:
                    yield chunk.data.choices[0].delta.content
        except Exception as e:
            yield f"Error with Mistral streaming: {str(e)}"

# Test this provider individually
if __name__ == "__main__":
    import sys
    import time
    from dotenv import load_dotenv
    
    load_dotenv()
    
    print("🤖 Testing Mistral Provider...")
    print("=" * 50)
    
    # Test configuration
    test_config = {
        "model": "mistral-small-latest",  # Using small model for faster testing
        "temperature": 0.7,
        "max_tokens": 100,
        "top_p": 1.0
    }
    
    system_prompt = "You are Mistral AI, a helpful and accurate assistant. Be concise."
    user_prompt = "What are the key innovations in the Mixtral architecture?"
    
    print(f"Model: {test_config['model']}")
    print(f"System: {system_prompt}")
    print(f"User: {user_prompt}")
    print("-" * 50)
    
    try:
        provider = MistralProvider()
        
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
        print("✅ Mistral test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("Make sure your MISTRAL_API_KEY is set in the .env file")
        sys.exit(1)