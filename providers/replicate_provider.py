"""
Replicate Provider
Documentation: https://replicate.com/collections/language-models
API Reference: https://replicate.com/docs/reference/http

Supported Models:
- meta/llama-2-70b-chat (LLaMA 2 70B)
- meta/llama-2-13b-chat (LLaMA 2 13B)
- meta/llama-2-7b-chat (LLaMA 2 7B)
- mistralai/mixtral-8x7b-instruct-v0.1 (Mixtral 8x7B)
- mistralai/mistral-7b-instruct-v0.2 (Mistral 7B)
- anthropic/claude-3-sonnet-20240229 (Claude 3 Sonnet)
- anthropic/claude-3-haiku-20240307 (Claude 3 Haiku)
- stability-ai/stable-diffusion-xl (Image generation)
- midjourney/midjourney (Image generation)
- cjwbw/anything-v4.0 (Image generation)

Default Parameters:
- temperature: 0.7 (0.0-1.0, higher = more creative)
- max_tokens: 2000 (1-4096 for most models)
- top_p: 1.0 (0.0-1.0, nucleus sampling)
- top_k: 50 (1-100, top-k sampling)
- stop_sequences: [] (List of strings to stop generation)
- repetition_penalty: 1.0 (1.0-2.0, higher = less repetition)

Note: Replicate hosts a wide variety of models including text, image, audio, and video generation
"""

import os
import replicate
from typing import Dict, Any

class ReplicateProvider:
    def __init__(self):
        os.environ["REPLICATE_API_TOKEN"] = os.getenv("REPLICATE_API_TOKEN", "")
    
    def generate_response(self, 
                         system_prompt: str, 
                         user_prompt: str, 
                         model_config: Dict[str, Any]) -> str:
        """Generate response using Replicate API"""
        try:
            model_name = model_config.get("model", "meta/llama-2-70b-chat")
            
            # Format prompt for LLaMA-style models
            formatted_prompt = f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"
            
            input_data = {
                "prompt": formatted_prompt,
                "temperature": model_config.get("temperature", 0.7),
                "max_tokens": model_config.get("max_tokens", 2000),
                "top_p": model_config.get("top_p", 1.0)
            }
            
            output = replicate.run(model_name, input=input_data)
            
            # Handle different output formats
            if isinstance(output, list):
                return "".join(output)
            elif isinstance(output, str):
                return output
            else:
                return str(output)
                
        except Exception as e:
            return f"Error with Replicate: {str(e)}"
    
    def stream_response(self, 
                       system_prompt: str, 
                       user_prompt: str, 
                       model_config: Dict[str, Any]):
        """Stream response using Replicate API"""
        try:
            model_name = model_config.get("model", "meta/llama-2-70b-chat")
            
            formatted_prompt = f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"
            
            input_data = {
                "prompt": formatted_prompt,
                "temperature": model_config.get("temperature", 0.7),
                "max_tokens": model_config.get("max_tokens", 2000),
                "top_p": model_config.get("top_p", 1.0)
            }
            
            for event in replicate.stream(model_name, input=input_data):
                yield str(event)
                
        except Exception as e:
            yield f"Error with Replicate streaming: {str(e)}"

# Test this provider individually
if __name__ == "__main__":
    import sys
    import time
    from dotenv import load_dotenv
    
    load_dotenv()
    
    print("🤖 Testing Replicate Provider...")
    print("=" * 50)
    
    # Test configuration
    test_config = {
        "model": "mistralai/mistral-7b-instruct-v0.2",  # Using Mistral for testing
        "temperature": 0.7,
        "max_tokens": 100,
        "top_p": 1.0
    }
    
    system_prompt = "You are a helpful AI assistant powered by Replicate. Be concise and informative."
    user_prompt = "What are the benefits of using Replicate for AI model deployment?"
    
    print(f"Model: {test_config['model']}")
    print(f"System: {system_prompt}")
    print(f"User: {user_prompt}")
    print("-" * 50)
    
    try:
        provider = ReplicateProvider()
        
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
        print("✅ Replicate test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("Make sure your REPLICATE_API_TOKEN is set in the .env file")
        sys.exit(1)