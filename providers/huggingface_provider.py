"""
Hugging Face Provider
Documentation: https://huggingface.co/docs/api-inference/index
API Reference: https://huggingface.co/docs/api-inference/detailed_parameters

Supported Models:
- microsoft/DialoGPT-large (Conversational model)
- facebook/blenderbot-400M-distill (Conversational model)
- gpt2 (Text generation)
- gpt2-xl (Larger GPT-2)
- EleutherAI/gpt-j-6B (6B parameter model)
- EleutherAI/gpt-neox-20b (20B parameter model)
- bigscience/bloom (176B parameter model)
- google/flan-t5-xxl (Text-to-text model)
- google/flan-ul2 (Instruction-tuned model)
- stabilityai/stablelm-tuned-alpha-7b (StableLM 7B)

Default Parameters:
- temperature: 0.7 (0.0-1.0, higher = more creative)
- max_tokens: 2000 (1-4096 for most models)
- top_p: 1.0 (0.0-1.0, nucleus sampling)
- top_k: 50 (1-100, top-k sampling)
- repetition_penalty: 1.0 (1.0-2.0, higher = less repetition)
- do_sample: true (Whether to use sampling)
- wait_for_model: true (Wait if model is loading)

Note: Hugging Face hosts thousands of models with different capabilities and parameters
"""

import os
import requests
from typing import Dict, Any

class HuggingFaceProvider:
    def __init__(self):
        self.api_key = os.getenv("HUGGINGFACE_API_KEY")
        self.base_url = "https://api-inference.huggingface.co/models/"
    
    def generate_response(self, 
                         system_prompt: str, 
                         user_prompt: str, 
                         model_config: Dict[str, Any]) -> str:
        """Generate response using Hugging Face API"""
        try:
            model = model_config.get("model", "microsoft/DialoGPT-large")
            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            # Format prompt based on model type
            if "gpt" in model.lower() or "dialo" in model.lower():
                # For GPT-style models
                formatted_prompt = f"{system_prompt}\n\nUser: {user_prompt}\nAssistant:"
            elif "t5" in model.lower() or "flan" in model.lower():
                # For T5-style models
                formatted_prompt = f"system: {system_prompt} user: {user_prompt}"
            else:
                # Default format
                formatted_prompt = f"{system_prompt}\n{user_prompt}"
            
            payload = {
                "inputs": formatted_prompt,
                "parameters": {
                    "temperature": model_config.get("temperature", 0.7),
                    "max_new_tokens": model_config.get("max_tokens", 2000),
                    "top_p": model_config.get("top_p", 1.0),
                    "top_k": model_config.get("top_k", 50),
                    "repetition_penalty": model_config.get("repetition_penalty", 1.0),
                    "do_sample": model_config.get("do_sample", True),
                    "return_full_text": False
                },
                "options": {
                    "wait_for_model": model_config.get("wait_for_model", True)
                }
            }
            
            response = requests.post(f"{self.base_url}{model}", headers=headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            
            # Handle different response formats
            if isinstance(result, list) and len(result) > 0:
                if "generated_text" in result[0]:
                    return result[0]["generated_text"]
                else:
                    return str(result[0])
            elif isinstance(result, dict) and "generated_text" in result:
                return result["generated_text"]
            else:
                return str(result)
                
        except Exception as e:
            return f"Error with Hugging Face: {str(e)}"
    
    def stream_response(self, 
                       system_prompt: str, 
                       user_prompt: str, 
                       model_config: Dict[str, Any]):
        """Stream response using Hugging Face API (Note: True streaming not supported by all models)"""
        try:
            # Hugging Face doesn't support true streaming for most models
            # We'll simulate it by returning chunks of the full response
            response = self.generate_response(system_prompt, user_prompt, model_config)
            
            # Simulate streaming by yielding chunks of text
            chunk_size = 10  # Characters per chunk
            for i in range(0, len(response), chunk_size):
                yield response[i:i+chunk_size]
                
        except Exception as e:
            yield f"Error with Hugging Face streaming: {str(e)}"

# Test this provider individually
if __name__ == "__main__":
    import sys
    import time
    from dotenv import load_dotenv
    
    load_dotenv()
    
    print("🤖 Testing Hugging Face Provider...")
    print("=" * 50)
    
    # Test configuration
    test_config = {
        "model": "microsoft/DialoGPT-large",
        "temperature": 0.7,
        "max_tokens": 100,
        "top_p": 1.0,
        "top_k": 50,
        "wait_for_model": True
    }
    
    system_prompt = "You are a helpful AI assistant. Be concise and informative."
    user_prompt = "What are the advantages of using Hugging Face for AI development?"
    
    print(f"Model: {test_config['model']}")
    print(f"System: {system_prompt}")
    print(f"User: {user_prompt}")
    print("-" * 50)
    
    try:
        provider = HuggingFaceProvider()
        
        # Test regular response
        start_time = time.time()
        response = provider.generate_response(system_prompt, user_prompt, test_config)
        end_time = time.time()
        
        print(f"Response: {response}")
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        print(f"Characters: {len(response)}")
        
        # Test simulated streaming response
        print("\n🔄 Testing streaming (simulated)...")
        start_time = time.time()
        full_response = ""
        for chunk in provider.stream_response(system_prompt, user_prompt, test_config):
            print(chunk, end="", flush=True)
            full_response += chunk
            time.sleep(0.05)  # Add small delay to simulate streaming
        end_time = time.time()
        
        print(f"\nStreaming time: {end_time - start_time:.2f} seconds")
        print("✅ Hugging Face test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("Make sure your HUGGINGFACE_API_KEY is set in the .env file")
        sys.exit(1)