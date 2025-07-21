"""
Together AI Provider
Documentation: https://docs.together.ai/docs/inference-models
API Reference: https://docs.together.ai/reference/chat-completions

Supported Models:
- meta-llama/Llama-2-70b-chat-hf (LLaMA 2 70B)
- meta-llama/Llama-2-13b-chat-hf (LLaMA 2 13B)
- meta-llama/Llama-2-7b-chat-hf (LLaMA 2 7B)
- mistralai/Mixtral-8x7B-Instruct-v0.1 (Mixtral 8x7B)
- mistralai/Mistral-7B-Instruct-v0.2 (Mistral 7B)
- togethercomputer/StripedHyena-Nous-7B (StripedHyena 7B)
- Qwen/Qwen1.5-72B-Chat (Qwen 72B)
- Qwen/Qwen1.5-14B-Chat (Qwen 14B)
- NousResearch/Nous-Hermes-2-Yi-34B (Nous Hermes 34B)
- google/gemma-7b-it (Gemma 7B)

Default Parameters:
- temperature: 0.7 (0.0-1.0, higher = more creative)
- max_tokens: 2000 (1-4096 for most models)
- top_p: 1.0 (0.0-1.0, nucleus sampling)
- top_k: 40 (1-100, top-k sampling)
- repetition_penalty: 1.0 (1.0-2.0, higher = less repetition)
- stop: [] (List of strings to stop generation)

Note: Together AI offers a wide range of open-source models at competitive prices
"""

import os
import requests
from typing import Dict, Any
import json

class TogetherProvider:
    def __init__(self):
        self.api_key = os.getenv("TOGETHER_API_KEY")
        self.base_url = "https://api.together.xyz/v1/chat/completions"
    
    def generate_response(self, 
                         system_prompt: str, 
                         user_prompt: str, 
                         model_config: Dict[str, Any]) -> str:
        """Generate response using Together AI API"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": model_config.get("model", "meta-llama/Llama-2-70b-chat-hf"),
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": model_config.get("temperature", 0.7),
                "max_tokens": model_config.get("max_tokens", 2000),
                "top_p": model_config.get("top_p", 1.0)
            }
            
            response = requests.post(self.base_url, headers=headers, json=data)
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error with Together AI: {str(e)}"
    
    def stream_response(self, 
                       system_prompt: str, 
                       user_prompt: str, 
                       model_config: Dict[str, Any]):
        """Stream response using Together AI API"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": model_config.get("model", "meta-llama/Llama-2-70b-chat-hf"),
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": model_config.get("temperature", 0.7),
                "max_tokens": model_config.get("max_tokens", 2000),
                "top_p": model_config.get("top_p", 1.0),
                "stream": True
            }
            
            response = requests.post(self.base_url, headers=headers, json=data, stream=True)
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data_str = line[6:]
                        if data_str.strip() == '[DONE]':
                            break
                        try:
                            data = json.loads(data_str)
                            if 'choices' in data and len(data['choices']) > 0:
                                delta = data['choices'][0].get('delta', {})
                                if 'content' in delta:
                                    yield delta['content']
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            yield f"Error with Together AI streaming: {str(e)}"

# Test this provider individually
if __name__ == "__main__":
    import sys
    import time
    from dotenv import load_dotenv
    
    load_dotenv()
    
    print("🤖 Testing Together AI Provider...")
    print("=" * 50)
    
    # Test configuration
    test_config = {
        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",  # Using Mixtral for testing
        "temperature": 0.7,
        "max_tokens": 100,
        "top_p": 1.0
    }
    
    system_prompt = "You are a helpful AI assistant powered by Together AI. Be concise and informative."
    user_prompt = "What are the advantages of using open-source models for AI applications?"
    
    print(f"Model: {test_config['model']}")
    print(f"System: {system_prompt}")
    print(f"User: {user_prompt}")
    print("-" * 50)
    
    try:
        provider = TogetherProvider()
        
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
        print("✅ Together AI test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("Make sure your TOGETHER_API_KEY is set in the .env file")
        sys.exit(1)