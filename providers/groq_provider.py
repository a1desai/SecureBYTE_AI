"""
Groq Provider
Documentation: https://console.groq.com/docs/models
API Reference: https://console.groq.com/docs/quickstart

Supported Models:
- mixtral-8x7b-32768 (Mixtral with 32k context)
- llama2-70b-4096 (LLaMA 2 70B with 4k context)
- gemma-7b-it (Gemma 7B instruction-tuned)
- claude-3-opus-20240229 (Claude 3 Opus)
- claude-3-sonnet-20240229 (Claude 3 Sonnet)
- claude-3-haiku-20240307 (Claude 3 Haiku)

Default Parameters:
- temperature: 0.7 (0.0-1.0, higher = more creative)
- max_tokens: 2000 (1-32768 depending on model)
- top_p: 1.0 (0.0-1.0, nucleus sampling)
- top_k: 40 (1-100, top-k sampling)
- frequency_penalty: 0.0 (-2.0 to 2.0)
- presence_penalty: 0.0 (-2.0 to 2.0)

Note: Groq is known for extremely fast inference speeds
"""

import os
from groq import Groq
from typing import Dict, Any

class GroqProvider:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    def generate_response(self, 
                         system_prompt: str, 
                         user_prompt: str, 
                         model_config: Dict[str, Any]) -> str:
        """Generate response using Groq API"""
        try:
            response = self.client.chat.completions.create(
                model=model_config.get("model", "mixtral-8x7b-32768"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=model_config.get("temperature", 0.7),
                max_tokens=model_config.get("max_tokens", 2000),
                top_p=model_config.get("top_p", 1.0)
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error with Groq: {str(e)}"
    
    def stream_response(self, 
                       system_prompt: str, 
                       user_prompt: str, 
                       model_config: Dict[str, Any]):
        """Stream response using Groq API"""
        try:
            stream = self.client.chat.completions.create(
                model=model_config.get("model", "mixtral-8x7b-32768"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=model_config.get("temperature", 0.7),
                max_tokens=model_config.get("max_tokens", 2000),
                top_p=model_config.get("top_p", 1.0),
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            yield f"Error with Groq streaming: {str(e)}"

# Test this provider individually
if __name__ == "__main__":
    import sys
    import time
    from dotenv import load_dotenv
    
    load_dotenv()
    
    print("🤖 Testing Groq Provider...")
    print("=" * 50)
    
    # Test configuration
    test_config = {
        "model": "mixtral-8x7b-32768",
        "temperature": 0.7,
        "max_tokens": 100,
        "top_p": 1.0
    }
    
    system_prompt = "You are a helpful AI assistant powered by Groq. Be concise and informative."
    user_prompt = "What makes Groq's inference speed so fast compared to other providers?"
    
    print(f"Model: {test_config['model']}")
    print(f"System: {system_prompt}")
    print(f"User: {user_prompt}")
    print("-" * 50)
    
    try:
        provider = GroqProvider()
        
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
        print("✅ Groq test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("Make sure your GROQ_API_KEY is set in the .env file")
        sys.exit(1)