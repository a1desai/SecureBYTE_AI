"""
Google Provider
Documentation: https://ai.google.dev/docs
API Reference: https://ai.google.dev/api/rest

Supported Models:
- gemini-1.5-pro-latest (Latest Gemini 1.5 Pro)
- gemini-1.5-pro (Gemini 1.5 Pro)
- gemini-1.5-flash-latest (Latest Gemini 1.5 Flash)
- gemini-1.5-flash (Gemini 1.5 Flash - faster)
- gemini-pro (Gemini Pro - text only)
- gemini-pro-vision (Gemini Pro with vision)
- gemini-1.0-pro (Gemini 1.0 Pro)
- gemini-1.0-pro-001 (Gemini 1.0 Pro snapshot)
- gemini-1.0-pro-latest (Latest Gemini 1.0 Pro)
- text-bison-001 (PaLM 2 for text)
- chat-bison-001 (PaLM 2 for chat)

Default Parameters:
- temperature: 0.7 (0.0-2.0, higher = more creative)
- max_output_tokens: 2000 (1-8192)
- top_p: 1.0 (0.0-1.0, nucleus sampling)
- top_k: 40 (1-40, top-k sampling)
- candidate_count: 1 (number of responses)
"""

import os
import google.generativeai as genai
from typing import Dict, Any

class GoogleProvider:
    def __init__(self):
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    
    def generate_response(self, 
                         system_prompt: str, 
                         user_prompt: str, 
                         model_config: Dict[str, Any]) -> str:
        """Generate response using Google Gemini API"""
        try:
            model = genai.GenerativeModel(
                model_name=model_config.get("model", "gemini-pro"),
                generation_config=genai.types.GenerationConfig(
                    temperature=model_config.get("temperature", 0.7),
                    max_output_tokens=model_config.get("max_output_tokens", 2000),
                    top_p=model_config.get("top_p", 1.0),
                    top_k=model_config.get("top_k", 40),
                    candidate_count=model_config.get("candidate_count", 1)
                )
            )
            
            # Combine system and user prompts for Gemini
            full_prompt = f"System: {system_prompt}\n\nUser: {user_prompt}"
            response = model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            return f"Error with Google Gemini: {str(e)}"
    
    def stream_response(self, 
                       system_prompt: str, 
                       user_prompt: str, 
                       model_config: Dict[str, Any]):
        """Stream response using Google Gemini API"""
        try:
            model = genai.GenerativeModel(
                model_name=model_config.get("model", "gemini-pro"),
                generation_config=genai.types.GenerationConfig(
                    temperature=model_config.get("temperature", 0.7),
                    max_output_tokens=model_config.get("max_output_tokens", 2000),
                    top_p=model_config.get("top_p", 1.0),
                    top_k=model_config.get("top_k", 40),
                    candidate_count=model_config.get("candidate_count", 1)
                )
            )
            
            full_prompt = f"System: {system_prompt}\n\nUser: {user_prompt}"
            response = model.generate_content(full_prompt, stream=True)
            
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            yield f"Error with Google Gemini streaming: {str(e)}"

# Test this provider individually
if __name__ == "__main__":
    import sys
    import time
    from dotenv import load_dotenv
    
    load_dotenv()
    
    print("🤖 Testing Google Provider...")
    print("=" * 50)
    
    # Test configuration
    test_config = {
        "model": "gemini-pro",
        "temperature": 0.7,
        "max_output_tokens": 100,
        "top_p": 1.0,
        "top_k": 40,
        "candidate_count": 1
    }
    
    system_prompt = "You are Gemini, Google's AI assistant. Be helpful and informative."
    user_prompt = "What are the key advantages of multimodal AI models?"
    
    print(f"Model: {test_config['model']}")
    print(f"System: {system_prompt}")
    print(f"User: {user_prompt}")
    print("-" * 50)
    
    try:
        provider = GoogleProvider()
        
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
        print("✅ Google test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("Make sure your GOOGLE_API_KEY is set in the .env file")
        sys.exit(1)