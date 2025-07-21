"""
Anthropic Provider
Documentation: https://docs.anthropic.com/claude/docs/models-overview
API Reference: https://docs.anthropic.com/claude/reference/

Supported Models:
- claude-3-opus-20240229 (Most capable, best for complex tasks)
- claude-3-sonnet-20240229 (Balanced performance and speed)
- claude-3-haiku-20240307 (Fastest, good for simple tasks)
- claude-2.1 (Previous generation, 200k context)
- claude-2.0 (Previous generation, 100k context)
- claude-instant-1.2 (Fast and affordable)

Default Parameters:
- temperature: 0.7 (0.0-1.0, higher = more creative)
- max_tokens: 2000 (1-4096 for most models)
- top_p: 1.0 (0.0-1.0, nucleus sampling)
- top_k: 40 (1-500, top-k sampling)
"""

import os
import anthropic
from typing import Dict, Any

class AnthropicProvider:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    def generate_response(self, 
                         system_prompt: str, 
                         user_prompt: str, 
                         model_config: Dict[str, Any]) -> str:
        """Generate response using Anthropic API"""
        try:
            response = self.client.messages.create(
                model=model_config.get("model", "claude-3-sonnet-20240229"),
                max_tokens=model_config.get("max_tokens", 2000),
                temperature=model_config.get("temperature", 0.7),
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            return response.content[0].text
        except Exception as e:
            return f"Error with Anthropic: {str(e)}"
    
    def stream_response(self, 
                       system_prompt: str, 
                       user_prompt: str, 
                       model_config: Dict[str, Any]):
        """Stream response using Anthropic API"""
        try:
            with self.client.messages.stream(
                model=model_config.get("model", "claude-3-sonnet-20240229"),
                max_tokens=model_config.get("max_tokens", 2000),
                temperature=model_config.get("temperature", 0.7),
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            ) as stream:
                for text in stream.text_stream:
                    yield text
        except Exception as e:
            yield f"Error with Anthropic streaming: {str(e)}"

# Test this provider individually
if __name__ == "__main__":
    import sys
    import time
    from dotenv import load_dotenv
    
    load_dotenv()
    
    print("🤖 Testing Anthropic Provider...")
    print("=" * 50)
    
    # Test configuration
    test_config = {
        "model": "claude-3-haiku-20240307",
        "temperature": 0.7,
        "max_tokens": 100
    }
    
    system_prompt = "You are Claude, an AI assistant created by Anthropic. Be helpful and concise."
    user_prompt = "What makes you different from other AI assistants?"
    
    print(f"Model: {test_config['model']}")
    print(f"System: {system_prompt}")
    print(f"User: {user_prompt}")
    print("-" * 50)
    
    try:
        provider = AnthropicProvider()
        
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
        print("✅ Anthropic test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("Make sure your ANTHROPIC_API_KEY is set in the .env file")
        sys.exit(1)