"""
Multi-Provider Comparison Example
Compare responses from multiple providers for the same prompt
"""

import os
import sys
import time
from dotenv import load_dotenv

# Add parent directory to path to import from main
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import LLMManager

def compare_providers(prompt, providers=None, system_prompt=None):
    """Compare responses from multiple providers for the same prompt"""
    load_dotenv()
    
    llm = LLMManager()
    
    # Default system prompt if none provided
    if not system_prompt:
        system_prompt = "You are a helpful AI assistant. Provide a concise and accurate response."
    
    # Determine which providers to test
    available_providers = []
    for provider in llm.providers.keys():
        env_var = f"{provider.upper()}_API_KEY"
        if provider == "replicate":
            env_var = "REPLICATE_API_TOKEN"
        if os.getenv(env_var):
            available_providers.append(provider)
    
    if not available_providers:
        print("❌ No API keys found. Please add at least one API key to your .env file.")
        return
    
    # Use specified providers if provided, otherwise use all available
    test_providers = providers if providers else available_providers
    test_providers = [p for p in test_providers if p in available_providers]
    
    if not test_providers:
        print("❌ No valid providers to test.")
        return
    
    print(f"🔍 Testing providers: {', '.join(test_providers)}")
    print("=" * 60)
    print(f"Prompt: {prompt}")
    print(f"System: {system_prompt}")
    print("=" * 60)
    
    results = {}
    
    # Get responses from each provider
    for provider in test_providers:
        print(f"\n📝 Testing {provider}...")
        
        try:
            # Switch to the provider
            llm.switch_provider(provider)
            
            # Get model info
            model = llm.get_model_config().get("model", "unknown")
            print(f"Model: {model}")
            
            # Get response with timing
            start_time = time.time()
            response = llm.generate_response(prompt, system_prompt=system_prompt)
            end_time = time.time()
            
            # Store results
            results[provider] = {
                "model": model,
                "response": response,
                "time": end_time - start_time,
                "characters": len(response)
            }
            
            # Print response
            print(f"Time: {end_time - start_time:.2f}s")
            print(f"Length: {len(response)} characters")
            print("-" * 40)
            print(response)
            print("-" * 40)
            
        except Exception as e:
            print(f"❌ Error with {provider}: {e}")
            results[provider] = {"error": str(e)}
    
    # Print summary
    print("\n📊 Summary:")
    print("-" * 60)
    print(f"{'Provider':<15} {'Model':<25} {'Time':<10} {'Length'}")
    print("-" * 60)
    
    for provider, data in results.items():
        if "error" in data:
            print(f"{provider:<15} {'ERROR':<25} {'-':<10} -")
            continue
            
        model = data.get("model", "unknown")
        response_time = data.get("time", 0)
        chars = data.get("characters", 0)
            
        print(f"{provider:<15} {model[:25]:<25} {response_time:<10.2f}s {chars}")
    
    print("-" * 60)

if __name__ == "__main__":
    # Example usage
    test_prompt = "Explain the concept of neural networks and how they work in 3-4 sentences."
    
    # Uncomment to test specific providers
    # compare_providers(test_prompt, providers=["openai", "anthropic", "google"])
    
    # Default: test all available providers
    compare_providers(test_prompt)