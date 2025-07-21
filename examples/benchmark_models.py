"""
Benchmark Models Example
Compare performance of different LLM providers and models
"""

import os
import sys
import time
import json
from dotenv import load_dotenv

# Add parent directory to path to import from main
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import LLMManager

def benchmark_models(providers=None, test_prompts=None):
    """Run benchmarks on specified providers with test prompts"""
    load_dotenv()
    
    llm = LLMManager()
    
    # Default test prompts if none provided
    if not test_prompts:
        test_prompts = [
            "Explain the concept of machine learning in 3 sentences.",
            "What are the key differences between Python and JavaScript?",
            "Write a short poem about technology and nature.",
            "Summarize the impact of artificial intelligence on healthcare.",
            "How would you explain quantum computing to a 10-year-old?"
        ]
    
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
    
    print(f"🔍 Found API keys for: {', '.join(available_providers)}")
    
    # Use specified providers if provided, otherwise use all available
    benchmark_providers = providers if providers else available_providers
    benchmark_providers = [p for p in benchmark_providers if p in available_providers]
    
    if not benchmark_providers:
        print("❌ No valid providers to benchmark.")
        return
    
    print(f"🚀 Benchmarking providers: {', '.join(benchmark_providers)}")
    print("=" * 60)
    
    # Run the benchmark
    results = llm.compare_providers(benchmark_providers, test_prompts)
    
    # Save results with timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"benchmark_results_{timestamp}.json"
    llm.save_benchmark_results(results, filename)
    
    # Print summary table
    print("\n📊 Benchmark Summary:")
    print("-" * 60)
    print(f"{'Provider':<15} {'Model':<25} {'Avg Time':<10} {'Avg Length':<12} {'Success'}")
    print("-" * 60)
    
    for provider, data in results["providers"].items():
        if "error" in data:
            print(f"{provider:<15} {'ERROR':<25} {'-':<10} {'-':<12} ❌")
            continue
            
        model = data.get("model", "unknown")
        avg_time = data.get("average_time", 0)
        avg_chars = data.get("average_characters", 0)
        
        # Calculate success rate
        tests = data.get("tests", [])
        if tests:
            success_count = sum(1 for t in tests if t.get("success", False))
            success_rate = f"{success_count}/{len(tests)}"
        else:
            success_rate = "0/0"
            
        print(f"{provider:<15} {model[:25]:<25} {avg_time:<10.2f}s {avg_chars:<12} {success_rate}")
    
    print("-" * 60)
    print(f"💾 Full results saved to {filename}")

if __name__ == "__main__":
    # Example usage with specific providers
    # Uncomment and modify as needed
    # benchmark_models(providers=["openai", "anthropic", "google"])
    
    # Default: benchmark all available providers
    benchmark_models()