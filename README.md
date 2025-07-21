# Multi-LLM Provider Template System

A comprehensive Python template system for interacting with multiple Large Language Model (LLM) providers. Switch between OpenAI, Anthropic, Google, Cohere, Mistral, Groq, Together AI, Replicate, and Hugging Face with ease.

## 🚀 Quick Start

1. **Clone and Setup**
   ```bash
   git clone <your-repo-url>
   cd <your-repo-name>
   pip install -r requirements.txt
   ```

2. **Configure API Keys**
   ```bash
   cp .env.example .env
   # Edit .env with your actual API keys
   ```

3. **Customize Settings**
   - Edit `config.py` to choose your provider and model
   - Modify system and user prompts
   - Adjust model parameters (temperature, max_tokens, etc.)

4. **Run Examples**
   ```bash
   python main.py
   ```

## 📁 Project Structure

```
├── .env.example          # Template for API keys
├── config.py            # Main configuration file
├── main.py              # Example usage script
├── requirements.txt     # Python dependencies
└── providers/           # Provider implementations
    ├── openai_provider.py
    ├── anthropic_provider.py
    ├── google_provider.py
    ├── cohere_provider.py
    ├── mistral_provider.py
    ├── groq_provider.py
    ├── together_provider.py
    ├── replicate_provider.py
    └── huggingface_provider.py
```

## ⚙️ Configuration

### Switching Providers
Edit `config.py` and change the `CURRENT_PROVIDER` variable:
```python
CURRENT_PROVIDER = "openai"  # or "anthropic", "google", etc.
```

### Customizing Models
Each provider has its own model configuration in the `MODELS` dictionary:
```python
MODELS = {
    "openai": {
        "model": "gpt-4",           # Change model here
        "temperature": 0.7,         # Creativity (0.0-2.0)
        "max_tokens": 2000,         # Response length
        "top_p": 1.0,              # Nucleus sampling
    },
    # ... other providers
}
```

### Customizing Prompts
Modify the prompts in `config.py`:
```python
# System prompt - Sets AI behavior
SYSTEM_PROMPT = """Your custom system prompt here..."""

# Default user prompt
DEFAULT_USER_PROMPT = "Your default question here..."
```

## 🔑 API Keys Setup

1. Copy `.env.example` to `.env`
2. Add your API keys for the providers you want to use
3. Keep unused keys commented out or remove them

**Important**: Never commit your `.env` file to version control!

## 🎯 Usage Examples

### Basic Usage
```python
from main import LLMManager

# Initialize with current config
llm = LLMManager()

# Generate response
response = llm.generate_response("What is machine learning?")
print(response)
```

### Custom Prompts
```python
# Custom system and user prompts
response = llm.generate_response(
    user_prompt="Explain quantum computing",
    system_prompt="You are a physics professor explaining complex topics simply."
)
```

### Streaming Responses
```python
# Enable streaming in config.py first
for chunk in llm.stream_response("Tell me a story"):
    print(chunk, end="", flush=True)
```

### Switch Providers Programmatically
```python
# Change provider at runtime
llm.switch_provider("anthropic")
response = llm.generate_response("Hello from Claude!")
```

## 🚦 How to Use

1. **Set Up API Keys**
   - Copy `.env.example` to `.env` and fill in your API keys for the providers you want to use.
   - You can use as many or as few providers as you like—just leave unused keys blank.

2. **Choose Provider, Model, and Parameters**
   - Open `config.py`.
   - Set `CURRENT_PROVIDER` to the provider you want (e.g., `openai`, `anthropic`, etc.).
   - In the `MODELS` dictionary, pick your model and adjust parameters (temperature, max tokens, etc.).
   - You can copy model names directly from the provider’s documentation.

3. **Run a Single Test**
   - Run `python main.py` to send a test prompt to your selected provider/model and see the response.

4. **Interactive Console Mode**
   - Run `python main.py interactive` for a REPL-style console.
   - Type prompts and get responses instantly.
   - Use commands like `/switch <provider>`, `/providers`, `/config`, `/benchmark`, and `/quit`.

5. **Benchmarking**
   - Run `python main.py benchmark` to test all configured providers on a set of prompts.
   - Results are saved as a JSON file for later review.
   - You can specify which providers to benchmark by passing them as a comma-separated list: `python main.py benchmark openai,anthropic`.

6. **Test a Provider Directly**
   - Run any provider file directly (e.g., `python providers/openai_provider.py`) to send a test request and log the result.

7. **Switch Providers and Models**
   - Change `CURRENT_PROVIDER` and model name in `config.py`, or use `/switch` in interactive mode.
   - Edit parameters in the `MODELS` dictionary for fine-tuning.

8. **Add New Models**
   - Copy the model name from the provider’s docs and paste it into the `model` field in `config.py` under the correct provider.
   - Adjust parameters as needed.

9. **Handle Errors**
   - If you see an error like `Error with OpenAI: ...`, check your API key, model name, and parameters.
   - Make sure your API key is valid and you have access to the model.
   - Typos or unsupported models will result in clear error messages.

**Main Files to Edit/Use:**
- `.env` — API keys
- `config.py` — Provider, model, and parameter selection
- `main.py` — Main entry point for testing, benchmarking, and interactive mode
- `providers/` — Individual provider implementations (can be run directly)

## 🔧 Supported Providers & Models

| Provider | Popular Models | Notes |
|----------|----------------|-------|
| **OpenAI** | gpt-4, gpt-4-turbo, gpt-3.5-turbo | Most versatile |
| **Anthropic** | claude-3-opus, claude-3-sonnet, claude-3-haiku | Great for analysis |
| **Google** | gemini-pro, gemini-pro-vision | Multimodal support |
| **Cohere** | command, command-light | Enterprise focused |
| **Mistral** | mistral-large, mistral-medium | European alternative |
| **Groq** | mixtral-8x7b, llama2-70b | Ultra-fast inference |
| **Together AI** | Various open-source models | Cost-effective |
| **Replicate** | Meta Llama, Mixtral | Easy deployment |
| **Hugging Face** | Various open models | Research friendly |

## 🛠️ Advanced Configuration

### Timeout and Retries
```python
REQUEST_TIMEOUT = 30    # API timeout in seconds
MAX_RETRIES = 3        # Retry failed requests
```

### Streaming
```python
ENABLE_STREAMING = True  # Enable streaming where supported
```

### Provider-Specific Settings
Each provider may have unique parameters. Check the individual provider files in the `providers/` directory for specific options.

## 🚨 Error Handling

The system includes built-in error handling:
- API key validation
- Network timeout handling
- Rate limit management
- Graceful fallbacks

## 📝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add new providers in the `providers/` directory
4. Update `config.py` with new model configurations
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Support

- Check the provider documentation for API-specific issues
- Review the error messages for troubleshooting hints
- Ensure your API keys are valid and have sufficient credits

---

**Happy prompting! 🎉**