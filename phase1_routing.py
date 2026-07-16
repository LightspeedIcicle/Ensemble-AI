import asyncio
import anthropic
from google import genai
from dotenv import load_dotenv
import os

# Load API keys from .env file
load_dotenv()

# Initialize clients
anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def query_claude(prompt):
    response = anthropic_client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

def query_gemini(prompt):
    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text

async def query_both(prompt):
    print(f"\nPrompt: {prompt}")
    print("-" * 60)
    
    # Run both queries concurrently
    loop = asyncio.get_event_loop()
    claude_task = loop.run_in_executor(None, query_claude, prompt)
    gemini_task = loop.run_in_executor(None, query_gemini, prompt)
    
    claude_response, gemini_response = await asyncio.gather(claude_task, gemini_task)
    
    print(f"\nCLAUDE:\n{claude_response}")
    print("-" * 60)
    print(f"\nGEMINI:\n{gemini_response}")
    print("-" * 60)

# Entry point
if __name__ == "__main__":
    test_prompt = "Explain what a neural network is in 3 sentences."
    asyncio.run(query_both(test_prompt))
