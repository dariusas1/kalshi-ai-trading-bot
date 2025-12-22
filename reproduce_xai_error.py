import asyncio
import os
import sys

# Add src to path
sys.path.append(os.getcwd())

from src.clients.xai_client import XAIClient
from src.config.settings import settings

async def main():

    print("Initializing XAIClient...")
    client = XAIClient()
    
    try:
        print("Testing _make_completion_request with CORRECT dictionary format (Verifying FIX)...")
        # Initialize client.client usually done in __init__
        
        analysis_prompt = "Test prompt"
        messages = [{"role": "user", "content": analysis_prompt}]
        
        # Calling private method to verify the fix
        res, cost = await client._make_completion_request(
            messages, 
            max_tokens=10, 
            temperature=0.1
        )
        print("Result:", res)
        
        # Test 2: Try to mimic what might be happening
        # Maybe accessing .get on the client object?
        # print("Trying access .get on client.client...")
        # try:
        #     client.client.get("foo")
        # except Exception as e:
        #     print(f"Caught expected error: {e!r}")

    except Exception as e:
        print(f"Error caught: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
