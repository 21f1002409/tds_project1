# /// script
# dependencies = [
#   "httpx",
# ]
# ///

import httpx
import logging
import json
import os
from typing import Dict, Any

AIPROXY_TOKEN = os.environ.get("AIPROXY_TOKEN")
AIPROXY_URL = "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

def query_llm(task: str, tools: list[Dict[str, Any]]) -> Dict[str, Any]:
    try:
        response = httpx.post(
            AIPROXY_URL,
            headers={
                "Authorization": f"Bearer {AIPROXY_TOKEN}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "user", "content": task},
                ],
                "tools": tools,
                "tool_choice": "auto",
            },
            timeout=10.0
        )
        response_data = response.json()
        logging.info(f"GPT API Response: {json.dumps(response_data, indent=2)}")

        if "choices" not in response_data or not response_data["choices"]:
            return {"error": "Invalid response from API"}

        return response_data["choices"][0]["message"]

    except Exception as e:
        logging.error(f"Error querying GPT: {str(e)}")
        return {"error": f"API request failed: {str(e)}"}