"""
HTTP client module for the LLM inference engine serving layer.

Provides a Python client that sends requests to the running API server.
Used for testing, scripting, and benchmarks. Communicates via HTTP
using httpx — does not import any model or scheduler code.
"""

import httpx

class Client:
    def __init__(self, base_url: str) -> None:
        
        """
        Description:
            Initialize the client with the server's base URL.
        Args:
            base_url (str): The base URL of the running server.
        Raises:
            ValueError: If base_url is empty.
        Returns:
            None
        """
        
        if base_url.strip() == "":
            raise ValueError("Base URL cannot be empty.")
        
        self.base_url = base_url
        
    def generate(self, prompt: str, max_tokens: int) -> dict:
        
        """
        Description:
            Send a generation request to the server.
        Args:
            prompt (str): The input text to generate from.
            max_tokens (int): Maximum number of tokens to generate.
        Returns:
            dict: Server response with request_id, prompt, generated_text, token_count, stop_reason.
        Raises:
            ValueError: If the server returns 400 (bad request).
            RuntimeError: If the server returns any other non-200 status.
        """
        
        url = f"{self.base_url}/generate"
        payload = {"prompt": prompt, "max_tokens": max_tokens}
        response = httpx.post(url, json=payload)
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 400:
            raise ValueError(f"Bad Request: {response.text}")
        else:
            raise RuntimeError(f"Request failed with status code {response.status_code}: {response.text}")
        
    def health(self) -> dict:
        
        """
        Description"
            Check if the server is alive.
        Args:
            None
        Returns:
            dict: Server response with status field (e.g. {"status": "ok"}).
        """
        
        url = f"{self.base_url}/health"
        response = httpx.get(url)
        return response.json()
        

        
        