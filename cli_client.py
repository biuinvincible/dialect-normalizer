#!/usr/bin/env python3
"""
A command-line client to interact with the Vietnamese Dialect Normalizer API.
"""

import requests
import sys

# The URL of the running FastAPI application
API_URL = "http://localhost:8000/normalize"

def normalize_text(text: str) -> str:
    """
    Sends text to the API and returns the normalized version.
    """
    payload = {"text": text}
    try:
        response = requests.post(API_URL, json=payload, timeout=10)
        
        # Check if the request was successful
        response.raise_for_status()
        
        # Parse the JSON response and return the normalized text
        return response.json().get("normalized", "Error: 'normalized' key not found in response.")
        
    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to the API. Is the Docker container running?"
    except requests.exceptions.HTTPError as http_err:
        return f"Error: HTTP error occurred: {http_err} - {response.text}"
    except requests.exceptions.RequestException as err:
        return f"Error: An unexpected error occurred: {err}"

def main():
    """
    Main function to run the CLI client.
    """
    print("Vietnamese Dialect Normalizer CLI Client")
    print("-----------------------------------------")
    print("Enter text to normalize. Type 'quit' or 'exit' to stop.")
    
    while True:
        try:
            # Get user input
            user_input = input("\nInput  > ")
            
            # Check for exit commands
            if user_input.lower() in ["quit", "exit"]:
                print("Exiting client. Goodbye!")
                break
            
            if not user_input.strip():
                continue

            # Get the normalized text
            normalized = normalize_text(user_input)
            
            # Print the result
            print(f"Output > {normalized}")

        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            print("\nExiting client. Goodbye!")
            sys.exit(0)

if __name__ == "__main__":
    main()
