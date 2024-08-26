import requests

api_key = "164944310amsh42317543591df12p1cdfb4jsn361155ee8346"
host = "chatgpt-42.p.rapidapi.com"

def query_chatgpt(query, temperature=0.9, top_k=5, top_p=0.9, max_tokens=256, web_access=False):
    url = "https://chatgpt-42.p.rapidapi.com/conversationgpt4-2"
    
    payload = {
        "messages": [
            {
                "role": "user",
                "content": query
            }
        ],
        "system_prompt": "",
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "web_access": web_access
    }
    
    headers = {
        "x-rapidapi-key": api_key,
        "x-rapidapi-host": host,
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # Raise an error for HTTP errors
        response_json = response.json()
        # Extract and return the result from the response
        result = response_json.get('result', 'No result found')
        return result.strip()  # Clean up any extra whitespace
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None

# # Example usage
# query = "Famous places in Jammu"

# result = query_chatgpt(query)
# print(result)
