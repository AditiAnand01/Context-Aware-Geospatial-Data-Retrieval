import requests

# ChatGPT API credentials
api_key = "1b93a7ab60msh3322e2c9adfd6f7p15a938jsna2a645cdfbe4"
host = "chatgpt-42.p.rapidapi.com"

# Geocoding API credentials
geocode_url = "https://map-geocoding.p.rapidapi.com/json"
geocode_headers = {
    "x-rapidapi-key": api_key,
    "x-rapidapi-host": "map-geocoding.p.rapidapi.com"
}

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

def get_geocode(place_name):
    querystring = {"address": place_name}
    
    try:
        response = requests.get(geocode_url, headers=geocode_headers, params=querystring)
        response.raise_for_status()
        data = response.json()
        results = data.get('results', [])
        if results:
            location = results[0]['geometry']['location']
            return location['lat'], location['lng']
        else:
            return None, None
    except requests.exceptions.RequestException as e:
        print(f"Geocoding request failed: {e}")
        return None, None

def check_if_query_is_finding(query):
    # Define the prompt to determine if the query is about finding weather conditions
    prompt = f"Determine if the following query is only about finding weather conditions of a certain place or not. output format: 1 for yes or 0 for no: '{query}'"
    
    result = query_chatgpt(prompt)
    if result is not None:
        result = result.strip().lower()
        if result in ['1', '0']:
            if result == '1':
                # Extract place name from the query
                # For simplicity, assume the place name is mentioned after 'of' in the query
                # This is a very basic approach, and might need a more sophisticated method in real scenarios
                place_name = query.split('of')[-1].strip().replace('?', '')
                latitude, longitude = get_geocode(place_name)
                if latitude and longitude:
                    return 1, latitude, longitude
                else:
                    return 1, None, None
            else:
                return 0, None, None
        else:
            print("Received invalid response from ChatGPT.")
            return None, None, None
    else:
        print("Failed to get a response from ChatGPT.")
        return None, None, None

# Example usage
# query = "Why is the weather of New Delhi cold?"
# result, latitude, longitude = check_if_query_is_finding(query)
# if result == 1:
#     print(f"The query is about finding weather conditions.")
#     if latitude and longitude:
#         print(f"Longitude: {longitude}, Latitude: {latitude}")
#     else:
#         print("Could not find the location details.")
# else:
#     print("The query is not about finding weather conditions.")



