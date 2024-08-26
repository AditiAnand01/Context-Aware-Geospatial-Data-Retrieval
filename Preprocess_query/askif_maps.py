import requests

# ChatGPT API credentials
api_key = "f589f5edfdmsh721f60105a15cebp1102acjsnd65ba082541c"
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

def check_if_query_is_finding_nearby(query):
    # First, ask if the query is about finding nearby places
    yes_no_prompt = f"Does the given query ask about finding places near a location? Answer with 'yes' or 'no': '{query}'"
    
    yes_no_result = query_chatgpt(yes_no_prompt)
    
    if yes_no_result is not None:
        yes_no_result = yes_no_result.strip().lower()
        
        if yes_no_result == 'yes':
            # If yes, ask for the place type and location
            detail_prompt = f"Identify the type of place (e.g., restaurant, park, etc.) and the location in the query: '{query}'. Format: place_type, location_name."
            detail_result = query_chatgpt(detail_prompt)
            
            if detail_result is not None:
                detail_result = detail_result.strip().lower().split(',')
                if len(detail_result) == 2:
                    place_type = detail_result[0].strip()
                    place_name = detail_result[1].strip().replace('?', '')
                    latitude, longitude = get_geocode(place_name)
                    if latitude and longitude:
                        return 1, place_type, latitude, longitude
                    else:
                        return 1, place_type, None, None
        elif yes_no_result == 'no':
            return 0, None, None, None
        else:
            print("Received an unexpected response from ChatGPT.")
            return None, None, None, None
    else:
        print("Failed to get a response from ChatGPT.")
        return None, None, None, None

# Example usage
query = "Are there any good parks nearby in New Delhi?"
result, place_type, latitude, longitude = check_if_query_is_finding_nearby(query)
if result == 1:
    print(f"The query is about finding nearby places.")
    if latitude and longitude:
        print(f"Place type: {place_type.capitalize()}")
        print(f"Longitude: {longitude}, Latitude: {latitude}")
    else:
        print("Could not find the location details.")
else:
    print("The query is not about finding nearby places.")


