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

def check_if_query_is_finding_routes(query):
    # Define the prompt to determine if the query is about finding routes
    prompt = f"Given the query, find if it is asking about transportation connective routes between two places, answer in 1 or 0'{query}'"
    
    result = query_chatgpt(prompt)
    if result is not None:
        result = result.strip().lower()
        if result in ['1', '0']:
            if result == '1':
                # Extract source and destination from the query
                # For simplicity, this assumes a query format like "from [source] to [destination]"
                parts = query.lower().split('to')
                if len(parts) == 2:
                    source = parts[0].replace('from', '').strip().replace('?', '')
                    destination = parts[1].strip().replace('?', '')
                    
                    source_lat, source_lng = get_geocode(source)
                    dest_lat, dest_lng = get_geocode(destination)
                    
                    return 1, source_lat, source_lng, dest_lat, dest_lng
                else:
                    print("Query format not recognized. Could not extract source and destination.")
                    return 1, None, None, None, None
            else:
                return 0, None, None, None, None
        else:
            print("Received invalid response from ChatGPT.")
            return None, None, None, None, None
    else:
        print("Failed to get a response from ChatGPT.")
        return None, None, None, None, None

# Example usage
query = "which route is better air or railway"
result, source_lat, source_lng, dest_lat, dest_lng = check_if_query_is_finding_routes(query)
if result == 1:
    print(f"The query is about finding routes.")
    if source_lat and source_lng:
        print(f"Source Latitude: {source_lat}, Source Longitude: {source_lng}")
    else:
        print("Could not find the source location details.")
    if dest_lat and dest_lng:
        print(f"Destination Latitude: {dest_lat}, Destination Longitude: {dest_lng}")
    else:
        print("Could not find the destination location details.")
else:
    print("The query is not about finding routes.")



