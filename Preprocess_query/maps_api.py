import requests

def find_places_nearby(place_type, latitude, longitude, radius=180, language="en"):
    url = "https://trueway-places.p.rapidapi.com/FindPlacesNearby"
    
    location = f"{latitude},{longitude}"
    
    querystring = {
        "location": location,
        "type": place_type,
        "radius": str(radius),
        "language": language
    }

    headers = {
        "x-rapidapi-key": "f589f5edfdmsh721f60105a15cebp1102acjsnd65ba082541c",
        "x-rapidapi-host": "trueway-places.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)

    # Check if the request was successful
    if response.status_code == 200:
        response_json = response.json()
        
        # Check if the response contains 'results' key
        if 'results' in response_json:
            # Construct a text summary of the places found
            places_summary = "Nearby Places:\n"
            for place in response_json['results']:
                places_summary += f"Name: {place.get('name', 'N/A')}\n"
                places_summary += f"Address: {place.get('vicinity', 'N/A')}\n"
                places_summary += f"Rating: {place.get('rating', 'N/A')}\n"
                places_summary += f"Types: {', '.join(place.get('types', []))}\n"
                places_summary += "\n"
            
            return places_summary
        else:
            return "No results found in the response."
    else:
        return f"API request failed with status code: {response.status_code}"

# Example usage
place_type = "restaurant"
latitude = 32.8034929
longitude = 74.896265
places_text = find_places_nearby(place_type, latitude, longitude)

print(places_text)
