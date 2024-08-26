import requests

def get(query):
    url = "https://copilot5.p.rapidapi.com/copilot"
    payload = {
        "message": query,
        "conversation_id": None,
        "tone": "BALANCED",
        "markdown": False,
        "photo_url": None
    }
    headers = {
        "x-rapidapi-key": "55ed732f30msh60271654f3c06d2p15c128jsn9aeeb6b028bf",
        "x-rapidapi-host": "copilot5.p.rapidapi.com",
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)
    data = response.json()

    # Extracting and formatting results
    message = data.get('data', {}).get('message', 'No information available')
    places = data.get('data', {}).get('places', [])

    results = [message]
    for place in places:
        title = place.get('title')
        url = place.get('url')
        image = place.get('image', '')
        if title and url:
            result = f"- {title}: {url}"
            if image:
                result += f" (Image: {image})"
            results.append(result)
    
    return "\n".join(results)

# # Example usage
# query = "waether in JAMMU"
# print(get(query))
