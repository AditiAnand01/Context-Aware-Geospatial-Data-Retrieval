import requests

def fetch(query):
    url = "https://google-api31.p.rapidapi.com/websearch"
    payload = {
        "text": query,
        "safesearch": "off",
        "timelimit": "",
        "region": "wt-wt",
        "max_results": 10
    }
    headers = {
        "x-rapidapi-key": "55ed732f30msh60271654f3c06d2p15c128jsn9aeeb6b028bf",
        "x-rapidapi-host": "google-api31.p.rapidapi.com",
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)
    data = response.json()

    # Extracting and formatting results
    results = data.get('result', [])
    formatted_results = [f"Results for '{query}':"]

    for item in results:
        title = item.get('title', 'No title')
        href = item.get('href', 'No link available')
        body = item.get('body', 'No description available')

        formatted_results.append(f"- {title}: {href}\n  Description: {body}")

    return "\n".join(formatted_results)

# # Example usage
# query = "weather in Jammu"
# print(fetch(query))
