import requests

def get_weather_data(latitude, longitude):
    url = "https://weatherapi-com.p.rapidapi.com/current.json"
    querystring = {"q": f"{latitude},{longitude}"}

    headers = {
        "x-rapidapi-key": "164944310amsh42317543591df12p1cdfb4jsn361155ee8346",
        "x-rapidapi-host": "weatherapi-com.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)
    
    # Return the JSON response
    return response.json()

# Example usage
latitude = 32.8034929
longitude = 74.896265
weather_data = get_weather_data(latitude, longitude)
print(weather_data)
