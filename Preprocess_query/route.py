import requests
import json

def get_driving_directions(lat1, lon1, lat2, lon2, distance_units="auto", avoid_routes="tolls,ferries", language="en"):
    url = "https://driving-directions1.p.rapidapi.com/get-directions"
    
    origin = f"{lat1},{lon1}"
    destination = f"{lat2},{lon2}"
    
    querystring = {
        "origin": origin,
        "destination": destination,
        "distance_units": distance_units,
        "avoid_routes": avoid_routes,
        "language": language
    }

    headers = {
        "x-rapidapi-key": "f589f5edfdmsh721f60105a15cebp1102acjsnd65ba082541c",
        "x-rapidapi-host": "driving-directions1.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)

    # Check if the request was successful
    if response.status_code == 200:
        response_json = response.json()
        
        # Check if the response contains the 'best_routes' key
        if 'data' in response_json and 'best_routes' in response_json['data']:
            best_route = response_json['data']['best_routes'][0]  # Take the first route
            
            # Construct a text summary of the route
            route_summary = f"Route Summary:\n"
            route_summary += f"Route Name: {best_route['route_name']}\n"
            route_summary += f"Distance: {best_route['distance_label']}\n"
            route_summary += f"Duration: {best_route['duration_label']}\n"
            route_summary += f"Highlights: {', '.join(best_route['highlights'])}\n\n"
            
            route_summary += "Route Instructions:\n"
            if 'route_parts' in best_route:
                for part in best_route['route_parts']:
                    for instruction in part['instructions']:
                        route_summary += f"Summary: {instruction['summary']}\n"
                        route_summary += f"  Distance: {instruction['distance_label']}\n"
                        route_summary += f"  Duration: {instruction['duration_label']}\n"
                        route_summary += "\n"
            else:
                route_summary += "No route instructions available."
            
            return route_summary
        else:
            return "No best routes found in the response."
    else:
        return f"API request failed with status code: {response.status_code}"

# Example usage
lat1, lon1 = 32.8034929, 74.896265  # IIT Jammu, Nagrota
lat2, lon2 = 32.716801, 74.849907   # Tawi Railway Station, Jammu
directions_text = get_driving_directions(lat1, lon1, lat2, lon2)
print(directions_text)
