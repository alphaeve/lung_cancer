
import requests

def get_pollution(city="Ahmedabad"):
    try:
        url = f"https://api.openaq.org/v2/latest?city={city}"
        r = requests.get(url,timeout=10).json()
        pm25 = r["results"][0]["measurements"][0]["value"]
        return pm25
    except:
        return 60
