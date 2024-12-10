import requests # a lib for making http requests
import json # a lib for working with json

url = "https://cpsc322-finalproject-1.onrender.com/predict?top_elevation=3842&elevation_diff=2807&slope_length=152&lifts=65&number_slopes=102&snowfall=450"

response = requests.get(url=url)

# first thing, check the response's status_code
print(response.status_code)
if response.status_code == 200:
    # STATUS OK
    # we can extract the prediction from the response's JSON text
    json_object = json.loads(response.text)
    print(json_object)
    pred = json_object["prediction"]
    print("prediction:", pred)