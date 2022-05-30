import requests
import json
# import pprint

car = {
    "buying": 'high',
    "maint": 'high',
    "doors": '3',
    "persons": '4',
    "lug_boot": 'med',
    "safety": 'med'
}


url = "https://car-test-api.herokuapp.com"
response = requests.post(f"{url}/predict",
                         json=car)

print(f"Request: {url}/predict")
print(f"Car: \n buying: {car['buying']}\n maint: {car['maint']}\n"\
      f" doors: {car['doors']}\n persons: {car['persons']}\n"\
      f" lug_boot: {car['lug_boot']}\n"\
      f" safety: {car['safety']}\n"
     )
print(f"Result of model inference: {response.json()}")
print(f"Status code: {response.status_code}")