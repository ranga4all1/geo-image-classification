import requests

# url = 'http://localhost:9696/predict'
url = 'http://localhost:8080/predict'

data = {'url': 'https://github.com/ranga4all1/geo-image-classification/blob/main/images/glacier.jpg?raw=true'}

result = requests.post(url, json=data).json()
print(result)