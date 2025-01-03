import requests

# url = "http://localhost:9696/predict"  # for use with docker local testing
url = "http://localhost:8080/predict"  # for use with kind(kubernates)local testing

data = {
    "url": "https://github.com/ranga4all1/geo-image-classification/blob/main/images/glacier.jpg?raw=true"
}

result = requests.post(url, json=data).json()
print(result)
