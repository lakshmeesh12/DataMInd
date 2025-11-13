import requests

url = "http://localhost:6333/collections"
resp = requests.get(url)
print(resp.json())
