import requests


def test_case(url, request_type, files=None):
    if request_type == 'GET':
        resp = requests.get(url)
    elif request_type == 'POST':
        resp = requests.post(url, files=files)

    print(resp)
    print(resp.content)
    print(resp.headers)
    print(resp.history)


test_case(
    'http://localhost:8000/upload','POST',
    {'file': open('test_img.png', 'rb')}
)

test_case(
    'http://localhost:8000/predict/?filename=test_img.png','GET'
)
