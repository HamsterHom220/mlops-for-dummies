import requests

def run_tests():
    def test_case(url, request_type, files=None):
        if request_type == 'GET':
            resp = requests.get(url)
        elif request_type == 'POST':
            resp = requests.post(url, files=files)

        return resp.status_code, resp.content, resp.headers

    for code, content, headers in [
        test_case(
        'http://localhost:8000/upload','POST',
        {'file': open('~/MLOps/flipped_class/test_img.png', 'rb')}
        ),
        test_case(
            'http://localhost:8000/predict/?filename=test_img.png', 'GET'
        )
    ]:
        print(f"CODE: {code}",
              f"CONTENT: {content}",
              f"HEADERS: {headers}")
        assert code == 200


if __name__ == '__main__':
    run_tests()

