import sys
import requests
import json

from sample_data import INDONESIA_USER_POLY
from sample_data import AZE_TEST
from sample_data import LANDRIGHTS_TEST


def test_hello():
    url = 'http://localhost:9000/v1/polyIntersect/hello'
    result = requests.get(url)
    print(result)


def test_local():
    url = 'http://localhost:5700/api/v1/polyIntersect/{}'
    run_request(url)


def test_control_tower():
    url = 'http://localhost:9000/v1/polyIntersect/{}'
    run_request(url)


def test_control_tower_remote():
    url = 'http://staging-api.globalforestwatch.org/v1/polyIntersect/{}'
    run_request(url)


def run_request(url):
    payload = {}
    if endpoint == 'land-rights':
        payload['user_json'] = LANDRIGHTS_TEST
        # payload['user_json'] = "{\"type\": \"FeatureCollection\", \"features\": [{\"type\": \"Feature\", \"properties\": {}, \"geometry\": {\"type\": \"Polygon\", \"coordinates\": [[[102.65625, -0.11535636737818807], [102.32666015625, -0.17578097424708533], [102.5628662109375, -0.41198375451568836], [102.68920898437499, -0.21972602392080884], [102.65625, -0.11535636737818807]]]}}]}"
    elif endpoint == 'aze':
        payload['user_json'] = AZE_TEST
    else:
        payload['user_json'] = INDONESIA_USER_POLY
    payload['unit'] = 'hectare'

    print(url.format(endpoint))

    result = requests.post(url.format(endpoint), json=payload,
                           allow_redirects=True)
    # print(result)
    try:
        response = result.json()
        assert isinstance(response, dict)
        for fc in response.values():
            assert json.loads(fc)['type'] == 'FeatureCollection'
            assert 'features' in json.loads(fc).keys()
            for f in json.loads(fc)['features']:
                assert f['type'] == 'Feature'
                assert 'geometry' in f.keys()
            json.dump(json.loads(fc), open('output.json', 'w'))
        print(response.keys())
    except:
        print(result.content)


if __name__ == '__main__':
    global endpoint
    endpoint = sys.argv[1]
    test_hello()
    # test_local()
    test_control_tower()
    # test_brazil_biomes_control_tower_remote()
