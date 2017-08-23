import os, sys
import requests
import json

from sample_data import INDONESIA_USER_POLY
from sample_data import AZE_TEST
from sample_data import LANDRIGHTS_TEST


def test_hello_local():
    url = 'http://localhost:5700/api/v1/polyIntersect/{}/hello'
    print(url.format(analysis))
    result = requests.get(url.format(analysis))
    print(result)

def test_hello_control_tower():
    url = 'http://localhost:9000/v1/polyIntersect/{}/hello'
    print(url.format(analysis))
    result = requests.get(url.format(analysis))
    print(result)


def test_local():
    url = 'http://localhost:5700/api/v1/polyIntersect/{}/{}'
    run_request(url)


def test_control_tower():
    url = 'http://localhost:9000/v1/polyIntersect/{}/{}'
    run_request(url)


def test_control_tower_remote():
    url = 'http://staging-api.globalforestwatch.org/v1/polyIntersect/{}/{}'
    run_request(url)


def run_request(url):
    payload = {}
    if dataset == 'land-rights':
        payload['user_json'] = LANDRIGHTS_TEST
        # payload['user_json'] = "{\"type\": \"FeatureCollection\", \"features\": [{\"type\": \"Feature\", \"properties\": {}, \"geometry\": {\"type\": \"Polygon\", \"coordinates\": [[[102.65625, -0.11535636737818807], [102.32666015625, -0.17578097424708533], [102.5628662109375, -0.41198375451568836], [102.68920898437499, -0.21972602392080884], [102.65625, -0.11535636737818807]]]}}]}"
    elif dataset == 'aze':
        payload['user_json'] = AZE_TEST
    else:
        payload['user_json'] = INDONESIA_USER_POLY
    payload['unit'] = 'hectare'

    print(url.format(analysis, dataset))

    result = requests.post(url.format(analysis, dataset), json=payload,
                           allow_redirects=True)
    # print(result)
    try:
        response = result.json()
        assert isinstance(response, dict)
        if analysis == 'count-species-dissolved':
            for count in response.values():
                assert isinstance(count, int)
            print(response)
        else:
            for fc in response.values():
                assert json.loads(fc)['type'] == 'FeatureCollection'
                assert 'features' in json.loads(fc).keys()
                for f in json.loads(fc)['features']:
                    assert f['type'] == 'Feature'
                    assert 'geometry' in f.keys()
                json.dump(json.loads(fc), open('output.json', 'w'))
            print(response.keys())
    except Exception as e:
        print(str(e))
        print(result.content)


if __name__ == '__main__':
    global analysis
    global dataset
    analysis = os.environ['ANALYSIS']
    dataset = sys.argv[1]
    test_hello_local()
    test_hello_control_tower()
    test_local()
    test_control_tower()
    # test_brazil_biomes_control_tower_remote()
