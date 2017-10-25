import os, sys
import requests
import json

from sample_data import INDONESIA_USER_POLY
from sample_data import AZE_TEST
from sample_data import LANDRIGHTS_TEST
from sample_data import SOY_BRAZIL
from sample_data import INTERSECT_BASE_GEOJSON
from sample_data import INTERSECT_PARTIALLY_WITHIN_GEOJSON


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
    dataset = sys.argv[1]
    payload = {}
    if dataset == 'land-rights':
        payload['geojson'] = json.loads(LANDRIGHTS_TEST)
        # payload['user_json'] = "{\"type\": \"FeatureCollection\", \"features\": [{\"type\": \"Feature\", \"properties\": {}, \"geometry\": {\"type\": \"Polygon\", \"coordinates\": [[[102.65625, -0.11535636737818807], [102.32666015625, -0.17578097424708533], [102.5628662109375, -0.41198375451568836], [102.68920898437499, -0.21972602392080884], [102.65625, -0.11535636737818807]]]}}]}"
    elif dataset == 'aze':
        payload['geojson'] = json.loads(AZE_TEST)
    elif dataset == 'soy' or dataset == 'brazil-biomes':
        payload['geojson'] = json.loads(SOY_BRAZIL)
    elif dataset == 'none':
        payload['geojson'] = json.loads(INTERSECT_BASE_GEOJSON)
        payload['geojson2'] = json.loads(INTERSECT_PARTIALLY_WITHIN_GEOJSON)
    else:
        payload['geojson'] = json.loads(INDONESIA_USER_POLY)
    if dataset == 'modis':
        payload['period'] = '2014-01-01,2015-12-31'

    print(url.format(analysis, dataset))

    if dataset == 'none':
        dataset = ''
    result = requests.post(url.format(analysis, dataset), json=payload,
                           allow_redirects=True)
    # print(result)
    try:
        response = result.json()
        assert isinstance(response, dict)
        if "intersect-geom" in response.keys():
            fc = json.loads(response['intersect-geom'])
            assert fc['type'] == 'FeatureCollection'
            # print(response['intersect-geom'])
            json.dump(fc, open('output.json', 'w'))
        for key, val in response.items():
            if key != "intersect-geom":
                print('{}: {}'.format(key, val))
    except Exception as e:
        print(str(e))
        print(result.content)


if __name__ == '__main__':
    global analysis
    global dataset
    analysis = os.environ['ANALYSIS']
    dataset = sys.argv[1]
    test_hello_local()
    #test_hello_control_tower()
    test_local()
    #test_control_tower()
    # test_brazil_biomes_control_tower_remote()
