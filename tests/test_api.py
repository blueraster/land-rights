import json

import pytest

from polyIntersect import app

# data
from .sample_data import BRAZIL_USER_POLY
from .sample_data import INDONESIA_USER_POLY


# test flask client
app = app.test_client()

# slow = pytest.mark.skipif(
#     not pytest.config.getoption("--runslow"),
#     reason="need --runslow option to run"
# )


# def test_hello():
#     url = '/api/v1/polyIntersect/hello?'
#     result = app.get(url)
#     assert result.status_code == 200


# def test_example_graph():
#     url = '/api/v1/polyIntersect/generic'

#     payload = {}
#     payload['analysis'] = 'area-percentarea'
#     payload['dataset'] = 'ifl'
#     payload['user_json'] = INDONESIA_USER_POLY

#     headers = {'content-type': 'application/json'}

#     result = app.post(url, data=json.dumps(payload), headers=headers)
#     result_obj = json.loads(result.data)
#     print(result_obj)

#     assert result.status_code == 200
#     assert result.content_type == 'application/json'
#     assert 'intersect-area' in list(result_obj.keys())
#     assert 'intersect-area-percent' in list(result_obj.keys())
#     assert result_obj['intersect-area'] > 0
