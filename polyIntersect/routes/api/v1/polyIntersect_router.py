from os import path
import dask
import json
from flask import request, jsonify
from polyIntersect.routes.api.v1 import endpoints
import polyIntersect.micro_functions.poly_intersect as analysis_funcs
import requests


def create_dag_from_json(graphJson):
    graph_obj = json.loads(graphJson)

    graph = dict()

    for k, v in graph_obj.items():

        if not isinstance(k, str):
            raise ValueError('graph keys must be strings')

        if not isinstance(v, list) or not len(v):
            raise ValueError(('graph values must be lists'
                              '[<func_name>, <func_arg1>, <func_arg2>]'))

        # special_funcs = ['geojson', 'esri:server', 'gfw:pro']
        special_funcs = ['geojson', 'esri:server', 'cartodb']

        func_name = v[0]
        is_valid = analysis_funcs.is_valid(func_name)
        is_special = func_name in special_funcs
        if not is_valid and not is_special:
            raise ValueError('invalid function: {}'.format(func_name))

        func_args = v[1:] if len(v) else []
        for arg in func_args:
            if not isinstance(arg, str):
                raise ValueError('graph function arguments must be strings')

        if func_name == 'geojson':
            graph[k] = tuple([analysis_funcs.json2ogr] + func_args)
        elif func_name == 'esri:server':
            graph[k] = tuple([analysis_funcs.esri_server2ogr] + func_args)
        elif func_name == 'cartodb':
            graph[k] = tuple([analysis_funcs.cartodb2ogr] + func_args)
        else:
            graph[k] = tuple([getattr(analysis_funcs, func_name)] + func_args)

    return graph


def compute(graph, outputs):
    final_output = {}
    results = dask.get(graph, outputs)
    for result, name in zip(results, outputs):
        if isinstance(result, dict) and 'features' in result.keys():
            final_output[name] = analysis_funcs.ogr2json(result)
        else:
            final_output[name] = result
    return final_output


def execute_model(analysis, dataset, user_json, unit):
    # validate user json and add unique id
    if isinstance(user_json, str):
        user_json = json.loads(user_json)
    if user_json['type'] != 'FeatureCollection':
        raise ValueError('User json must be a feature collection')
    for i in range(len(user_json['features'])):
        user_json['features'][i]['properties']['id'] = i

    # read config files
    with open(path.join(path.dirname(__file__), 'analyses.json')) as f:
        analyses = json.load(f)
    with open(path.join(path.dirname(__file__), 'datasets.json')) as f:
        datasets = json.load(f)

    # get category for dataset
    category = datasets[dataset]['category']
    field = datasets[dataset]['field']
    out_fields = ','.join([f for f in [category, field] if f])

    # get gfw api url for dataset based on its id
    dataset_id = datasets[dataset]['id']
    host = 'https://staging-api.globalforestwatch.org/v1'
    dataset_endpoint = 'dataset/{}'.format(dataset_id)
    dataset_url = path.join(host, dataset_endpoint)

    # query gfw api for the layer url
    try:
        dataset_info = requests.get(dataset_url).json()
    except:
        raise ValueError(requests.get(dataset_url).text)
    if 'errors' in dataset_info.keys():
        raise ValueError(dataset_info['errors'])
    layer_url = dataset_info['data']['attributes']['connectorUrl']
    if '?' in layer_url:
        layer_url = layer_url[:layer_url.find('?')]
    if dataset_info['data']['attributes']['provider'] == 'featureservice':
        gfw_dataset = 'esri:server'
    elif dataset_info['data']['attributes']['provider'] == 'cartodb':
        gfw_dataset = 'cartodb'
    else:
        raise ValueError('GFW dataset endpoint not supported')

    # get graph and populate with parameters
    graph = analyses[analysis]['graph']
    for key, vals in graph.items():
        vals = [val.format(user_json=json.dumps(user_json),
                           gfw_dataset=gfw_dataset,
                           out_fields=out_fields,
                           layer_url=layer_url,
                           category=category,
                           field=field,
                           unit=unit) for val in vals]
        graph[key] = vals
    outputs = analyses[analysis]['outputs']

    # create and compute graph
    dag = create_dag_from_json(json.dumps(graph))
    data = compute(dag, outputs)
    response = jsonify(data)
    response.status_code = 200
    return response


@endpoints.route('/hello', strict_slashes=False, methods=['GET', 'POST'])
def hello():
    request.json
    data = dict(name='hello adnan')
    return jsonify(data)


@endpoints.route('/plantation-species', strict_slashes=False,
                 methods=['POST'])
def plantationspecies():
    return execute_model('area-percentarea-both',
                         'plantation-species',
                         request.json['user_json'],
                         request.json['unit'])


@endpoints.route('/wdpa', strict_slashes=False,
                 methods=['POST'])
def wdpa():
    return execute_model('area-percentarea-both',
                         'wdpa',
                         request.json['user_json'],
                         request.json['unit'])


@endpoints.route('/land-rights', strict_slashes=False,
                 methods=['POST'])
def landrights():
    return execute_model('area-percentarea-both',
                         'land-rights',
                         request.json['user_json'],
                         request.json['unit'])


@endpoints.route('/aze', strict_slashes=False,
                 methods=['POST'])
def aze():
    return execute_model('area-percentarea-both',
                         'aze',
                         request.json['user_json'],
                         request.json['unit'])


@endpoints.route('/ifl', strict_slashes=False,
                 methods=['POST'])
def ifl():
    return execute_model('area-percentarea-both',
                         'ifl',
                         request.json['user_json'],
                         request.json['unit'])


@endpoints.route('/peat', strict_slashes=False,
                 methods=['POST'])
def peat():
    return execute_model('area-percentarea-both',
                         'peat',
                         request.json['user_json'],
                         request.json['unit'])
