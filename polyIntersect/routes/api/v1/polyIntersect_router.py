from os import path
import dask
import json
from flask import request, jsonify
from polyIntersect.routes.api.v1 import endpoints
import polyIntersect.micro_functions.poly_intersect as analysis_funcs
import requests
from datetime import datetime


def convert_date(date):
    return datetime.strptime(date, '%Y-%m-%d').strftime('%#m/%#d/%Y')


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
        special_funcs = ['geojson', 'esri:server', 'esri:imageserver',
                         'cartodb']

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
        elif func_name == 'esri:imageserver':
            graph[k] = tuple([analysis_funcs.esri_server2histo] + func_args)
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


def execute_model(analysis, dataset, user_json, geojson2, period):
    # validate user json
    if isinstance(user_json, str):
        user_json = json.loads(user_json)
    if user_json['type'] != 'FeatureCollection':
        raise ValueError('User json must be a feature collection')

    # validate optional second user json
    if geojson2 == '':
        geojson2 = {'type': 'FeatureCollection', 'features': []}
    else:
        if isinstance(geojson2, str):
            geojson2 = json.loads(geojson2)
        if geojson2['type'] != 'FeatureCollection':
            raise ValueError('Second user json must be a feature collection')

    # read config files
    with open(path.join(path.dirname(__file__), 'analyses.json')) as f:
        analyses = json.load(f)
    with open(path.join(path.dirname(__file__), 'datasets.json')) as f:
        datasets = json.load(f)

    # get category for dataset
    category = datasets[dataset]['category']
    field = datasets[dataset]['field']
    out_fields = ','.join([f for f in [category, field] if f])
    where_dates = {
        'start_date': convert_date(period.split(',')[0]) if period else '',
        'end_date': convert_date(period.split(',')[1]) if period else ''
    }
    where = datasets[dataset]['where'].format(**where_dates)

    # get gfw api url for dataset based on its id
    dataset_id = datasets[dataset]['id']
    host = 'https://production-api.globalforestwatch.org/v1'
    dataset_endpoint = 'dataset/{}'.format(dataset_id)
    dataset_url = path.join(host, dataset_endpoint)

    # query gfw api for the layer url
    if dataset_url:
        try:
            dataset_info = requests.get(dataset_url).json()
            if 'errors' in dataset_info.keys():
                raise ValueError(dataset_info['errors'])
            layer_url = dataset_info['data']['attributes']['connectorUrl']
            if '?' in layer_url:
                layer_url = layer_url.split('?')[0]
            provider = dataset_info['data']['attributes']['provider']
            if provider == 'featureservice':
                gfw_dataset = 'esri:server'
            elif provider == 'cartodb':
                gfw_dataset = 'cartodb'
            else:
                raise ValueError('GFW dataset endpoint not supported')
        except:
            raise ValueError(requests.get(dataset_url).text)
    else:
        layer_url = ''
        gfw_dataset = ''

    # get graph and populate with parameters
    graph = analyses[analysis]['graph']
    for key, vals in graph.items():
        vals = [val.format(user_json=json.dumps(user_json),
                           user_json_2=json.dumps(geojson2),
                           gfw_dataset=gfw_dataset,
                           out_fields=out_fields,
                           layer_url=layer_url,
                           category=category,
                           field=field,
                           where=where) for val in vals]
        graph[key] = vals
    outputs = analyses[analysis]['outputs']

    # create and compute graph
    dag = create_dag_from_json(json.dumps(graph))
    data = compute(dag, outputs)
    response = jsonify(data)
    response.status_code = 200
    return response


@endpoints.route('/erase-and-intersect-geom/hello',
                 strict_slashes=False, methods=['GET', 'POST'])
def hello():
    request.json
    data = dict(name='hello adnan')
    return jsonify(data)
