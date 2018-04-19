import json
import itertools
import rtree
import requests
from parse import search
from geomet import wkt
from datetime import datetime, timedelta

from functools import partial, lru_cache
import pyproj
import numpy as np
from time import time
import logging

from shapely.geometry import shape, mapping, box
from shapely.geometry.polygon import Polygon
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry.collection import GeometryCollection
from shapely.geometry.point import Point
from shapely.ops import unary_union, transform


__all__ = ['json2ogr', 'ogr2json', 'dissolve', 'intersect', 'project_local',
           'project_global', 'buffer_to_dist', 'get_area', 'get_area_percent',
           'esri_server2ogr', 'get_species_count', 'esri_server2histo',
           'esri_count_groupby', 'cartodb2ogr', 'esri_count_30days',
           'esri_last_instance', 'erase', 'get_date_from_timestamp',
           'get_feature_count', 'test_ip', 'esri_attributes', 'get_presence',
           'get_histo_loss_area', 'get_histo_pre2001_area', 'get_histo_total_area',
           'get_area_by_attributes', 'get_geom_by_attributes', 'pad_counts',
           'vals_by_year', 'split', 'split_featureset', 'get_counts_by_year',
           'get_count_by_year', 'combine_counts_by_year', 'get_ok']


HA_CONVERSION = 10000
COMPLEXITY_THRESHOLD = 1.2
REQUEST_THRESHOLD = 20
FUNCTION_COUNT = 0


def get_ok():
    return 'ok'


def test_ip():
    return requests.get('http://checkip.amazonaws.com').text.replace('\n', '')


def json2ogr(in_json):
    '''
    Convert geojson object to GDAL geometry
    '''
    global FUNCTION_COUNT
    FUNCTION_COUNT += 1
    logging.info('FUNCTION json2ogr STEP {} START'.format(FUNCTION_COUNT))
    t0 = time()

    if isinstance(in_json, str):
        in_json = json.loads(in_json)

    if not isinstance(in_json, dict):
        raise ValueError('input json must be dictionary')

    if 'features' not in in_json.keys():
        raise ValueError('input json must contain features property')

    for f in in_json['features']:
        f['geometry'] = shape(f['geometry'])
        if not f['geometry'].is_valid:
            f['geometry'] = f['geometry'].buffer(0)

    for i in range(len(in_json['features'])):
        in_json['features'][i]['properties']['id'] = i

    logging.info('FUNCTION json2ogr STEP {} DONE - {} SECONDS'.format(FUNCTION_COUNT, time()-t0))
    return in_json


def ogr2json(featureset):
    '''
    Convert GDAL geometry to geojson object
    '''
    global FUNCTION_COUNT
    FUNCTION_COUNT += 1
    logging.info('FUNCTION ogr2json STEP {} START'.format(FUNCTION_COUNT))
    t0 = time()

    new_features = []
    for f in featureset['features']:
        new_features.append(dict(geometry=mapping(f['geometry']),
                                 properties=f['properties'],
                                 type=f['type']))
        # f['geometry'] = mapping(f['geometry'])

    new_featureset = dict(type=featureset['type'],
                          features=new_features)
    if 'crs' in featureset.keys():
        new_featureset['crs'] = featureset['crs']
    logging.info('FUNCTION ogr2json STEP {} DONE - {} SECONDS'.format(FUNCTION_COUNT, time()-t0))
    return json.dumps(new_featureset)


def explode(coords):
    """Explode a GeoJSON geometry's coordinates object and yield coordinate
    tuples. As long as the input is conforming, the type of the geometry
    doesn't matter.
    https://gis.stackexchange.com/questions/90553/fiona-get-each-feature-
    extent-bounds"""
    for e in coords:
        if isinstance(e, (float, int)):
            yield coords
            break
        else:
            for f in explode(e):
                yield f


def bounds(f):
    if isinstance(f['geometry'], dict):
        geom = f['geometry']['coordinates']
    else:
        try:
            geom = mapping(f['geometry'])['coordinates']
        except Exception as e:
            raise ValueError((str(e),f['geometry'],mapping(f['geometry'])))
    x, y = zip(*list(explode(geom)))
    return min(x), min(y), max(x), max(y)


def bbox(f):
    tups = mapping(box(*bounds(f)))['coordinates']
    # raise ValueError((bounds(f), tups))
    return [[list(tup) for tup in tups[0]]]


def ogr2rings(f):
    return [[list(tup) for tup in mapping(f['geometry'])['coordinates'][0]]]


# @lru_cache(5)
def esri_server2ogr(layer_endpoint, aoi, out_fields, where='1=1', token=''):
    global FUNCTION_COUNT
    FUNCTION_COUNT += 1
    logging.info('FUNCTION esri_server2ogr STEP {} START'.format(FUNCTION_COUNT))
    t0 = time()

    url = layer_endpoint.replace('?f=pjson', '') + '/query'

    params = {}
    params['where'] = where
    if 'objectid' not in out_fields:
        out_fields = 'objectid,' + out_fields if out_fields else 'objectid'
    params['outFields'] = out_fields
    params['returnGeometry'] = True
    params['returnM'] = False
    params['returnZ'] = False
    params['f'] = 'geojson'
    params['geometryType'] = 'esriGeometryPolygon'
    params['spatialRel'] = 'esriSpatialRelIntersects'
    # params['geometry'] = str({'rings': bbox(json.loads(aoi)['features'][0]),
    #                           'spatialReference': {'wkid': 4326}})

    # if protected service, retrieve token
    if token:
        params['token'] = token

    # iterate through aoi features (Esri does not accept multipart polygons
    # as a spatial filter, and the aoi features may be too far apart to combine
    # into one bounding box)
    featureset = {'type': 'FeatureCollection', 'features': []}
    features = []
    objectids = []
    if isinstance(aoi, str):
        aoi = json.loads(aoi)
    for f in aoi['features']:
        params['geometry'] = str({'rings': bbox(f),
                                  'spatialReference': {'wkid': 4326}})
        req = requests.post(url, data=params)
        req.raise_for_status()
        try:
            # response = json2ogr(req.text)
            response = req.json()
            assert 'features' in response
        except:
            raise ValueError((req.text, url, params))

        # append response to full dataset, except features already included
        for h in response['features']:
            feat_id = ','.join([str(prop) for prop in h['properties'].values()])
            if feat_id not in objectids:
                features.append(h)
                objectids.append(feat_id)

    featureset = json2ogr(dict(type='FeatureCollection',
                               features=features))

    logging.info('FUNCTION esri_server2ogr STEP {} DONE - {} SECONDS'.format(FUNCTION_COUNT, time()-t0))
    return featureset

    # req = requests.post(url, data=params)
    # req.raise_for_status()

    # return json2ogr(req.text)


def esri_server2histo(layer_endpoint, aoi):
    global FUNCTION_COUNT
    FUNCTION_COUNT += 1
    logging.info('FUNCTION esri_server2histo STEP {} START'.format(FUNCTION_COUNT))
    t0 = time()

    url = layer_endpoint.replace('?f=pjson', '') + '/computeHistograms'

    params = {}
    params['f'] = 'json'
    params['geometryType'] = 'esriGeometryPolygon'
    params['spatialRel'] = 'esriSpatialRelIntersects'
    params['returnGeometry'] = True
    params['where'] = '1=1'

    if isinstance(aoi, str):
        aoi = json.loads(aoi)
    # if featureset['features']:
    #     f = featureset['features'][0]
    histogram = [0] * 256
    for f in aoi['features']:
        params['geometry'] = str({'rings': ogr2rings(f),
                                  'spatialReference': {'wkid': 4326}})
        req = requests.post(url, data=params)
        req.raise_for_status()
        try:
            response = req.json()['histograms']
            if response:
                for i, count in enumerate(response[0]['counts']):
                    histogram[i] += count
        except Exception as e:
            raise ValueError('{} --- {}'.format(e, req.text))

    logging.info('FUNCTION esri_server2histo STEP {} DONE - {} SECONDS'.format(FUNCTION_COUNT, time()-t0))
    return histogram


def esri_attributes(layer_endpoint, aoi, out_fields):
    global FUNCTION_COUNT
    FUNCTION_COUNT += 1
    logging.info('FUNCTION esri_attributes STEP {} START'.format(FUNCTION_COUNT))
    t0 = time()

    url = layer_endpoint.replace('?f=pjson', '') + '/query'

    params = {}
    params['f'] = 'json'
    params['geometryType'] = 'esriGeometryPolygon'
    params['where'] = '1=1'
    params['spatialRel'] = 'esriSpatialRelIntersects'
    params['returnGeometry'] = False
    params['outFields'] = out_fields

    if isinstance(aoi, str):
        aoi = json.loads(aoi)
    objectids = []
    attributes = []
    for f in aoi['features']:
        params['geometry'] = str({'rings': ogr2rings(f),
                                  'spatialReference': {'wkid': 4326}})
        req = requests.post(url, data=params)
        req.raise_for_status()

        # return [feat['attributes'] for feat in req.json()['features']]
        for h in req.json()['features']:
            feat_id = ','.join([str(prop) for prop in h['attributes'].values()])
            if feat_id not in objectids:
                attributes.append(h['attributes'])
                objectids.append(feat_id)

    logging.info('FUNCTION esri_attributes STEP {} DONE - {} SECONDS'.format(FUNCTION_COUNT, time()-t0))
    return attributes


def esri_count_groupby(layer_endpoint, aoi, count_fields):
    global FUNCTION_COUNT
    FUNCTION_COUNT += 1
    logging.info('FUNCTION esri_count_groupby STEP {} START'.format(FUNCTION_COUNT))
    t0 = time()

    url = layer_endpoint.replace('?f=pjson', '') + '/query'

    params = {}
    params['f'] = 'json'
    params['geometryType'] = 'esriGeometryPolygon'
    params['where'] = '1=1'
    params['spatialRel'] = 'esriSpatialRelIntersects'
    params['returnGeometry'] = False
    params['groupByFieldsForStatistics'] = count_fields
    count_fields = count_fields.split(',')
    params['outStatistics'] = json.dumps([{
        'statisticType': 'count',
        'onStatisticField': count_fields[0],
        'outStatisticFieldName': 'count'
    }])

    if isinstance(aoi, str):
        aoi = json.loads(aoi)
    counts = {}
    for f in aoi['features']:
        params['geometry'] = str({'rings': ogr2rings(f),
                                  'spatialReference': {'wkid': 4326}})
        req = requests.post(url, data=params)
        req.raise_for_status()
        try:
            f_counts = {'-'.join([str(item['attributes'][field]) for field in
                                  count_fields]): item['attributes']['count']
                        for item in req.json()['features']}
            for key, val in f_counts.items():
                if not key in counts.keys():
                    counts[key] = val
                else:
                    counts[key] += val

        except Exception as e:
            raise ValueError((str(e), url, params, req.text))

    logging.info('FUNCTION esri_count_groupby STEP {} DONE - {} SECONDS'.format(FUNCTION_COUNT, time()-t0))
    return counts


def esri_count_30days(layer_endpoint, aoi, date_field):
    global FUNCTION_COUNT
    FUNCTION_COUNT += 1
    logging.info('FUNCTION esri_count_30days STEP {} START'.format(FUNCTION_COUNT))
    t0 = time()

    url = layer_endpoint.replace('?f=pjson', '') + '/query'

    date = (datetime.utcnow() - timedelta(days=30)).strftime('%Y-%m-%d')

    params = {}
    params['f'] = 'json'
    params['geometryType'] = 'esriGeometryPolygon'
    params['where'] = "{} >= date '{}'".format(date_field, date)
    params['spatialRel'] = 'esriSpatialRelIntersects'
    params['returnGeometry'] = False
    params['returnCountOnly'] = True

    if isinstance(aoi, str):
        aoi = json.loads(aoi)
    count = 0
    for f in aoi['features']:
        params['geometry'] = str({'rings': ogr2rings(f),
                                  'spatialReference': {'wkid': 4326}})
        req = requests.post(url, data=params)
        req.raise_for_status()
        count += req.json()['count']

    logging.info('FUNCTION esri_count_30days STEP {} DONE - {} SECONDS'.format(FUNCTION_COUNT, time()-t0))
    return count


def esri_last_instance(layer_endpoint, aoi, field):
    global FUNCTION_COUNT
    FUNCTION_COUNT += 1
    logging.info('FUNCTION esri_last_instance STEP {} START'.format(FUNCTION_COUNT))
    t0 = time()

    url = layer_endpoint.replace('?f=pjson', '') + '/query'

    params = {}
    params['f'] = 'json'
    params['geometryType'] = 'esriGeometryPolygon'
    params['where'] = '1=1'
    params['spatialRel'] = 'esriSpatialRelIntersects'
    params['returnGeometry'] = False
    params['outFields'] = field
    params['orderByFields'] = field
    params['returnDistinctValues'] = True

    if isinstance(aoi, str):
        aoi = json.loads(aoi)
    last_instance = None
    for f in aoi['features']:
        params['geometry'] = str({'rings': ogr2rings(f),
                                  'spatialReference': {'wkid': 4326}})
        req = requests.post(url, data=params)
        req.raise_for_status()
        try:
            instances = [item['attributes'][field] for item in
                         req.json()['features']]
            if instances:
                if not last_instance:
                    last_instance = instances[-1]
                elif instances[-1] > last_instance:
                    last_instance = instances[-1]

        except Exception as e:
            raise ValueError((str(e), url, params, req.text))

    logging.info('FUNCTION esri_last_instance STEP {} DONE - {} SECONDS'.format(FUNCTION_COUNT, time()-t0))
    return last_instance


# @lru_cache(5)
def cartodb2ogr(service_endpoint, aoi, out_fields, where='', _=''):
    global FUNCTION_COUNT
    FUNCTION_COUNT += 1
    logging.info('FUNCTION cartodb2ogr STEP {} START'.format(FUNCTION_COUNT))
    t0 = time()

    endpoint_template = 'https://{}.carto.com/tables/{}/'
    username, table = search(endpoint_template, service_endpoint + '/')
    url = 'https://{username}.carto.com/api/v2/sql'.format(username=username)

    if isinstance(aoi, str):
        aoi = json.loads(aoi)

    # raise ValueError()

    params = {}
    fields = ['ST_AsGeoJSON(the_geom) as geometry']
    out_fields = out_fields.split(',')
    for field in out_fields:
        if field:
            fields.append('{field} as {field}'.format(field=field))

    temp = "ST_Intersects(ST_Buffer(ST_GeomFromText('{}',4326),0),the_geom)"
    features = []
    objectids = []
    for f in aoi['features']:
        where_clause = temp.format(wkt.dumps({'type': 'Polygon',
                                              'coordinates': bbox(f)}))
        if where and not where == '1=1':
            where_clause += 'AND {}'.format(where)

        q = 'SELECT {fields} FROM {table} WHERE {where}'
        params = {'q': q.format(fields=','.join(fields), table=table,
                  where=where_clause)}

        try:
            req = requests.get(url, params=params)
            req.raise_for_status()
        except Exception as e:
            raise ValueError((e, url, bbox(f)))

        response = json.loads(req.text)['rows']
        features += [{
            'type': 'Feature',
            'geometry': json.loads(h['geometry']),
            'properties': {field: h[field] for field in out_fields if field}
        } for h in response]

    featureset = json2ogr({
        'type': 'FeatureCollection',
        'features': features
    })

    logging.info('FUNCTION cartodb2ogr STEP {} DONE - {} SECONDS'.format(FUNCTION_COUNT, time()-t0))
    return featureset


def split_featureset(featureset):
    '''
    Separate featureset into dissolved sections based on features's general proximity
    '''
    global FUNCTION_COUNT
    FUNCTION_COUNT += 1
    logging.info('FUNCTION split_featureset STEP {} START'.format(FUNCTION_COUNT))
    t0 = time()

    # new_featuresets = []
    feature_groups = []
    x1s, y1s, x2s, y2s = zip(*[bounds(f) for f in featureset['features']])  # all min/max x's and y's
    x1fc, y1fc, x2fc, y2fc = min(x1s), min(y1s), max(x2s), max(y2s)         # min/max x/y for whole feature class
    # if x2fc - x1fc > REQUEST_THRESHOLD:
    x_splits = ([x1fc + (x + 1) * REQUEST_THRESHOLD for x in
                 range(int((x2fc - x1fc - 1) / REQUEST_THRESHOLD))]
                if x2fc - x1fc > REQUEST_THRESHOLD else [])
    # if y2fc - y2fc > REQUEST_THRESHOLD:
    y_splits = ([y1fc + (y + 1) * REQUEST_THRESHOLD for y in
                 range(int((y2fc - y1 - 1) / REQUEST_THRESHOLD))]
                if y2fc - y2fc > REQUEST_THRESHOLD else [])
    if x_splits:
        # x_featuresets = [dict(type=featureset['type'], features=[])
        #                  for i in range(len(x_splits) + 1)]
        x_feature_groups = [[] for i in range(len(x_splits) + 1)]
        for f in featureset['features']:
            x1, y1, x2, y2 = bounds(f)
            for i, x_split in enumerate(x_splits):
                if x1 + (x2 - x1) / 2 < x_split:
                    # x_featuresets[i]['features'].append(f)
                    x_feature_groups[i].append(f)
                    break
                if i + 1 == len(x_splits):
                    # x_featuresets[-1]['features'].append(f)
                    x_feature_groups[-1].append(f)
    else:
        # x_featuresets = [dict(type=featureset['type'],
        #                  features=featureset['features'])]
        x_feature_groups = [featureset['features']]

    if y_splits:
        # for x_featureset in x_featuresets:
        for x_feature_group in x_feature_groups:
            # y_featuresets = [dict(type=x_featureset['type'], features=[])
            #                  for i in range(len(y_splits) + 1)]
            y_feature_groups = [[] for i in range(len(y_splits) + 1)]
            # for f in x_featureset['features']:
            for f in x_feature_group:
                x1, y1, x2, y2 = bounds(f)
                for i, y_split in enumerate(y_splits):
                    if y1 + (y2 - y1) / 2 < y_split:
                        # y_featuresets[i]['features'].append(f)
                        y_feature_groups[i].append(f)
                        break
                    if i + 1 == len(y_splits):
                        # y_featuresets[-1]['features'].append(f)
                        y_feature_groups[-1].append(f)
            # new_features.extend(y_featuresets)
            feature_groups.extend(y_feature_groups)
    else:
        # new_featuresets.extend(x_featuresets)
        feature_groups.extend(x_feature_groups)
    feature_groups = [grp for grp in x_feature_groups if grp]

    new_features = [dict(type='Feature',
                         geometry=unary_union([f['geometry']
                                               if f['geometry'].is_valid
                                               else f['geometry'].buffer(0)
                                               for f in features]),
                         properties={}) for features in feature_groups]

    new_featureset = dict(type=featureset['type'],
                          features=new_features)

    logging.info('FUNCTION split_featureset STEP {} DONE - {} SECONDS'.format(FUNCTION_COUNT, time()-t0))
    return new_featureset


        #     if x1 + (x2 - x1) / 2 <= x_split:
        #         new_features[0]['features'].append(f)
        #     else:
        #         new_features[1]['features'].append(f)
        # featureset_left = dict(type=user_json['type'],
        #                        features=[f for f in user_json['features'] if
        #                                  analysis_funcs.bounds_centroid(f)[0]
        #                                  <= x_half])
        # featureset_right = dict(type=user_json['type'],
        #                         features=[f for f in user_json['features'] if
        #                                   analysis_funcs.bounds_centroid(f)[0]
        #                                   > x_half])


def get_split_boxes(f):
    '''
    Check if number of vertices or width or height of bounding box exceed
    thresholds. If they do, returns two revised bounding boxes (Left/Upper
    and Right/Bottom) for intersecting with the geometry
    '''
    x1, y1, x2, y2 = bounds(f)
    if (x2 - x1 > COMPLEXITY_THRESHOLD or y2 - y1 > COMPLEXITY_THRESHOLD):
        if x2 - x1 > y2 - y1:
            x_split = x1 + (x2 - x1) / 2
            return [box(x1, y1, x_split, y2), box(x_split, y1, x2, y2)]
        else:
            y_split = y1 + (y2 - y1) / 2
            return [box(x1, y1, x2, y_split), box(x1, y_split, x2, y2)]

    return None


def split_multipolygon(f):
    '''
    Split multipolygon into coterminous polygons
    '''
    new_features = [{'type': 'Feature',
                     'properties': f['properties'],
                     'geometry': poly} for poly in f['geometry']]
    return new_features


def split_polygon(f):
    '''
    Split complex geometry in half until they are below vertex and bounding
    box size constraints
    '''
    bbs = get_split_boxes(f)
    new_features = []
    if bbs:
        for bb in bbs:
            geom = f['geometry']
            if not geom.is_valid:
                geom = geom.buffer(0)
            split_feat = {'type': 'Feature',
                        'properties': f['properties'],
                        'geometry': geom.intersection(bb)}
            if split_feat['geometry'].type == 'MultiPolygon':
                poly_feats = split_multipolygon(split_feat)
                for h in poly_feats:
                    new_features.extend(split_polygon(h))
            else:
                new_features.extend(split_polygon(split_feat))
    else:
        new_features.append(f)

    return new_features


def split(featureset):
    '''
    First split all multipolygons into coterminous polygons. Then check each
    against vertex and bounding box size constraints, and split into multiple
    polygons using recursive halving if necessary
    '''
    global FUNCTION_COUNT
    FUNCTION_COUNT += 1
    logging.info('FUNCTION split STEP {} START'.format(FUNCTION_COUNT))
    t0 = time()

    new_features = []
    split_id = 0
    for f in featureset['features']:
        f['properties']['split_id'] = split_id
        split_id += 1
        if f['geometry'].type == 'MultiPolygon':
            poly_feats = split_multipolygon(f)
            for h in poly_feats:
                new_features.extend(split_polygon(h))
        elif f['geometry'].type == 'Polygon':
            new_features.extend(split_polygon(f))

    new_featureset = dict(type=featureset['type'],
                          features=new_features)
    if 'crs' in featureset.keys():
        new_featureset['crs'] = featureset['crs']

    logging.info('FUNCTION split STEP {} DONE - {} SECONDS'.format(FUNCTION_COUNT, time()-t0))
    return new_featureset


def condense_properties(properties):
    '''
    Combine common properties with duplicate values from all features
    being dissolved
    '''
    return {key: val for key, val in properties[0].items()
            if all(key in p.keys() and val == p[key] for p in properties)}


def dissolve(featureset, fields=None):
    '''
    Dissolve a set of geometries on a field, or dissolve fully to a single
    feature if no field is provided
    '''
    global FUNCTION_COUNT
    FUNCTION_COUNT += 1
    logging.info('FUNCTION dissolve STEP {} START'.format(FUNCTION_COUNT))
    t0 = time()

    if fields:
        def sort_func(k):
            return ','.join([str(k['properties'][field])
                             for field in fields.split(',')])
    else:
        sort_func = None

    new_features = []
    dissolve_id = 0
    try:
        assert isinstance(featureset, dict)
        assert 'features' in featureset.keys()
        assert isinstance(featureset['features'], list)
    except Exception as e:
        raise ValueError((str(e),featureset))
    if len(featureset['features']) > 0:
        if sort_func:
            features = sorted(featureset['features'], key=sort_func)
            for key, group in itertools.groupby(features, key=sort_func):
                properties, geoms = zip(*[(f['properties'],
                                          f['geometry']) for f in group])
                if geoms and not any(geom is None for geom in geoms):
                    try:
                        new_geom = unary_union(geoms)
                    except Exception as e:
                        new_geom = unary_union([geom if geom.is_valid
                                                else geom.buffer(0)
                                                for geom in geoms])
                    new_properties = condense_properties(properties)
                    new_properties['dissolve_id'] = dissolve_id
                    dissolve_id += 1
                    new_features.append(dict(type='Feature',
                                             geometry=new_geom,
                                             properties=new_properties))

        else:
            geoms = [f['geometry'] if f['geometry'].is_valid else
                     f['geometry'].buffer(0) for f in featureset['features']]
            new_properties = condense_properties([f['properties'] for f in
                                                  featureset['features']])
            new_properties['dissolve_id'] = dissolve_id
            dissolve_id += 1
            new_features.append(dict(type='Feature',
                                         geometry=unary_union(geoms),
                                         properties=new_properties))

    new_featureset = dict(type=featureset['type'],
                          features=new_features)
    if 'crs' in featureset.keys():
        new_featureset['crs'] = featureset['crs']

    logging.info('FUNCTION dissolve STEP {} DONE - {} SECONDS'.format(FUNCTION_COUNT, time()-t0))
    return new_featureset


def index_featureset(featureset):
    '''
    '''
    index = rtree.index.Index()
    for i, f in enumerate(featureset['features']):
        geom = f['geometry']
        if isinstance(geom, GeometryCollection):
            minx = np.min([item.bounds[0] for item in geom])
            miny = np.min([item.bounds[1] for item in geom])
            maxx = np.max([item.bounds[2] for item in geom])
            maxy = np.max([item.bounds[3] for item in geom])
            index.insert(i, (minx, miny, maxx, maxy))
        else:
            index.insert(i, geom.bounds)
    return index


def intersect(featureset1, featureset2):
    '''
    '''
    global FUNCTION_COUNT
    FUNCTION_COUNT += 1
    logging.info('FUNCTION intersect STEP {} START'.format(FUNCTION_COUNT))
    t0 = time()

    index = index_featureset(featureset2)

    new_features = []

    for f in featureset1['features']:
        feat1 = f
        geom1 = f['geometry']
        for fid in list(index.intersection(geom1.bounds)):
            feat2 = featureset2['features'][fid]
            geom2 = feat2['geometry']
            if not geom1.is_valid:
                # raise ValueError('Geometry from featureset1 is not valid')
                geom1 = geom1.buffer(0)
            if not geom2.is_valid:
                # raise ValueError('Geometry from featureset2 is not valid')
                geom2 = geom2.buffer(0)

            if geom1.intersects(geom2):  # TODO: optimize to on intersect call?
                new_geom = geom1.intersection(geom2)
                new_feat = dict(properties={**feat2['properties'],
                                            **feat1['properties']},
                                geometry=new_geom,
                                type='Feature')
                new_features.append(new_feat)

    new_featureset = dict(type=featureset2['type'],
                          features=new_features)
    if 'crs' in featureset2.keys():
        new_featureset['crs'] = featureset2['crs']

    logging.info('FUNCTION intersect STEP {} DONE - {} SECONDS'.format(FUNCTION_COUNT, time()-t0))
    return new_featureset


def erase(featureset, erase_featureset):
    '''
    '''
    global FUNCTION_COUNT
    FUNCTION_COUNT += 1
    logging.info('FUNCTION erase STEP {} START'.format(FUNCTION_COUNT))
    t0 = time()

    index = index_featureset(erase_featureset)

    new_features = []

    for f in featureset['features']:
        feat = f
        geom = f['geometry']
        for fid in list(index.intersection(geom.bounds)):
            erase_feat = erase_featureset['features'][fid]
            erase_geom = erase_feat['geometry']
            if geom.intersects(erase_geom):
                new_geom = geom.difference(erase_geom)
                new_feat = dict(properties={**feat['properties']},
                                geometry=new_geom,
                                type='Feature')
                new_features.append(new_feat)

    new_featureset = dict(type=featureset['type'],
                          features=new_features)
    if 'crs' in featureset.keys():
        new_featureset['crs'] = featureset['crs']

    logging.info('FUNCTION erase STEP {} DONE - {} SECONDS'.format(FUNCTION_COUNT, time()-t0))
    return new_featureset


def project_feature(f, project):
    if isinstance(f['geometry'], Polygon):
        geom = Polygon(f['geometry'])
    elif isinstance(f['geometry'], MultiPolygon):
        geom = MultiPolygon(f['geometry'])
    elif isinstance(f['geometry'], GeometryCollection):
        geom = GeometryCollection(f['geometry'])
    elif isinstance(f['geometry'], Point):
        geom = Point(f['geometry'])

    projected_geom = transform(project, geom)
    new_feat = dict(properties=f['properties'],
                    geometry=projected_geom,
                    type='Feature')

    return new_feat


def project_local(featureset):
    global FUNCTION_COUNT
    FUNCTION_COUNT += 1
    logging.info('FUNCTION project_local STEP {} START'.format(FUNCTION_COUNT))
    t0 = time()

    if ('crs' in featureset.keys() and
            featureset['crs']['properties']['name'] ==
            'urn:ogc:def:uom:EPSG::9102'):
        return featureset

    name = 'urn:ogc:def:uom:EPSG::9102'

    # get cumulative centroid of all features
    # x, y = 0, 0
    new_features = []
    for f in featureset['features']:
        if isinstance(f['geometry'], GeometryCollection):
            x = np.mean([geom_item.centroid.x for geom_item in f['geometry']])
            y = np.mean([geom_item.centroid.y for geom_item in f['geometry']])
        else:
            x = f['geometry'].centroid.x
            y = f['geometry'].centroid.y
    # x = x / len(featureset['features']) if featureset['features'] else 0
    # y = y / len(featureset['features']) if featureset['features'] else 0

        # define local projection
        proj4 = '+proj=aeqd +lat_0={} +lon_0={} +x_0=0 +y_0=0 +datum=WGS84 \
                 +units=m +no_defs +R=6371000 '.format(y, x)

        # define projection transformation
        project = partial(pyproj.transform,
                          pyproj.Proj(init='epsg:4326'),
                          pyproj.Proj(proj4))

        # peoject features and add projection info
        new_feat = project_feature(f, project)
        new_feat['properties']['centroid'] = (x,y)
        new_features.append(new_feat)

    new_featureset = dict(type=featureset['type'],
                          features=new_features,
                          crs=dict(type="name",
                                   properties=dict(name=name)))

    logging.info('FUNCTION project_local STEP {} DONE - {} SECONDS'.format(FUNCTION_COUNT, time()-t0))
    return new_featureset


def project_global(featureset):
    global FUNCTION_COUNT
    FUNCTION_COUNT += 1
    logging.info('FUNCTION project_global STEP {} START'.format(FUNCTION_COUNT))
    t0 = time()

    if ('crs' in featureset.keys() and
            featureset['crs']['properties']['name'] == 'EPSG:4326'):
        return featureset
    elif 'crs' not in featureset.keys():
        raise ValueError('Local projection must have crs info to reproject')

    name = 'EPSG:4326'
    # [x, y] = featureset['crs']['properties']['centroid']

    new_features = []
    for f in featureset['features']:
        (x, y) = f['properties']['centroid']

        proj4 = '+proj=aeqd +lat_0={} +lon_0={} +x_0=0 +y_0=0 +datum=WGS84 \
                 +units=m +no_defs +R=6371000 '.format(y, x)

        project = partial(pyproj.transform,
                          pyproj.Proj(proj4),
                          pyproj.Proj(init='epsg:4326'))

        new_feat = project_feature(f, project)
        new_feat['properties'] = {key: val for key, val in
                                  new_feat['properties'].items()
                                  if not key == 'centroid'}
        new_features.append(new_feat)

    new_featureset = dict(type=featureset['type'],
                          features=new_features,
                          crs=dict(type="name",
                                   properties=dict(name=name)))
    logging.info('FUNCTION project_global STEP {} DONE - {} SECONDS'.format(FUNCTION_COUNT, time()-t0))
    return new_featureset


def buffer_to_dist(featureset, distance):
    '''
    Buffer a geometry with a given distance (assumed to be kilometers)
    '''
    global FUNCTION_COUNT
    FUNCTION_COUNT += 1
    logging.info('FUNCTION buffer_to_dist STEP {} START'.format(FUNCTION_COUNT))
    t0 = time()

    if not (featureset['crs']['properties']['name'] ==
            'urn:ogc:def:uom:EPSG::9102'):
        raise ValueError('geometries must be projected with the World ' +
                         'Azimuthal Equidistant coordinate system')

    new_features = []

    for f in featureset['features']:
        geom = f['geometry']
        buffered_geom = geom.buffer(int(distance) * 1000.0)
        new_feat = dict(properties=f['properties'],
                        geometry=buffered_geom,
                        type='Feature')
        new_features.append(new_feat)

    new_featureset = dict(type=featureset['type'],
                          features=new_features)
    if 'crs' in featureset.keys():
        new_featureset['crs'] = featureset['crs']
    logging.info('FUNCTION buffer_to_dist STEP {} DONE - {} SECONDS'.format(FUNCTION_COUNT, time()-t0))
    return new_featureset


def get_presence(attributes, field):
    global FUNCTION_COUNT
    FUNCTION_COUNT += 1
    logging.info('FUNCTION get_presence STEP {} START'.format(FUNCTION_COUNT))
    t0 = time()

    result = any(item[field] > 0 for item in attributes)

    logging.info('FUNCTION get_presence STEP {} DONE - {} SECONDS'.format(FUNCTION_COUNT, time()-t0))
    return result


def get_area_by_attributes(featureset, posfields, negfields):
    global FUNCTION_COUNT
    FUNCTION_COUNT += 1
    logging.info('FUNCTION get_area_by_attributes STEP {} START'.format(FUNCTION_COUNT))
    t0 = time()

    posfields = posfields.split(',') if posfields else []
    negfields = negfields.split(',') if negfields else []
    try:
        area_m = sum([f['geometry'].area for f in featureset['features']
                      if all(f['properties'][fld] and
                             f['properties'][fld] > 0 for fld in posfields)
                      and all(f['properties'][fld] and
                              f['properties'][fld] < 0 for fld in negfields)])
    except:
        raise ValueError([f['properties'] for field in posfields for f in featureset['features'] if f['properties'][field] is None])
    
    logging.info('FUNCTION get_area_by_attributes STEP {} DONE - {} SECONDS'.format(FUNCTION_COUNT, time()-t0))
    return area_m / HA_CONVERSION


def get_geom_by_attributes(featureset, posfields, negfields):
    global FUNCTION_COUNT
    FUNCTION_COUNT += 1
    logging.info('FUNCTION get_geom_by_attributes STEP {} START'.format(FUNCTION_COUNT))
    t0 = time()

    posfields = posfields.split(',') if posfields else []
    negfields = negfields.split(',') if negfields else []
    features = [f for f in featureset['features']
                if all(f['properties'][fld] and f['properties'][fld] > 0
                       for fld in posfields)
                  and all(f['properties'][fld] and f['properties'][fld] < 0
                          for fld in negfields)]
    new_featureset = dict(type=featureset['type'],
                          features=features)
    if 'crs' in featureset.keys():
        new_featureset['crs'] = featureset['crs']
    logging.info('FUNCTION get_geom_by_attributes STEP {} DONE - {} SECONDS'.format(FUNCTION_COUNT, time()-t0))
    return new_featureset


# def get_counts_by_year(layer_endpoints, featureset):
#     global FUNCTION_COUNT
#     FUNCTION_COUNT += 1
#     logging.info('FUNCTION get_counts_by_year STEP {} START'.format(FUNCTION_COUNT))
#     t0 = time()

#     counts = {}
#     for layer_endpoint in layer_endpoints.split(','):
#         yr = layer_endpoint.replace('/ImageServer','')[-4:]
#         frequencies = esri_server2histo(layer_endpoint, featureset)
#         counts[yr] = sum([i * freq for i, freq in enumerate(frequencies)])

#     logging.info('FUNCTION get_counts_by_year STEP {} DONE - {} SECONDS'.format(FUNCTION_COUNT, time()-t0))
#     return counts


def get_count_by_year(layer_endpoint, featureset, yr):
    global FUNCTION_COUNT
    FUNCTION_COUNT += 1
    logging.info('FUNCTION get_counts_by_year STEP {} START'.format(FUNCTION_COUNT))
    t0 = time()

    frequencies = esri_server2histo(layer_endpoint.replace('2000', yr), featureset)

    logging.info('FUNCTION get_counts_by_year STEP {} DONE - {} SECONDS'.format(FUNCTION_COUNT, time()-t0))
    return (yr, sum([i * freq for i, freq in enumerate(frequencies)]))


def combine_counts_by_year(*counts):
    logging.info(counts)
    return {yr: count for yr, count in counts}


# ------------------------- Calculation Functions --------------------------

def validate_featureset(featureset, fields=[None]):
    '''
    '''
    valid_fields = [f for f in fields if f]
    for field in valid_fields:
        for f in featureset['features']:
            if field not in f['properties'].keys():
                raise ValueError('Featureset with category field must ' +
                                 'have category field as a property of ' +
                                 'every feature')
    if len(valid_fields) == 0:
        if len(featureset['features']) > 1:
            raise ValueError('Featureset with multiple features must ' +
                             'be dissolved or have a category field in ' +
                             'order to calculate statistics')


def get_area(featureset, field=None):
    global FUNCTION_COUNT
    FUNCTION_COUNT += 1
    logging.info('FUNCTION get_area STEP {} START'.format(FUNCTION_COUNT))
    t0 = time()

    # validate_featureset(featureset, [field])

    if field:
        area = {}
        categories = set([f['properties'][field]
                          for f in featureset['features']])
        for cat in categories:
            area[cat] = sum([f['geometry'].area / HA_CONVERSION
                             for f in featureset['features']
                             if f['properties'][field] == cat])
    else:
        if featureset['features']:
            area = sum([f['geometry'].area / HA_CONVERSION
                        for f in featureset['features']])
        else:
            area = 0
    logging.info('FUNCTION get_area STEP {} DONE - {} SECONDS'.format(FUNCTION_COUNT, time()-t0))
    return area


# def get_area_percent(featureset, aoi_area, aoi_field=None, int_field=None):
#     # validate_featureset(featureset, [int_field, aoi_field])

#     if aoi_field and int_field:
#         area_pct = {}
#         for aoi, area in aoi_area.items():
#             area_pct[aoi] = {}
#             for f in [f for f in featureset['features'] if
#                       f['properties'][aoi_field] == aoi]:
#                 pct = f['geometry'].area / HA_CONVERSION / area * 100
#                 int_category = f['properties'][int_field]
#                 if int_category in area_pct[aoi].keys():
#                     area_pct[aoi][int_category] += pct
#                 else:
#                     area_pct[aoi][int_category] = pct
#     elif aoi_field:
#         area_pct = {}
#         for f in featureset['features']:
#             aoi = f['properties'][aoi_field]
#             area = aoi_area[aoi]
#             pct = f['geometry'].area / HA_CONVERSION / area * 100
#             if aoi in area_pct.keys():
#                 area_pct[aoi] += pct
#             else:
#                 area_pct[aoi] = pct
#         for aoi in aoi_area.keys():
#             if aoi not in area_pct.keys():
#                 area_pct[aoi] = 0
#     elif int_field:
#         area_pct = {}
#         for f in featureset['features']:
#             pct = f['geometry'].area / HA_CONVERSION / aoi_area * 100
#             int_category = f['properties'][int_field]
#             if int_category in area_pct.keys():
#                 area_pct[int_category] += pct
#             else:
#                 area_pct[int_category] = pct
#     else:
#         if featureset['features']:
#             area_pct = sum([f['geometry'].area / HA_CONVERSION / aoi_area
#                             * 100 for f in featureset['features']])
#         else:
#             area_pct = 0

#     return area_pct


def get_histo_loss_area(histograms, forest_density=30):
    '''
    Returns the sum of tree cover loss for years 2001 through 2014
    '''
    # density_map = {0: 10, 10: 25, 15: 40, 20: 55,
    #                25: 70, 30: 85, 50: 100, 75: 115}
    global FUNCTION_COUNT
    FUNCTION_COUNT += 1
    logging.info('FUNCTION get_histo_loss_area STEP {} START'.format(FUNCTION_COUNT))
    t0 = time()

    density_map = {10: 15, 15: 30, 20: 45, 25: 60,
                   30: 75, 50: 90, 75: 105, 100: 120}
    if forest_density not in density_map.keys():
        raise ValueError('Forest density must be one of the following:\n' +
                         '  10, 15, 20, 25, 30, 50, 75')
    year_indices = {(i+2001): range(density_map[forest_density] + i + 1, 135, 15)
                    for i in range(14)}
    histo_area_loss = {yr: 0.09 * sum([histograms[i] for i in indices])
                       for yr, indices in year_indices.items()}

    logging.info('FUNCTION get_histo_loss_area STEP {} DONE - {} SECONDS'.format(FUNCTION_COUNT, time()-t0))
    return histo_area_loss


def get_histo_pre2001_area(histograms):
    '''
    Returns the sum of histo on tree cover loss, aggregated on years prior to
    2001
    '''
    global FUNCTION_COUNT
    FUNCTION_COUNT += 1
    logging.info('FUNCTION get_histo_pre2001_area STEP {} START'.format(FUNCTION_COUNT))
    t0 = time()

    year_indices = range(15, 135, 15)
    histo_area_loss = 0.09 * sum([histograms[i] for i in year_indices])

    logging.info('FUNCTION get_histo_pre2001_area STEP {} DONE - {} SECONDS'.format(FUNCTION_COUNT, time()-t0))
    return histo_area_loss


def get_histo_total_area(histograms):
    '''
    Returns total area of histo within the aoi
    '''
    global FUNCTION_COUNT
    FUNCTION_COUNT += 1
    logging.info('FUNCTION get_histo_total_area STEP {} START'.format(FUNCTION_COUNT))
    t0 = time()

    year_indices = {(i+2001): range(i, 135, 15) for i in range(14)}
    histo_area_total = {yr: 0.09 * sum([histograms[i] for i in indices])
                        for yr, indices in year_indices.items()}

    logging.info('FUNCTION get_histo_total_area STEP {} DONE - {} SECONDS'.format(FUNCTION_COUNT, time()-t0))
    return histo_area_total


def get_date_from_timestamp(timestamp):
    '''
    Convert a timestamp (which may be in milliseconds, and is assumed to be
    UTC) to a date string of the form YYYY-MM-DD
    '''
    global FUNCTION_COUNT
    FUNCTION_COUNT += 1
    logging.info('FUNCTION get_date_from_timestamp STEP {} START'.format(FUNCTION_COUNT))
    t0 = time()

    if not timestamp:
        logging.info('FUNCTION get_date_from_timestamp STEP {} DONE - {} SECONDS'.format(FUNCTION_COUNT, time()-t0))
        return None
    if timestamp > 100000000000:
        timestamp = timestamp/1000
    logging.info('FUNCTION get_date_from_timestamp STEP {} DONE - {} SECONDS'.format(FUNCTION_COUNT, time()-t0))
    return datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d')


def get_species_count(intersection, field):
    '''
    Count number of unique species found within the features of an
    intersection with the user AOI
    '''
    global FUNCTION_COUNT
    FUNCTION_COUNT += 1
    logging.info('FUNCTION get_species_count STEP {} START'.format(FUNCTION_COUNT))
    t0 = time()

    species_list = []
    for f in intersection['features']:
        species_string = f['properties'][field][1:-1].replace('"', '')
        species_list += species_string.split(',')
    species_set = set(species_list)
    logging.info('FUNCTION get_species_count STEP {} DONE - {} SECONDS'.format(FUNCTION_COUNT, time()-t0))
    return len(species_set)


def get_feature_count(intersection, field):
    '''
    Count the number of features, or the number of features for each
    value in the intersection's field property
    '''
    global FUNCTION_COUNT
    FUNCTION_COUNT += 1
    logging.info('FUNCTION get_feature_count STEP {} START'.format(FUNCTION_COUNT))
    t0 = time()

    if field:
        counts = {}
        for f in intersection['features']:
            if f['properties'][field] in counts.keys():
                counts[f['properties'][field]] += 1
            else:
                counts[f['properties'][field]] = 1
        logging.info('FUNCTION get_feature_count STEP {} DONE - {} SECONDS'.format(FUNCTION_COUNT, time()-t0))
        return counts
    else:
        logging.info('FUNCTION get_feature_count STEP {} DONE - {} SECONDS'.format(FUNCTION_COUNT, time()-t0))
        return len(intersection['features'])


def pad_counts(counts, start_yr, end_yr):
    '''
    Pad result object for fires counts by month or year with zeros
    for all missing months or years
    '''
    global FUNCTION_COUNT
    FUNCTION_COUNT += 1
    logging.info('FUNCTION pad_counts STEP {} START'.format(FUNCTION_COUNT))
    t0 = time()

    if counts:
        if '-' in counts.keys():
            new_counts = {'{}-{}'.format(yr, mn): 0 for mn in range(1, 13)
                          for yr in range(int(start_yr), int(end_yr)+1)}
        else:
            new_counts = {str(yr): 0 for yr in range(int(start_yr),
                                                     int(end_yr)+1)}
    else:
        new_counts = {}
    for key in new_counts.keys():
        if key in counts.keys():
            new_counts[key] = counts[key]
    logging.info('FUNCTION pad_counts STEP {} DONE - {} SECONDS'.format(FUNCTION_COUNT, time()-t0))
    return new_counts


def vals_by_year(val, start_yr, end_yr):
    '''
    Store value in a by-year object (for consistency with the rest
    of the Palm Risk tool)
    '''
    global FUNCTION_COUNT
    FUNCTION_COUNT += 1
    logging.info('FUNCTION vals_by_year STEP {} START'.format(FUNCTION_COUNT))
    t0 = time()
    
    result = {str(yr): val for yr in range(int(start_yr), int(end_yr)+1)}

    logging.info('FUNCTION vals_by_year STEP {} DONE - {} SECONDS'.format(FUNCTION_COUNT, time()-t0))
    return result


def is_valid(analysis_method):
    '''
    Validate that method exists
    '''
    return analysis_method in __all__
