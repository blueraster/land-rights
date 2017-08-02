import polyIntersect.micro_functions.utils as u

import json
import itertools
import rtree
import requests

from functools import partial, lru_cache
import pyproj
import numpy as np
from scipy import stats

from shapely.geometry import Polygon, shape, mapping
from shapely.ops import unary_union, transform


__all__ = ['json2ogr', 'ogr2json', 'dissolve', 'intersect', 'buffer_to_dist',
           'get_aoi_area', 'get_intersect_area', 'get_intersect_area_percent',
           'get_intersect_count', 'esri_server2ogr']


def json2ogr(in_json):
    '''
    Convert geojson object to GDAL geometry
    '''

    if isinstance(in_json, str):
        in_json = json.loads(in_json)

    if not isinstance(in_json, dict):
        raise ValueError('input json must be dictionary')

    if 'features' not in in_json.keys():
        raise ValueError('input json must contain features property')

    for f in in_json['features']:
        f['geometry'] = shape(f['geometry'])

    return in_json


def ogr2json(featureset):
    '''
    Convert GDAL geometry to geojson object
    '''
    for f in featureset['features']:
        f['geometry'] = mapping(f['geometry'])

    return featureset


@lru_cache(5)
def esri_server2ogr(layer_endpoint, envelope, where='1=1',
                    out_fields='*', return_geometry=True,
                    token=None):

    url = layer_endpoint.replace('?f=pjson', '') + '/query'

    params = {}
    params['where'] = where
    params['outFields'] = out_fields
    params['returnGeometry'] = return_geometry
    params['token'] = token
    params['f'] = 'geojson'
    params['geometry'] = envelope
    params['geometryType'] = 'esriGeometryEnvelope'

    req = requests.post(url, data=params)
    req.raise_for_status()

    # TODO: remove extra json parse here....
    return json2ogr(req.text)


def cartodb2ogr(service_endpoint):
    pass


def dissolve(featureset, field=None):
    '''
    Dissolve a set of geometries on a field, or dissolve fully to a single
    feature if no field is provided
    '''

    if field:
        def sort_func(k):
            return k['properties'][field]
    else:
        sort_func = None

    new_features = []

    if sort_func:
        features = sorted(featureset['features'], key=sort_func)
        for key, group in itertools.groupby(features, key=sort_func):
            properties, geom = zip(*[(f['properties'],
                                      f['geometry']) for f in group])
            new_features.append({'geometry': unary_union(geom),
                                 'properties': properties[0]})

    else:
        geom = [f['geometry'] for f in featureset['features']]
        properties = {}  # TODO: decide which attributes should go in here
        new_features.append({'geometry': unary_union(geom),
                             'properties': properties})

    new_featureset = dict(type=featureset['type'],
                          features=new_features)
    if 'crs' in featureset.keys():
        new_featureset['crs'] = featureset['crs']
    return new_featureset


def index_featureset(featureset):
    '''
    '''
    index = rtree.index.Index()
    for i, f in enumerate(featureset['features']):
        geom = f['geometry']
        index.insert(i, geom.bounds)
    return index


def intersect(featureset1, featureset2):
    '''
    '''
    index = index_featureset(featureset2)

    new_features = []

    for f in featureset1['features']:
        geom1 = f['geometry']
        for fid in list(index.intersection(geom1.bounds)):
            feat2 = featureset2['features'][fid]
            geom2 = feat2['geometry']
            if geom1.intersects(geom2):  # TODO: optimize to on intersect call?
                new_geom = geom1.intersection(geom2)
                new_feat = dict(properties=feat2['properties'],
                                geometry=new_geom)
                new_features.append(new_feat)

    new_featureset = dict(type=featureset2['type'],
                          features=new_features)
    if 'crs' in featureset2.keys():
        new_featureset['crs'] = featureset2['crs']
    return new_featureset


def project_local(featureset):
    '''
    Transform geometry with a local projection centered at the
    shape's centroid
    '''
    if featureset['crs']['properties']['name'] == 'urn:ogc:def:uom:EPSG::9102':
        raise ValueError('geometries have already been projected with the \
                          World Azimuthal Equidistant coordinate system')

    new_features = []

    for f in featureset['features']:
        geom = Polygon(f['geometry'])
        proj4 = '+proj=aeqd +lat_0={} +lon_0={} +x_0=0 +y_0=0 +datum=WGS84 \
                 +units=m +no_defs '.format(geom.centroid.y, geom.centroid.x)
        project = partial(pyproj.transform,
                          pyproj.Proj(init='epsg:4326'),
                          pyproj.Proj(proj4))
        projected_geom = transform(project, geom)
        new_feat = dict(properties=f['properties'],
                        geometry=projected_geom)
        new_features.append(new_feat)

    return dict(type=featureset['type'],
                crs=dict(type='name',
                         properties=dict(name='urn:ogc:def:uom:EPSG::9102')),
                features=new_features)


def buffer_to_dist(featureset, distance):
    '''
    Buffer a geometry with a given distance (assumed to be kilometers)
    '''
    if not (featureset['crs']['properties']['name'] ==
            'urn:ogc:def:uom:EPSG::9102'):
        raise ValueError('geometries must be projected with the World \
                          Azimuthal Equidistant coordinate system')

    new_features = []

    for f in featureset['features']:
        geom = f['geometry']
        buffered_geom = geom.buffer(distance * 1000.0)
        new_feat = dict(properties=f['properties'],
                        geometry=buffered_geom)
        new_features.append(new_feat)

    new_featureset = dict(type=featureset['type'],
                          features=new_features)
    if 'crs' in featureset.keys():
        new_featureset['crs'] = featureset['crs']
    return new_featureset


# ------------------------- Calculation Functions --------------------------

def get_aoi_area(featureset):
    if len(featureset['features']) > 1:
        raise ValueError('AOI must be dissolved to a single feature')
    return np.sum([f['geometry'].area for f in featureset['features']])


def validate_featureset(featureset, field=None):
    '''
    '''
    if field:
        field_vals = [f['properties'][field] for f in featureset['features']]
        if not len(field_vals) == len(set(field_vals)):
            raise ValueError('Intersected area must be dissolved to a single \
                              feature per unique value in the category field')
    else:
        if len(featureset['features']) > 1:
            raise ValueError('Intersected area must be dissolved to a single \
                              feature if no category field is specified')


def get_intersect_area(featureset, intersection, category=None):
    '''
    Calculate the area overlap of an intersection with the user AOI. Can
    calculate areas by category using a groupby field.

    If calculating areas by category, there must be one feature per unique
    value in the category field. If not, there must be one feature total in
    the intersected featureset
    '''
    validate_featureset(intersection, category)
    for f in intersection['features']:
        f['properties']['area'] = f['geometry'].area

    if category:
        area_overlap = {f['properties'][category]: f['properties']['area']
                        for f in intersection['features']}
    else:
        f = intersection['features'][0]
        area_overlap = f['properties']['area']

    return area_overlap


def get_intersect_area_percent(featureset, intersection, category=None):
    '''
    Calculate the area overlap of an intersection with the user AOI. Can
    calculate areas by category using a groupby field.

    If calculating areas by category, there must be one feature per unique
    value in the category field. If not, there must be one feature total in
    the intersected featureset
    '''
    validate_featureset(intersection, category)
    aoi_area = get_aoi_area(featureset)
    for f in intersection['features']:
        f['properties']['area_percent'] = f['geometry'].area * 100. / aoi_area

    if category:
        pct_overlap = {f['properties'][category]:
                       f['properties']['area_percent']
                       for f in intersection['features']}
    else:
        f = intersection['features'][0]
        pct_overlap = f['properties']['area_percent']

    return pct_overlap


def get_intersect_count(intersection, field):
    '''
    Count numerical attribute from features of an intersection with the
    user AOI
    '''
    return np.sum([f['properties'][field] for f in intersection['features']])


def get_intersect_z_scores(intersection, category, field):
    '''
    Get z score of numerical attribute from features of an intersection with
    the user AOI
    '''
    validate_featureset(intersection, category)
    scores = stats.zscore([f['properties'][field]
                           for f in intersection['features']])
    for i, f in enumerate(intersection['features']):
        f['properties']['z-score'] = scores[i]
    return {f['properties'][category]: f['properties']['z-score']
            for f in intersection['features']}


def is_valid(analysis_method):
    '''
    Validate that method exists
    '''
    return analysis_method in __all__


# ---------------------------- Previous Version ----------------------------

def intersect_area_geom(user_json, intersect_json,
                        return_intersect_geom=False):
    try:
        # VALIDATE INPUT JSON FILES AND CONTAIN ONLY POLYGONS OR MULTIPOLYGONS
        u.verify_polygons(user_json)
        u.verify_polygons(intersect_json)

        # ===== DEFINE THE RESULT OBJECT ===== #
        result = {}

        # ===== CONVERT INPUT JSON TO OGR POLYGONS ===== #
        user_ogr = u.json_polys_to_ogr(user_json)
        intersect_ogr = u.json_polys_to_ogr(intersect_json)

        # ===== DISSOLVE USER POLYS AND CREATE 10KM AND 50KM BUFFERS ===== #
        user_dissolve = u.dissolve_ogr_to_single_feature(user_ogr)
        buff_10km = u.buffer_ogr_polygons(user_dissolve.Clone(), 10000)
        buff_50km = u.buffer_ogr_polygons(user_dissolve.Clone(), 50000)

        # ===== DISSOLVE INTERSECTION POLYGONS TO A SINGLE FEATURE ===== #
        intersect_polygons = u.dissolve_ogr_to_single_feature(intersect_ogr)

        # ===== CALCULATE INTERSECTIONS WITH INTERSECT POLYGONS ===== #
        intersection_user = user_dissolve.Intersection(intersect_polygons)
        intersection_10km = buff_10km.Intersection(intersect_polygons)
        intersection_50km = buff_50km.Intersection(intersect_polygons)

        # ===== IF RETURN INTERSECT GEOMETRY IS TRUE, POPULATE RESULTS WITH
        # INTERSECT GEOMETRY ===== #
        # ===== THIS IS DONE HERE BECAUSE THE u.calculate_dissolved_area
        # FUNCTION MODIFIES THE GEOMETRY OBJECTS ===== #
        if return_intersect_geom:
            result['intersect_geom_user'] = intersection_user.ExportToJson()
            result['intersect_geom_10km'] = intersection_10km.ExportToJson()
            result['intersect_geom_50km'] = intersection_50km.ExportToJson()

        # ===== CALCULATE AREA OF USER POLYS AND BUFFERS ===== #
        user_area_ha = u.calculate_dissolved_area(user_dissolve)
        buff_10km_ha = u.calculate_dissolved_area(buff_10km)
        buff_50km_ha = u.calculate_dissolved_area(buff_50km)

        # ===== CALCULATE AREA OF INTERSECTIONS ===== #
        user_area_int_ha = u.calculate_dissolved_area(intersection_user)
        buff_10km_int_ha = u.calculate_dissolved_area(intersection_10km)
        buff_50km_int_ha = u.calculate_dissolved_area(intersection_50km)

        # ===== CALCULATE % OVERLAP ===== #
        pct_overlap_user = user_area_int_ha / user_area_ha * 100
        pct_overlap_10km = buff_10km_int_ha / buff_10km_ha * 100
        pct_overlap_50km = buff_50km_int_ha / buff_50km_ha * 100

        # DEFINE THE RESULT OBJECT
        result = {}

        # DISSOLVE USER POLYS AND CREATE 10KM AND 50KM BUFFERS
        user_polys = u.dissolve_to_single_feature(user_json)
        buffs_10km = u.buffer_and_dissolve_to_single_feature(user_json, 10000)
        buffs_50km = u.buffer_and_dissolve_to_single_feature(user_json, 50000)

        # DISSOLVE INTERSECTION POLYGONS TO A SINGLE FEATURE
        intersect_polys = u.dissolve_to_single_feature(intersect_json)

        # CALCULATE INTERSECTIONS OF USER POLYGONS AND BUFFERS WITH POLYGONS
        intersection_user = user_polys.Intersection(intersect_polys)
        intersection_10km = buffs_10km.Intersection(intersect_polys)
        intersection_50km = buffs_50km.Intersection(intersect_polys)

        # IF RETURN INTERSECT GEOMETRY IS TRUE, POPULATE RESULTS WITH INTE
        # THIS IS DONE HERE BECAUSE THE u.calculate_area
        # FUNCTION MODIFIES THE GEOMETRY OBJECTS

        if return_intersect_geom:
            result['intersect_geom_user'] = intersection_user.ExportToJson()
            result['intersect_geom_10km'] = intersection_10km.ExportToJson()
            result['intersect_geom_50km'] = intersection_50km.ExportToJson()

        # CALCULATE AREA OF USER POLYS AND BUFFERS
        user_area_ha = u.calculate_area(user_polys)
        buff_10km_ha = u.calculate_area(buffs_10km)
        buff_50km_ha = u.calculate_area(buffs_50km)

        #   CALCULATE AREA OF INTERSECTIONS ===== #
        user_area_int_ha = u.calculate_area(intersection_user)
        buff_10km_int_ha = u.calculate_area(intersection_10km)
        buff_50km_int_ha = u.calculate_area(intersection_50km)

        #   CALCULATE % OVERLAP ===== #
        pct_overlap_user = user_area_int_ha / user_area_ha * 100
        pct_overlap_10km = buff_10km_int_ha / buff_10km_ha * 100
        pct_overlap_50km = buff_50km_int_ha / buff_50km_ha * 100

        #   POPULATE RESULTS WITH AREAS ===== #
        result['areaHa_user'] = user_area_ha
        result['areaHa_10km'] = buff_10km_ha
        result['areaHa_50km'] = buff_50km_ha
        result['pct_overlap_user'] = pct_overlap_user
        result['pct_overlap_10km'] = pct_overlap_10km
        result['pct_overlap_50km'] = pct_overlap_50km

        return result
    except:
        raise


def intersect_area_geom_from_endpoint(user_json, arcgis_server_layer,
                                      return_intersect_geom, buff,
                                      fields='*'):
    try:
        # ===== VERIFY THAT INPUT JSON FILES ARE VALID,
        # AND CONTAIN ONLY POLYGONS OR MULTIPOLYGONS ===== #
        u.verify_polygons(user_json)

        # ===== DEFINE THE RESULT OBJECT ===== #
        result = {}

        # ===== CONVERT INPUT JSON TO OGR POLYGONS ===== #
        user_ogr = u.json_polys_to_ogr(user_json)

        # ===== DISSOLVE USER POLYS AND CREATE 10KM AND 50KM BUFFERS ===== #
        user_dissolve = u.dissolve_ogr_to_single_feature(user_ogr)
        if buff:
            buff_10km = u.buffer_ogr_polygons(user_dissolve.Clone(), 10000)
            buff_50km = u.buffer_ogr_polygons(user_dissolve.Clone(), 50000)

        # ===== GET INTERSECT GEOMETRY FROM ARC ENDPOINT ===== #
        if buff:
            intersect_polys_json = \
                u.get_intersect_geom_from_endpoint(buff_50km,
                                                   arcgis_server_layer,
                                                   fields)
        else:
            intersect_polys_json = \
                u.get_intersect_geom_from_endpoint(user_dissolve,
                                                   arcgis_server_layer,
                                                   fields)
        u.verify_polygons(intersect_polys_json)
        # ===== CONVERT INTERSECT GEOMETRY TO OGR POLYGONS ===== #
        intersect_ogr = u.json_polys_to_ogr(intersect_polys_json)

        # ===== DISSOLVE INTERSECTION POLYGONS TO A SINGLE FEATURE ===== #
        intersect_polygons = u.dissolve_ogr_to_single_feature(intersect_ogr)

        # ===== CALCULATE INTERSECTIONS OF USER POLYGONS AND BUFFERS WITH
        # INTERSECT POLYGONS ===== #
        intersection_user = user_dissolve.Intersection(intersect_polygons)
        if buff:
            intersection_10km = buff_10km.Intersection(intersect_polygons)
            intersection_50km = buff_50km.Intersection(intersect_polygons)

        # ===== IF RETURN INTERSECT GEOMETRY IS TRUE, POPULATE RESULTS WITH
        # INTERSECT GEOMETRY ===== #
        # ===== THIS IS DONE HERE BECAUSE THE u.calculate_dissolved_area
        # FUNCTION MODIFIES THE GEOMETRY OBJECTS ===== #
        if return_intersect_geom:
            result['intersect_geom_user'] = intersection_user.ExportToJson()
            if buff:
                result['intersect_geom_10km'] = \
                    intersection_10km.ExportToJson()
                result['intersect_geom_50km'] = \
                    intersection_50km.ExportToJson()

        # ===== CALCULATE AREA OF USER POLYS AND BUFFERS ===== #
        user_area_ha = u.calculate_dissolved_area(user_dissolve)
        if buff:
            buff_10km_ha = u.calculate_dissolved_area(buff_10km)
            buff_50km_ha = u.calculate_dissolved_area(buff_50km)

        # ===== CALCULATE AREA OF INTERSECTIONS ===== #
        user_area_intersection_ha = \
            u.calculate_dissolved_area(intersection_user)
        if buff:
            buff_10km_intersection_ha = \
                u.calculate_dissolved_area(intersection_10km)
            buff_50km_intersection_ha = \
                u.calculate_dissolved_area(intersection_50km)

        # ===== CALCULATE % OVERLAP ===== #
        pct_overlap_user = user_area_intersection_ha / user_area_ha * 100
        if buff:
            pct_overlap_10km = buff_10km_intersection_ha / buff_10km_ha * 100
            pct_overlap_50km = buff_50km_intersection_ha / buff_50km_ha * 100

        # ===== POPULATE RESULTS WITH AREAS ===== #
        result['areaHa_user'] = user_area_ha
        result['pct_overlap_user'] = pct_overlap_user
        if buff:
            result['areaHa_10km'] = buff_10km_ha
            result['areaHa_50km'] = buff_50km_ha
            result['pct_overlap_10km'] = pct_overlap_10km
            result['pct_overlap_50km'] = pct_overlap_50km

        return str(result)
    except:
        raise
