from os import path
import sys
import rtree
import json

from polyIntersect.micro_functions.poly_intersect import esri_server2ogr
from polyIntersect.micro_functions.poly_intersect import cartodb2ogr
from polyIntersect.micro_functions.poly_intersect import json2ogr
from polyIntersect.micro_functions.poly_intersect import ogr2json
from polyIntersect.micro_functions.poly_intersect import dissolve
from polyIntersect.micro_functions.poly_intersect import intersect
from polyIntersect.micro_functions.poly_intersect import index_featureset
from polyIntersect.micro_functions.poly_intersect import buffer_to_dist
from polyIntersect.micro_functions.poly_intersect import project_features
from polyIntersect.micro_functions.poly_intersect import get_area
from polyIntersect.micro_functions.poly_intersect import get_area_percent


from shapely.geometry.polygon import Polygon
from shapely.geometry.multipolygon import MultiPolygon

from .sample_data import DISSOLVE_GEOJSON
from .sample_data import INTERSECT_BASE_GEOJSON
from .sample_data import INTERSECT_PARTIALLY_WITHIN_GEOJSON
from .sample_data import INTERSECT_MULTIPLE_FEATURES
from .sample_data import INDONESIA_USER_POLY
from .sample_data import BRAZIL_USER_POLY
from .sample_data import AZE_TEST

fixtures = path.abspath(path.join(path.dirname(__file__), 'fixtures'))


def test_successfully_index_featureset():
    featureset = json2ogr(DISSOLVE_GEOJSON)
    index = index_featureset(featureset)
    assert isinstance(index, rtree.index.Index)


def test_successfully_dissolve_string_field():
    featureset = json2ogr(DISSOLVE_GEOJSON)
    assert len(featureset['features']) == 4

    geom_diss = dissolve(featureset, field='str_value')
    assert len(geom_diss['features']) == 2


def test_successfully_dissolve_int_field():
    featureset = json2ogr(DISSOLVE_GEOJSON)
    assert len(featureset['features']) == 4

    geom_diss = dissolve(featureset, field='int_value')
    assert len(geom_diss['features']) == 2


def test_successfully_dissolve_decimal_field():
    featureset = json2ogr(DISSOLVE_GEOJSON)
    assert len(featureset['features']) == 4

    geom_diss = dissolve(featureset, field='dec_value')
    assert len(geom_diss['features']) == 2


def test_successfully_dissolve_no_field_arg():
    featureset = json2ogr(DISSOLVE_GEOJSON)
    assert len(featureset['features']) == 4

    geom_diss = dissolve(featureset)
    assert len(geom_diss['features']) == 1


def test_maintain_crs():
    featureset = json2ogr(DISSOLVE_GEOJSON)
    assert len(featureset['features']) == 4

    geom_diss = dissolve(featureset, field='str_value')
    assert len(geom_diss['features']) == 2
    assert 'crs' in geom_diss.keys()


def test_successful_intersection():

    featureset1 = json2ogr(INTERSECT_PARTIALLY_WITHIN_GEOJSON)
    featureset2 = json2ogr(INTERSECT_BASE_GEOJSON)

    result_featureset = intersect(featureset1, featureset2)
    assert len(result_featureset['features']) == 1
    assert isinstance(result_featureset['features'][0]['geometry'],
                      MultiPolygon)


def test_project():
    featureset = json2ogr(DISSOLVE_GEOJSON)
    assert len(featureset['features']) == 4

    geom_projected = project_features(featureset)
    assert isinstance(geom_projected, dict)
    assert 'features' in geom_projected.keys()
    assert (geom_projected['crs']['properties']['name']
            == 'urn:ogc:def:uom:EPSG::9102')
    assert (featureset['crs']['properties']['name']
           != 'urn:ogc:def:uom:EPSG::9102')


def test_project_already_projected():
    featureset = json2ogr(DISSOLVE_GEOJSON)
    assert len(featureset['features']) == 4

    geom_projected1 = project_features(featureset)
    try:
        geom_projected2 = project_features(geom_projected1)
    except ValueError as e:
        assert str(e) == 'geometries have already been projected with the \
                          World Azimuthal Equidistant coordinate system'


def test_projected_buffer():
    featureset = json2ogr(DISSOLVE_GEOJSON)
    assert len(featureset['features']) == 4

    geom_projected = project_features(featureset)
    geom_buffered = buffer_to_dist(geom_projected, 10)
    assert isinstance(geom_buffered, dict)
    assert 'features' in geom_buffered.keys()
    assert len(geom_buffered['features']) == 4

    for f_in, f_out in zip(featureset['features'], geom_buffered['features']):
        assert f_out['geometry'].area > f_in['geometry'].area


def test_not_projected_buffer():
    featureset = json2ogr(DISSOLVE_GEOJSON)
    assert len(featureset['features']) == 4

    try:
        geom_buffered = buffer_to_dist(featureset, 10)
    except ValueError as e:
        assert str(e) == ('geometries must be projected with the World ' +
                          'Azimuthal Equidistant coordinate system')


def test_area_percent_no_categories():
    featureset1 = json2ogr(INTERSECT_PARTIALLY_WITHIN_GEOJSON)
    for i,f in enumerate(featureset1['features']):
        f['properties']['id'] = i
    featureset2 = json2ogr(INTERSECT_BASE_GEOJSON)

    result_featureset = intersect(featureset1, featureset2)
    assert len(result_featureset['features']) == 1

    featureset1_projected = project_features(featureset1)
    result_projected = project_features(result_featureset)

    aoi_area = get_area(featureset1_projected)
    area_pct = get_area_percent(result_projected, aoi_area)

    assert area_pct
    assert isinstance(area_pct, float)
    assert area_pct > 0 and area_pct <=100


def test_area_percent_with_categories():
    featureset1 = json2ogr(INTERSECT_BASE_GEOJSON)
    for i,f in enumerate(featureset1['features']):
        f['properties']['id'] = i
    featureset2 = json2ogr(INTERSECT_MULTIPLE_FEATURES)

    result_featureset = intersect(featureset1, featureset2)
    assert len(result_featureset['features']) == 2
    field_vals = [f['properties']['value'] for f in result_featureset['features']]
    assert len(field_vals) == len(set(field_vals))

    featureset1_projected = project_features(featureset1)
    result_projected = project_features(result_featureset)

    aoi_area = get_area(featureset1_projected, 'id')
    area_pct = get_area_percent(result_projected, aoi_area, 'id', 'value')

    assert area_pct
    assert isinstance(area_pct, dict)
    for area_pct_cats in area_pct.values():
        assert isinstance(area_pct_cats, dict)
        for val in area_pct_cats.values():
            assert isinstance(val, float)
            assert val > 0 and val <=100


def test_json2ogr():
    geom_converted_version = json2ogr(DISSOLVE_GEOJSON)

    assert isinstance(geom_converted_version, dict)
    assert 'features' in geom_converted_version.keys()

    for f in geom_converted_version['features']:
        assert isinstance(f['geometry'], Polygon)


def test_ogr2json():

    geom_converted_version = json2ogr(DISSOLVE_GEOJSON)
    geom_converted_back = ogr2json(geom_converted_version)

    for i, f in enumerate(json.loads(geom_converted_back)['features']):
        assert isinstance(f['geometry'], dict)


def test_esri_server2json():
    host = 'http://gis-gfw.wri.org'
    layer = 'forest_cover/MapServer/0'
    layer_url = path.join(host, 'arcgis/rest/services', layer)

    featureset = esri_server2ogr(layer_url, INDONESIA_USER_POLY, '')
    assert 'features' in featureset.keys()
    assert len(featureset['features']) > 0
    assert isinstance(featureset['features'][0]['geometry'], (Polygon, MultiPolygon))


def test_cartodb2json():
    url = ('https://wri-01.carto.com/tables/' +
          'alliance_for_zero_extinction_sites_species_joi')

    featureset = cartodb2ogr(url, AZE_TEST, 'sitename_1,species')
    assert 'features' in featureset.keys()
    assert len(featureset['features']) > 0
    assert isinstance(featureset['features'][0]['geometry'], (Polygon, MultiPolygon))
