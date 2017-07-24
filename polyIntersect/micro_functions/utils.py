from osgeo import ogr, osr
import geojson as gj
from geomet import wkt as WKT


def verify_polygons(in_json):

    if not in_json:
        raise ValueError('JSON input is empty.')

    loaded_json = gj.loads(in_json)

    if 'features' not in loaded_json.keys():
        raise ValueError('JSON input must contain features property')

    for feature in loaded_json['features']:
        geom_type = feature['geometry']['type']
        if 'polygon' not in geom_type.lower():
            raise ValueError('Input JSON must be of geometry type polygon.')
    return


def dissolve_to_single_feature(in_json):
    polys = ogr.Geometry(ogr.wkbMultiPolygon)
    for feature in gj.loads(in_json)['features']:
        poly = ogr.CreateGeometryFromWkt(WKT.dumps(feature['geometry']))
        polys.AddGeometry(poly)
    polys_dissolved = polys.UnionCascaded()
    return polys_dissolved


def buffer_and_dissolve_to_single_feature(in_json, distance):
    buffers = ogr.Geometry(ogr.wkbMultiPolygon)
    for feature in gj.loads(in_json)['features']:
        buff = build_buffer(gj.dumps(feature['geometry']), distance)
        buffers.AddGeometry(buff)
    buffers_dissolved = buffers.UnionCascaded()
    return buffers_dissolved


def project(ogr_geom, centroid, direction, original_epsg=4326):
    wkt_proj = 'PROJCS["World_Azimuthal_Equidistant_custom_center", \
                    GEOGCS["GCS_WGS_1984", \
                        DATUM["WGS_1984", \
                            SPHEROID["WGS_1984",6378137,298.257223563]], \
                        PRIMEM["Greenwich",0], \
                        UNIT["Degree",0.017453292519943295]], \
                    PROJECTION["Azimuthal_Equidistant"], \
                    PARAMETER["False_Easting",0], \
                    PARAMETER["False_Northing",0], \
                    PARAMETER["Central_Meridian",{}], \
                    PARAMETER["Latitude_Of_Origin",{}], \
                    UNIT["Meter",1], \
                    AUTHORITY["EPSG","54032"]]' \
                    .format(centroid.GetX(), centroid.GetY())

    original_sr = osr.SpatialReference()
    original_sr.ImportFromEPSG(original_epsg)  # 4326 = WGS84

    target_sr = osr.SpatialReference()
    target_sr.ImportFromWkt(wkt_proj)

    if direction.lower() == 'to-custom':
        transform = osr.CoordinateTransformation(original_sr, target_sr)
    elif direction.lower() == 'to-original':
        transform = osr.CoordinateTransformation(target_sr, original_sr)
    else:
        msg = ("utils.project 'direction' parameter invalid."
               "Must be either 'to-original' or 'to-custom'")
        raise Exception(msg)

    ogr_geom.Transform(transform)

    return ogr_geom


def build_buffer(json_in, distance, original_epsg=4326,
                 export_as='OGR', return_to_original_sr=True):
    wkt = WKT.dumps(gj.loads(json_in))
    poly = ogr.CreateGeometryFromWkt(wkt)
    centroid = poly.Centroid()

    poly_prj = project(poly, centroid, 'to-custom')

    buff = poly_prj.Buffer(distance)

    if return_to_original_sr:
        buff_prj = project(buff, centroid, 'to-original')

        if export_as == 'JSON':
            return buff_prj.ExportToJson()
        elif export_as == 'WKT':
            return buff_prj.ExportToWkt()
        elif export_as == 'OGR':
            return buff_prj
    else:
        if export_as == 'JSON':
            return buff.ExportToJson()
        elif export_as == 'WKT':
            return buff.ExportToWkt()
        elif export_as == 'OGR':
            return buff


def calculate_area(ogr_geom, original_epsg=4326):
    area_m2 = 0
    parts = ogr_geom.GetGeometryCount()

    if parts > 1:
        for i in range(parts):
            geom = ogr_geom.GetGeometryRef(i)
            area_part = project(geom, geom.Centroid(), 'to-custom').GetArea()
            area_m2 += area_part
    else:
        area_m2 = project(ogr_geom, ogr_geom.Centroid(), 'to-custom').GetArea()

    area_ha = area_m2 * 0.0001
    return area_ha