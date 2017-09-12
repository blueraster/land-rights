from os import path, remove
import gzip

from urllib.request import urlretrieve

fixtures = path.abspath(path.join(path.dirname(__file__), 'fixtures'))


def get_indonesia():
    fixt = path.join(fixtures, 'IDN_adm0.shp.geojson')
    if not path.exists(fixt):
        download_sample_data()

    with open(fixt, 'rb') as f:
        return "".join(f.read().split())


def download_sample_data():

    big_files = ['IDN_adm0.shp.geojson.gz',
                 'FJI_adm0.shp.geojson.gz']

    repo = 'https://github.com/brendancol/geo-fixtures'

    for name in big_files:
        url = path.join(repo, 'blob/master', name + '?raw=true')
        gzip_path = path.join(fixtures, name)
        output_path = path.join(fixtures, name.replace('.gz', ''))

        urlretrieve(url, gzip_path)
        with gzip.open(gzip_path, 'rb') as f:
            with open(output_path, 'wb') as outf:
                outf.write(f.read())
                print('added sample data: {}'.format(name))

        remove(gzip_path)

with open(path.join(fixtures, 'maine.geojson')) as f:
    MAINE_GEOJSON = "".join(f.read().split())

with open(path.join(fixtures, 'self-intersecting.geojson')) as f:
    SELF_INTERSECTING_GEOJSON = "".join(f.read().split())

with open(path.join(fixtures, 'dissolve_me.geojson')) as f:
    DISSOLVE_GEOJSON = "".join(f.read().split())

with open(path.join(fixtures, 'intersect_base.geojson')) as f:
    INTERSECT_BASE_GEOJSON = "".join(f.read().split())

with open(path.join(fixtures, 'intersect_fully_within.geojson')) as f:
    INTERSECT_FULLY_WITHIN_GEOJSON = "".join(f.read().split())

with open(path.join(fixtures, 'intersect_partially_within.geojson')) as f:
    INTERSECT_PARTIALLY_WITHIN_GEOJSON = "".join(f.read().split())

with open(path.join(fixtures, 'brasil_user_poly.geojson')) as f:
    BRAZIL_USER_POLY = "".join(f.read().split())

with open(path.join(fixtures, 'indonesia_user_poly.geojson')) as f:
    INDONESIA_USER_POLY = "".join(f.read().split())

with open(path.join(fixtures, 'intersect_multiple_features.geojson')) as f:
    INTERSECT_MULTIPLE_FEATURES = "".join(f.read().split())

with open(path.join(fixtures, 'AZE_test.geojson')) as f:
    AZE_TEST = "".join(f.read().split())

with open(path.join(fixtures, 'land-rights_test.geojson')) as f:
    LANDRIGHTS_TEST = "".join(f.read().split())

with open(path.join(fixtures, 'soy_brazil.geojson')) as f:
    SOY_BRAZIL = "".join(f.read().split())
