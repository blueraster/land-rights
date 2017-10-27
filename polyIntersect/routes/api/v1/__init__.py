from flask import Blueprint, jsonify


def error(status=400, detail='Bad Request', traceback=''):
    return jsonify(errors=[{
        'status': status,
        'detail': detail,
        'traceback': traceback
        'david': 'david'
    }]), status


endpoints = Blueprint('endpoints', __name__)
import polyIntersect.routes.api.v1.polyIntersect_router  # noqa
