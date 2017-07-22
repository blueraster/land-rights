"""-"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from flask import Blueprint, jsonify


# GENERIC Error

def error(status=400, detail='Bad Request'):
    return jsonify(errors=[{
        'status': status,
        'detail': detail,
        'david': 'david'
    }]), status

endpoints = Blueprint('endpoints', __name__)
