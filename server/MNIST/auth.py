
from flask import (
    Blueprint, render_template
)

bp = Blueprint('auth', __name__, url_prefix='/')

@bp.route('/', methods=('GET', 'POST'))
def register():
    return render_template('index.html')