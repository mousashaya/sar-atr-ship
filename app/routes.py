"""
app/routes.py
=============
URL handlers for the Defense ML Viewer.

Routes:
  GET  /                          -- renders the main image viewer
  GET  /experiments               -- renders the experiments page
  GET  /atrnet-image/<id>         -- serves one ATRNet image chip

  GET  /api/experiments           -- list all experiments
  GET  /api/experiments/<id>      -- one experiment + confusion matrix
  GET  /api/experiments/<id>/images -- images for one confusion matrix cell
  GET  /api/training-progress     -- active training jobs from log files
"""

import re
import time
from pathlib import Path

from flask import Blueprint, render_template, jsonify, request, send_file, abort

from . import db

bp = Blueprint('main', __name__)

DATA_DIR = Path(__file__).parent.parent / 'data' / 'atrnet'


# ── Pages ─────────────────────────────────────────────────────────────────────

@bp.route('/')
def index():
    opts = db.get_atrnet_filter_options()
    return render_template('viewer.html', **opts)


@bp.route('/experiments')
def experiments():
    return render_template('experiments.html')


# ── Image serving ─────────────────────────────────────────────────────────────

@bp.route('/atrnet-image/<int:image_id>')
def serve_atrnet_image(image_id):
    """Serve an ATRNet image chip. Converts .tif to JPEG in memory for browser display."""
    from PIL import Image as PILImage
    import io
    detail = db.get_atrnet_image(image_id)
    if not detail or not detail.get('local_path'):
        abort(404)
    path = Path(detail['local_path'])
    if not path.exists():
        abort(404)
    # Convert to JPEG so browsers can display it (tif not supported by most browsers)
    try:
        img = PILImage.open(path).convert('RGB')
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=90)
        buf.seek(0)
        return send_file(buf, mimetype='image/jpeg')
    except Exception:
        abort(500)


# ── ATRNet image API ──────────────────────────────────────────────────────────

@bp.route('/api/atrnet/images')
def api_atrnet_images():
    ids, total = db.get_atrnet_image_list(
        main_class      = request.args.get('main_class', ''),
        subclass        = request.args.get('subclass', ''),
        target_type     = request.args.get('target_type', ''),
        scene           = request.args.get('scene', ''),
        depression_angle= request.args.get('depression_angle', ''),
        split           = request.args.get('split', ''),
        limit           = int(request.args.get('limit', 200)),
        offset          = int(request.args.get('offset', 0)),
    )
    return jsonify({'count': len(ids), 'total': total, 'images': ids})



@bp.route('/api/atrnet/image/<int:image_id>')
def api_atrnet_image(image_id):
    detail = db.get_atrnet_image(image_id)
    if not detail:
        return jsonify({'error': 'Not found'}), 404
    for k, v in detail.items():
        if hasattr(v, 'isoformat'):
            detail[k] = str(v)
        elif v is None:
            detail[k] = ''
    return jsonify(detail)


# ── Experiment API ────────────────────────────────────────────────────────────

@bp.route('/api/experiments')
def api_experiments():
    exps = db.get_experiments()
    for e in exps:
        for k, v in e.items():
            if hasattr(v, 'isoformat'):
                e[k] = str(v)
    return jsonify({'experiments': exps})


@bp.route('/api/experiments/<int:exp_id>')
def api_experiment_detail(exp_id):
    detail = db.get_experiment_detail(exp_id)
    if not detail:
        return jsonify({'error': 'Not found'}), 404
    for k, v in detail.items():
        if hasattr(v, 'isoformat'):
            detail[k] = str(v)
    return jsonify(detail)


@bp.route('/api/experiments/<int:exp_id>/images')
def api_experiment_images(exp_id):
    true_class      = request.args.get('true_class', '')
    predicted_class = request.args.get('predicted_class', '')
    images = db.get_experiment_images(exp_id, true_class, predicted_class)
    return jsonify({'images': images})


@bp.route('/api/training-progress')
def api_training_progress():
    """Scan logs/ for active training runs and return progress."""
    logs_dir = Path(__file__).parent.parent / 'logs'
    jobs = []

    if logs_dir.exists():
        for log_file in sorted(logs_dir.glob('train_*.log')):
            try:
                mtime     = log_file.stat().st_mtime
                age_hours = (time.time() - mtime) / 3600
                text      = log_file.read_text(errors='ignore')
                lines     = [l.strip() for l in text.splitlines() if l.strip()]
                if not lines:
                    continue

                exp_id = None
                for l in lines:
                    m = re.search(r'Experiment ID:\s*(\d+)', l)
                    if m:
                        exp_id = int(m.group(1))

                epoch_lines = [l for l in lines if re.match(r'Epoch\s+\d+/\d+', l)]
                if not epoch_lines:
                    continue

                last = epoch_lines[-1]
                m = re.match(r'Epoch\s+(\d+)/(\d+)\s+loss=([\d.]+)\s+val_acc=([\d.]+)%', last)
                if not m:
                    continue

                cur_ep   = int(m.group(1))
                total_ep = int(m.group(2))
                val_acc  = float(m.group(4))

                durations = [int(dm.group(1)) for l in lines
                             for dm in [re.search(r'\((\d+)s\)', l)] if dm]
                avg_epoch_s = int(sum(durations[-3:]) / len(durations[-3:])) if durations else None
                eta_min = round(avg_epoch_s * (total_ep - cur_ep) / 60) if avg_epoch_s else None

                if not (cur_ep < total_ep and age_hours < 1.0):
                    continue

                jobs.append({
                    'type':         'training',
                    'config':       log_file.stem.replace('train_', ''),
                    'exp_id':       exp_id,
                    'epoch':        cur_ep,
                    'total_epochs': total_ep,
                    'val_acc':      val_acc,
                    'eta_min':      eta_min,
                    'status':       'running',
                })
            except Exception:
                continue

    return jsonify({'jobs': jobs})
