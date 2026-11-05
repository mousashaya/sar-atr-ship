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


@bp.route('/ships')
def ships():
    filters = db.get_shiprs_filters()
    return render_template('ships.html', **filters)


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


# ── ShipRS image serving ──────────────────────────────────────────────────────

@bp.route('/shiprs-image/<int:image_id>')
def serve_shiprs_image(image_id):
    """Serve a ShipRS image chip cropped to the instance bounding box."""
    from PIL import Image as PILImage
    import io
    size = min(int(request.args.get('size', 256)), 768)
    row = db.get_shiprs_instance(image_id)
    if not row or not row.get('local_path'):
        abort(404)
    path = Path(row['local_path'])
    if not path.exists():
        abort(404)
    try:
        img = PILImage.open(path).convert('RGB')
        cx, cy = row.get('cx'), row.get('cy')
        bw, bh = row.get('bbox_w'), row.get('bbox_h')
        if cx and cy and bw and bh:
            pad = max(20, int(max(bw, bh) * 0.15))
            x1 = max(0, int(cx - bw/2) - pad)
            y1 = max(0, int(cy - bh/2) - pad)
            x2 = min(img.width,  int(cx + bw/2) + pad)
            y2 = min(img.height, int(cy + bh/2) + pad)
            img = img.crop((x1, y1, x2, y2))
        # Pad to square to preserve aspect ratio
        w, h = img.size
        side = max(w, h)
        square = PILImage.new('RGB', (side, side), (20, 20, 20))
        square.paste(img, ((side - w) // 2, (side - h) // 2))
        square = square.resize((size, size), PILImage.LANCZOS)
        buf = io.BytesIO()
        square.save(buf, format='JPEG', quality=92)
        buf.seek(0)
        return send_file(buf, mimetype='image/jpeg')
    except Exception:
        abort(500)


@bp.route('/api/shiprs/images')
def api_shiprs_images():
    level_1 = request.args.get('level_1', '')
    level_2 = request.args.get('level_2', '')
    level_3 = request.args.get('level_3', '')
    split   = request.args.get('split', '')
    limit   = int(request.args.get('limit', 200))
    offset  = int(request.args.get('offset', 0))
    images, total = db.get_shiprs_image_list(level_1, level_2, level_3, split, limit, offset)
    return jsonify({'count': len(images), 'total': total, 'images': images})


@bp.route('/api/shiprs/instance/<int:instance_id>')
def api_shiprs_instance(instance_id):
    row = db.get_shiprs_instance(instance_id)
    if not row:
        return jsonify({'error': 'Not found'}), 404
    return jsonify({k: (str(v) if hasattr(v, 'isoformat') else v) for k, v in row.items()})


@bp.route('/api/shiprs/filters')
def api_shiprs_filters():
    filters = db.get_shiprs_filters()
    return jsonify(filters)


# ── OpenSARShip routes ────────────────────────────────────────────────────────

@bp.route('/opensar')
def opensar():
    filters = db.get_opensar_filters()
    return render_template('opensar.html', **filters)


@bp.route('/opensar-image/<int:image_id>')
def serve_opensar_image(image_id):
    from PIL import Image as PILImage
    import io
    row = db.get_opensar_image(image_id)
    if not row or not row.get('local_path'):
        abort(404)
    path = Path(row['local_path'])
    if not path.exists():
        abort(404)
    try:
        img = PILImage.open(path).convert('RGB')
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=92)
        buf.seek(0)
        return send_file(buf, mimetype='image/jpeg')
    except Exception:
        abort(500)


PATCH_CAL_ROOT = Path('/Volumes/Machine Learning/opensar_ship/OpenSARShip2')
DB_MIN, DB_MAX = -30.0, 10.0


def _sar_border_suppress(arr_2d, border=10, k=3.0):
    import numpy as np
    h, w = arr_2d.shape
    if h < 2 * border + 5 or w < 2 * border + 5:
        return arr_2d
    intensity = arr_2d ** 2
    mask = np.ones((h, w), dtype=bool)
    mask[border:h - border, border:w - border] = False
    bg = intensity[mask]
    thresh = float((bg.mean() + k * bg.std()) ** 0.5)
    return (arr_2d * (arr_2d > thresh)).astype(arr_2d.dtype)


def _tif_channel_to_png(tif_path, pol):
    """Read Patch_Cal TIFF and return an 8-bit grayscale PIL Image for one pol."""
    import numpy as np
    import tifffile
    from PIL import Image as PILImage
    arr = tifffile.imread(str(tif_path))
    if arr.ndim == 2:
        ch = arr
    else:
        ch = arr[:, :, 0 if pol == 'VV' else 1]
    db = 10.0 * np.log10(np.maximum(ch, 1e-6))
    db = np.clip(db, DB_MIN, DB_MAX)
    px = ((db - DB_MIN) / (DB_MAX - DB_MIN) * 255).astype('uint8')
    return PILImage.fromarray(px, mode='L').convert('RGB')


def _reconstruct_tif_path(row):
    parts = row['image_id'].split('__', 1)
    if len(parts) < 2:
        return None
    pol      = row.get('polarization', 'VV')
    stem_pol = parts[1]
    stem     = stem_pol.rsplit(f'_{pol}', 1)[0]
    return PATCH_CAL_ROOT / row['scene_id'] / 'Patch_Cal' / f'{stem}.tif'


@bp.route('/opensar-image-clean/<int:image_id>')
def serve_opensar_image_clean(image_id):
    import numpy as np
    import tifffile
    import io
    from PIL import Image as PILImage

    row = db.get_opensar_image(image_id)
    if not row:
        abort(404)
    tif_path = _reconstruct_tif_path(row)
    if not tif_path or not tif_path.exists():
        abort(404)
    try:
        arr = tifffile.imread(str(tif_path))
        if arr.ndim == 2:
            ch = arr
        else:
            ch = arr[:, :, 0 if row.get('polarization') == 'VV' else 1]
        ch  = _sar_border_suppress(ch)
        db_v = 10.0 * np.log10(np.maximum(ch, 1e-6))
        db_v = np.clip(db_v, DB_MIN, DB_MAX)
        px   = ((db_v - DB_MIN) / (DB_MAX - DB_MIN) * 255).astype('uint8')
        img  = PILImage.fromarray(px, mode='L').convert('RGB')
        buf  = io.BytesIO()
        img.save(buf, format='JPEG', quality=92)
        buf.seek(0)
        return send_file(buf, mimetype='image/jpeg')
    except Exception:
        abort(500)


@bp.route('/api/opensar/images')
def api_opensar_images():
    images, total = db.get_opensar_image_list(
        ship_type   = request.args.get('ship_type', ''),
        polarization= request.args.get('polarization', ''),
        limit       = int(request.args.get('limit', 200)),
        offset      = int(request.args.get('offset', 0)),
    )
    return jsonify({'count': len(images), 'total': total, 'images': images})


@bp.route('/api/opensar/image/<int:image_id>')
def api_opensar_image(image_id):
    row = db.get_opensar_image(image_id)
    if not row:
        return jsonify({'error': 'Not found'}), 404
    return jsonify({k: (str(v) if hasattr(v, 'isoformat') else (v if v is not None else ''))
                    for k, v in row.items()})


@bp.route('/api/opensar/filters')
def api_opensar_filters():
    return jsonify(db.get_opensar_filters())


# ── MSTAR routes ──────────────────────────────────────────────────────────────

PNG_ROOT = Path('/Volumes/Machine Learning/mstar/chips_png')

@bp.route('/mstar')
def mstar():
    filters = db.get_mstar_filters()
    return render_template('mstar.html', **filters)


@bp.route('/mstar-image/<int:image_id>')
def serve_mstar_image(image_id):
    row = db.get_mstar_image(image_id)
    if not row:
        abort(404)
    # Serve pre-converted PNG from chips_png/
    target = row.get('target_short', '')
    dep    = row.get('desired_depression') or 0
    stem   = Path(row.get('filename', '')).stem
    png    = PNG_ROOT / target / f'{int(dep):02d}deg' / (stem + '.png')
    if png.exists():
        return send_file(png, mimetype='image/png')
    # Fallback: render from binary on the fly
    from PIL import Image as PILImage
    import io
    path = Path(row.get('local_path', ''))
    if not path.exists():
        abort(404)
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
        from ingest_mstar import read_mstar, mag_to_uint8
        _, mag = read_mstar(path)
        if mag is None:
            abort(500)
        img = PILImage.fromarray(mag_to_uint8(mag), mode='L').convert('RGB')
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        return send_file(buf, mimetype='image/png')
    except Exception:
        abort(500)


@bp.route('/api/mstar/images')
def api_mstar_images():
    images, total = db.get_mstar_image_list(
        target     = request.args.get('target', ''),
        depression = request.args.get('depression', ''),
        split      = request.args.get('split', ''),
        limit      = int(request.args.get('limit', 200)),
        offset     = int(request.args.get('offset', 0)),
    )
    return jsonify({'count': len(images), 'total': total, 'images': images})


@bp.route('/api/mstar/image/<int:image_id>')
def api_mstar_image(image_id):
    row = db.get_mstar_image(image_id)
    if not row:
        return jsonify({'error': 'Not found'}), 404
    return jsonify({k: (str(v) if hasattr(v, 'isoformat') else (v if v is not None else ''))
                    for k, v in row.items()})


@bp.route('/api/mstar/filters')
def api_mstar_filters():
    return jsonify(db.get_mstar_filters())


# ── HRSID routes ──────────────────────────────────────────────────────────────

@bp.route('/hrsid')
def hrsid():
    filters = db.get_hrsid_filters()
    return render_template('hrsid.html', **filters)


@bp.route('/hrsid-image/<int:chip_id>')
def serve_hrsid_image(chip_id):
    row = db.get_hrsid_chip(chip_id)
    if not row or not row.get('local_path'):
        abort(404)
    path = Path(row['local_path'])
    if not path.exists():
        abort(404)
    return send_file(path, mimetype='image/jpeg')


@bp.route('/api/hrsid/chips')
def api_hrsid_chips():
    chips, total = db.get_hrsid_chip_list(
        scene    = request.args.get('scene', ''),
        split    = request.args.get('split', ''),
        min_area = int(request.args.get('min_area', 0)),
        limit    = int(request.args.get('limit', 200)),
        offset   = int(request.args.get('offset', 0)),
    )
    return jsonify({'count': len(chips), 'total': total, 'chips': chips})


@bp.route('/api/hrsid/chip/<int:chip_id>')
def api_hrsid_chip(chip_id):
    row = db.get_hrsid_chip(chip_id)
    if not row:
        return jsonify({'error': 'Not found'}), 404
    return jsonify({k: (str(v) if hasattr(v, 'isoformat') else (v if v is not None else ''))
                    for k, v in row.items()})


@bp.route('/api/hrsid/filters')
def api_hrsid_filters():
    return jsonify(db.get_hrsid_filters())


# ── FUSAR-Ship routes ─────────────────────────────────────────────────────────

@bp.route('/fusar')
def fusar():
    filters = db.get_fusar_filters()
    return render_template('fusar.html', **filters)


@bp.route('/fusar-image/<int:image_id>')
def serve_fusar_image(image_id):
    from PIL import Image as PILImage
    import io
    row = db.get_fusar_image(image_id)
    if not row or not row.get('local_path'):
        abort(404)
    path = Path(row['local_path'])
    if not path.exists():
        abort(404)
    try:
        import tifffile, numpy as np
        arr = tifffile.imread(str(path))
        img = PILImage.fromarray(arr).convert('RGB')
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=92)
        buf.seek(0)
        return send_file(buf, mimetype='image/jpeg')
    except Exception:
        abort(500)


@bp.route('/api/fusar/images')
def api_fusar_images():
    images, total = db.get_fusar_image_list(
        category   = request.args.get('category', ''),
        subcategory= request.args.get('subcategory', ''),
        polar_mode = request.args.get('polar_mode', ''),
        split      = request.args.get('split', ''),
        limit      = int(request.args.get('limit', 200)),
        offset     = int(request.args.get('offset', 0)),
    )
    return jsonify({'count': len(images), 'total': total, 'images': images})


@bp.route('/api/fusar/image/<int:image_id>')
def api_fusar_image(image_id):
    row = db.get_fusar_image(image_id)
    if not row:
        return jsonify({'error': 'Not found'}), 404
    return jsonify({k: (str(v) if hasattr(v, 'isoformat') else (v if v is not None else ''))
                    for k, v in row.items()})


@bp.route('/api/fusar/filters')
def api_fusar_filters():
    return jsonify(db.get_fusar_filters())


@bp.route('/api/fusar/subcategories')
def api_fusar_subcategories():
    return jsonify(db.get_fusar_subcategories(request.args.get('category', '')))


# ── SAR Ship Re-ID routes ─────────────────────────────────────────────────────

import glob as _glob
import numpy as _np

SAR_CHIPS_DIR = Path('/Volumes/Machine Learning/sar_ship_chips')
SAR_CHIPS_GULF_DIR = Path('/Volumes/Machine Learning/sar_ship_chips_gulf')

# In-memory ship index — built once on first request, keyed by MMSI
_sar_index = None

def _build_sar_index():
    """Scan all .npz chip files and build ship-level index."""
    global _sar_index
    ships = {}
    dir_sources = [(SAR_CHIPS_DIR, 'houston')]
    if SAR_CHIPS_GULF_DIR.exists():
        dir_sources.append((SAR_CHIPS_GULF_DIR, 'gulf'))
    all_npz = sorted(
        (p, src) for d, src in dir_sources for p in d.glob('**/*.npz')
    )
    for npz_path, source in all_npz:
        if npz_path.name == 'embeddings.npz':
            continue
        try:
            d = _np.load(npz_path, allow_pickle=True)
            mmsi = int(d['mmsi'])
            chip_key = f"{npz_path.parent.name}/{npz_path.stem}"
            chip = {
                'chip_key':     chip_key,
                'source':       source,
                'acq_time':     str(d['acq_time']),
                'scene_id':     str(d['scene_id']),
                'vessel_name':  str(d['vessel_name']),
                'sog':          float(d['sog']),
                'heading':      float(d['heading']),
                'lat':          float(d['lat']),
                'lon':          float(d['lon']),
                'dt_sec':       float(d['dt_sec']),
                'n_other_large':int(d['n_other_large']) if 'n_other_large' in d else 0,
                'clean':        bool(d['clean']) if 'clean' in d else True,
            }
            if mmsi not in ships:
                ships[mmsi] = {
                    'mmsi':        mmsi,
                    'vessel_name': str(d['vessel_name']),
                    'vessel_type': int(d['vessel_type']) if 'vessel_type' in d else -1,
                    'length_m':    float(d['length_m']) if 'length_m' in d else 0,
                    'passes':           0,
                    'clean_passes':     0,
                    'open_water_passes':0,
                    'max_sog':          0.0,
                    'min_lat':          999.0,
                    'sources':          set(),
                    'chips':            [],
                }
            s = ships[mmsi]
            s['passes'] += 1
            if chip['clean']:
                s['clean_passes'] += 1
            # open water: south of Texas coast and east of Corpus Christi/Matagorda
            if chip['lat'] < 28.5 and chip['lon'] > -94.0:
                s['open_water_passes'] += 1
            s['max_sog'] = max(s['max_sog'], chip['sog'])
            s['min_lat'] = min(s['min_lat'], chip['lat'])
            s['sources'].add(source)
            s['chips'].append(chip)
        except Exception:
            continue
    for s in ships.values():
        s['sources'] = sorted(s['sources'])
    _sar_index = list(ships.values())


@bp.route('/sar-ships')
def sar_ships():
    return render_template('sar_ships.html')


@bp.route('/api/sar/ships')
def api_sar_ships():
    if _sar_index is None:
        _build_sar_index()
    # Return ship summaries without the full chip list (too large)
    summaries = [{k: v for k, v in s.items() if k != 'chips'} for s in _sar_index]
    return jsonify({'ships': summaries, 'total': len(summaries)})


@bp.route('/api/sar/ship/<int:mmsi>')
def api_sar_ship(mmsi):
    if _sar_index is None:
        _build_sar_index()
    ship = next((s for s in _sar_index if s['mmsi'] == mmsi), None)
    if not ship:
        return jsonify({'error': 'Not found'}), 404
    return jsonify({'mmsi': mmsi, 'chips': ship['chips']})


@bp.route('/sar-chip/<scene_id>/<stem>')
def serve_sar_chip(scene_id, stem):
    """Serve a SAR chip from .npz as a grayscale JPEG (dB scale, VV pol).

    ?size=256   output pixel size (default 256)
    ?zoom=1     if set, center-crop inner half before resizing (ship fills frame)
    """
    from PIL import Image as PILImage
    import io as _io
    size = min(int(request.args.get('size', 256)), 512)
    zoom = request.args.get('zoom', '0') not in ('0', 'false', '')
    npz_path = SAR_CHIPS_DIR / scene_id / (stem + '.npz')
    if not npz_path.exists():
        npz_path = SAR_CHIPS_GULF_DIR / scene_id / (stem + '.npz')
    if not npz_path.exists():
        abort(404)
    try:
        d = _np.load(npz_path, allow_pickle=True)
        chip = d['chip_vv'].astype(_np.float32)
        # Optional center crop: inner 256px of 512px chip → tight ship view
        if zoom and chip.shape[0] == 512:
            q = chip.shape[0] // 4   # 128
            chip = chip[q:chip.shape[0]-q, q:chip.shape[1]-q]  # 256×256
        # dB scale
        db = 10 * _np.log10(chip + 1.0)

        # Block-wise background subtraction — estimate local clutter per
        # 8×8 block using the block median, then subtract spatially.
        # Handles uneven sea state / wind gradients across the chip.
        h, w = db.shape
        bg = _np.zeros_like(db)
        n_blocks = 8
        bh, bw = h // n_blocks, w // n_blocks
        for bi in range(n_blocks):
            for bj in range(n_blocks):
                block = db[bi*bh:(bi+1)*bh, bj*bw:(bj+1)*bw]
                bg[bi*bh:(bi+1)*bh, bj*bw:(bj+1)*bw] = _np.median(block)
        db_sub = _np.maximum(0, db - bg)   # ship signal above clutter floor

        # Stretch the residual signal
        if db_sub.max() > 0:
            p98 = _np.percentile(db_sub[db_sub > 0], 98)
            px  = _np.clip(db_sub / max(p98, 1e-6), 0, 1)
        else:
            px = db_sub
        px = (px * 255).astype(_np.uint8)
        img = PILImage.fromarray(px, mode='L').convert('RGB')
        if size != img.width:
            img = img.resize((size, size), PILImage.LANCZOS)

        # Draw a small box around the ship position (doesn't obscure the signal)
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        cx, cy = size // 2, size // 2
        box_r = max(28, size // 6)   # half-size of box
        lw    = max(1, size // 128)
        color = (255, 80, 80)        # red
        draw.rectangle([(cx - box_r, cy - box_r), (cx + box_r, cy + box_r)],
                       outline=color, width=lw)

        buf = _io.BytesIO()
        img.save(buf, format='JPEG', quality=92)
        buf.seek(0)
        return send_file(buf, mimetype='image/jpeg')
    except Exception:
        abort(500)


@bp.route('/api/sar/refresh')
def api_sar_refresh():
    """Force rebuild of the in-memory chip index (call after running chip_ships_sar.py)."""
    global _sar_index
    _sar_index = None
    _build_sar_index()
    return jsonify({'ships': len(_sar_index)})


# ── Umbra Sub-Meter SAR Ship routes ───────────────────────────────────────────

UMBRA_CHIP_DIRS = {
    'la':         Path('/Volumes/Machine Learning/umbra_ship_chips_la'),
    'ny':         Path('/Volumes/Machine Learning/umbra_ship_chips_ny'),
    'savannah':   Path('/Volumes/Machine Learning/umbra_ship_chips_savannah'),
    'chesapeake': Path('/Volumes/Machine Learning/umbra_ship_chips_chesapeake'),
    'puget':      Path('/Volumes/Machine Learning/umbra_ship_chips_puget'),
}

_umbra_index = None


def _build_umbra_index():
    global _umbra_index
    ships = {}
    for region, chip_dir in UMBRA_CHIP_DIRS.items():
        if not chip_dir.exists():
            continue
        for npz_path in sorted(chip_dir.glob('**/*.npz')):
            try:
                d = _np.load(npz_path, allow_pickle=True)
                mmsi = int(d['mmsi'])
                scene_id = npz_path.parent.name
                chip_key = f"{region}/{scene_id}/{npz_path.stem}"
                chip = {
                    'chip_key':    chip_key,
                    'region':      region,
                    'scene_id':    scene_id,
                    'acq_time':    str(d['acq_time']),
                    'vessel_name': str(d['vessel_name']),
                    'sog':         float(d['sog']),
                    'heading':     float(d['heading']),
                    'lat':         float(d['lat']),
                    'lon':         float(d['lon']),
                    'dt_sec':      float(d['dt_sec']),
                    'n_other_large': int(d['n_other_large']) if 'n_other_large' in d else 0,
                    'clean':       bool(d['clean']) if 'clean' in d else True,
                }
                if mmsi not in ships:
                    ships[mmsi] = {
                        'mmsi':         mmsi,
                        'vessel_name':  str(d['vessel_name']),
                        'vessel_type':  int(d['vessel_type']) if 'vessel_type' in d else -1,
                        'length_m':     float(d['length_m']) if 'length_m' in d else 0,
                        'passes':            0,
                        'clean_passes':      0,
                        'max_sog':           0.0,
                        'regions':           set(),
                        'chips':             [],
                    }
                s = ships[mmsi]
                s['passes'] += 1
                if chip['clean']:
                    s['clean_passes'] += 1
                s['max_sog'] = max(s['max_sog'], chip['sog'])
                s['regions'].add(region)
                s['chips'].append(chip)
            except Exception:
                continue
    for s in ships.values():
        s['regions'] = sorted(s['regions'])
    _umbra_index = list(ships.values())


@bp.route('/umbra-ships')
def umbra_ships():
    return render_template('umbra_ships.html')


@bp.route('/api/umbra/labels')
def api_umbra_labels():
    """Resolved AIS labels for all Umbra vessels."""
    labels_path = Path(__file__).parent.parent / 'data' / 'resolved_umbra_labels.json'
    if not labels_path.exists():
        return jsonify({})
    import json as _json
    with open(labels_path) as f:
        return jsonify(_json.load(f))


@bp.route('/umbra-chip/<region>/<scene_id>/<stem>')
def serve_umbra_chip(region, scene_id, stem):
    """Serve an Umbra ship chip as JPEG.

    Rendering: log-compress → 3×3 speckle filter → adaptive percentile stretch.
    The ship stands out bright against gray sea/port clutter (no background
    subtraction, which removed spatial context and made chips unreadable).
    """
    import io as _io
    from PIL import Image as PILImage, ImageDraw
    from scipy.ndimage import uniform_filter

    size      = min(int(request.args.get('size', 256)), 1024)
    zoom      = request.args.get('zoom', '0') not in ('0', 'false', '')
    length_m  = float(request.args.get('length', 0) or 0)

    chip_dir = UMBRA_CHIP_DIRS.get(region)
    if not chip_dir:
        abort(404)
    npz_path = chip_dir / scene_id / (stem + '.npz')
    if not npz_path.exists():
        abort(404)
    try:
        d = _np.load(npz_path, allow_pickle=True)
        chip = d['chip'].astype(_np.float32)

        # Adaptive center crop based on vessel length (2.5× length context window)
        # Falls back to inner-half crop when length unknown.
        # At ~0.5m GSD: 1px ≈ 0.5m → length_m / 0.5 = ship pixels
        if zoom and chip.shape[0] >= 512:
            if length_m > 0:
                crop_m  = length_m * 2.5
                crop_px = min(chip.shape[0], max(128, int(crop_m / 0.5)))
            else:
                crop_px = chip.shape[0] // 2      # default: inner half
            margin = (chip.shape[0] - crop_px) // 2
            chip = chip[margin:margin+crop_px, margin:margin+crop_px]

        # Speckle reduction: 3×3 boxcar filter on amplitude (fast, preserves edges)
        chip_filt = uniform_filter(chip, size=3)

        # Log-compress: maps [1, 255] → [0, 5.5]; ships still much brighter than sea
        lc = _np.log1p(chip_filt)

        # Adaptive percentile stretch — preserves sea/port context (no BG subtraction)
        p2,  p98 = _np.percentile(lc, [2, 98])
        px = _np.clip((lc - p2) / max(p98 - p2, 1e-6), 0, 1)

        # Mild gamma boost (0.75) so midtone sea clutter is visible gray, not black
        px = px ** 0.75
        px = (px * 255).astype(_np.uint8)

        img = PILImage.fromarray(px, mode='L').convert('RGB')
        if size != img.width:
            img = img.resize((size, size), PILImage.LANCZOS)

        draw = ImageDraw.Draw(img)

        # Scale bar: bottom-left corner
        scene_m  = chip.shape[0] * 0.5 if zoom else 1024   # actual crop in meters
        m_per_px = scene_m / size
        bar_m    = 100 if scene_m >= 512 else 50
        bar_px   = int(bar_m / m_per_px)
        bx0, by  = 10, size - 18
        draw.rectangle([(bx0, by), (bx0 + bar_px, by + 5)], fill=(255, 200, 0))
        try:
            from PIL import ImageFont
            font = ImageFont.load_default()
            draw.text((bx0, by - 12), f'{bar_m}m', fill=(255, 200, 0), font=font)
        except Exception:
            pass

        # Red crosshair at ship center (smaller than the old box — less obtrusive)
        cx, cy = size // 2, size // 2
        arm   = max(14, size // 12)
        lw    = max(1, size // 192)
        color = (255, 60, 60)
        draw.line([(cx - arm, cy), (cx + arm, cy)], fill=color, width=lw)
        draw.line([(cx, cy - arm), (cx, cy + arm)], fill=color, width=lw)

        buf = _io.BytesIO()
        img.save(buf, format='JPEG', quality=92)
        buf.seek(0)
        return send_file(buf, mimetype='image/jpeg')
    except Exception:
        abort(500)


@bp.route('/api/umbra/ships')
def api_umbra_ships():
    if _umbra_index is None:
        _build_umbra_index()
    summaries = [{k: v for k, v in s.items() if k != 'chips'} for s in _umbra_index]
    return jsonify({'ships': summaries, 'total': len(summaries)})


@bp.route('/api/umbra/ship/<int:mmsi>')
def api_umbra_ship(mmsi):
    if _umbra_index is None:
        _build_umbra_index()
    ship = next((s for s in _umbra_index if s['mmsi'] == mmsi), None)
    if not ship:
        return jsonify({'error': 'Not found'}), 404
    return jsonify({'mmsi': mmsi, 'chips': ship['chips']})


@bp.route('/api/umbra/refresh')
def api_umbra_refresh():
    global _umbra_index
    _umbra_index = None
    _build_umbra_index()
    return jsonify({'ships': len(_umbra_index)})
