"""
scripts/ingest_shiprs.py
========================
Ingests ShipRSImageNet V1 into defense.shiprs_images.

Reads VOC-format XML annotations (one per image, multiple ship instances per image).
Uses COCO JSON category lists to build level_0–level_3 name mappings.
Uses ImageSets train/val/test splits.

Usage:
    python3 scripts/ingest_shiprs.py

Data expected at:
    /Volumes/Machine Learning/ShipRSImageNet/ShipRSImageNet_V1/
"""

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import psycopg2
import psycopg2.extras

sys.path.insert(0, str(Path(__file__).parent))
from db_config import DB_CONFIG

BASE = Path('/Volumes/Machine Learning/ShipRSImageNet/ShipRSImageNet_V1')
VOC  = BASE / 'VOC_Format'
COCO = BASE / 'COCO_Format'
IMGS = VOC / 'JPEGImages'
ANNS = VOC / 'Annotations'


def load_category_map(level: int) -> dict:
    """Returns {id: name} for the given hierarchy level."""
    candidates = list(COCO.glob(f'*train*level_{level}*.json'))
    if not candidates:
        raise FileNotFoundError(f'No COCO JSON for level {level}')
    d = json.loads(candidates[0].read_text())
    return {c['id']: c['name'] for c in d['categories']}


def load_splits() -> dict:
    """Returns {image_stem: split} for train/val/test."""
    result = {}
    for split in ('train', 'val', 'test'):
        for txt in (VOC / 'ImageSets').glob(f'{split}.txt'):
            for line in txt.read_text().splitlines():
                stem = Path(line.strip()).stem  # strip .bmp if present
                if stem:
                    result[stem] = split
    return result


def parse_xml(xml_path: Path, splits: dict, cat_maps: dict) -> list[dict]:
    """Parse one VOC XML and return a list of instance dicts."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.findtext('filename', default=xml_path.stem + '.bmp')
    stem     = Path(filename).stem
    split    = splits.get(stem, 'train')
    local_path = str(IMGS / filename)

    size = root.find('size')
    img_w = int(size.findtext('width',  '0')) if size is not None else 0
    img_h = int(size.findtext('height', '0')) if size is not None else 0

    instances = []
    for idx, obj in enumerate(root.findall('object')):
        l0 = int(obj.findtext('level_0', '0'))
        l1 = int(obj.findtext('level_1', '0'))
        l2 = int(obj.findtext('level_2', '0'))
        l3 = int(obj.findtext('level_3', '0'))

        env  = obj.find('Ship_env')
        rbox = obj.find('rotated_box')

        instances.append({
            'image_file':   filename,
            'local_path':   local_path,
            'split':        split,
            'instance_id':  idx,
            'level_0':      cat_maps[0].get(l0),
            'level_1':      cat_maps[1].get(l1),
            'level_2':      cat_maps[2].get(l2),
            'level_3':      cat_maps[3].get(l3) or obj.findtext('name'),
            'ship_size':    obj.findtext('Ship_size'),
            'ship_location':obj.findtext('Ship_location'),
            'env_normal':   env is not None and env.findtext('Ship_env_is_normal') == '1',
            'env_fog':      env is not None and env.findtext('Ship_env_is_fog')    == '1',
            'env_glint':    env is not None and env.findtext('Ship_env_is_glint')  == '1',
            'env_dark':     env is not None and env.findtext('Ship_env_is_dark')   == '1',
            'is_crowd':     obj.findtext('iscrowd') == '1',
            'difficult':    obj.findtext('difficult') == '1',
            'cx':           float(rbox.findtext('cx',  '0')) if rbox is not None else None,
            'cy':           float(rbox.findtext('cy',  '0')) if rbox is not None else None,
            'bbox_w':       float(rbox.findtext('width',  '0')) if rbox is not None else None,
            'bbox_h':       float(rbox.findtext('height', '0')) if rbox is not None else None,
            'bbox_rot':     float(rbox.findtext('rot', '0'))    if rbox is not None else None,
            'image_width':  img_w,
            'image_height': img_h,
        })
    return instances


def main():
    print('Loading category maps...')
    cat_maps = {lvl: load_category_map(lvl) for lvl in range(4)}
    for lvl, m in cat_maps.items():
        print(f'  Level {lvl}: {len(m)} categories')

    print('Loading splits...')
    splits = load_splits()
    print(f'  {len(splits)} images assigned to splits')

    xml_files = sorted(ANNS.glob('*.xml'))
    print(f'Parsing {len(xml_files)} XML files...')

    conn = psycopg2.connect(**DB_CONFIG)
    cur  = conn.cursor()

    # Clear existing data
    cur.execute('DELETE FROM defense.shiprs_images')
    conn.commit()

    INSERT = """
        INSERT INTO defense.shiprs_images
            (image_file, local_path, split, instance_id,
             level_0, level_1, level_2, level_3,
             ship_size, ship_location,
             env_normal, env_fog, env_glint, env_dark,
             is_crowd, difficult,
             cx, cy, bbox_w, bbox_h, bbox_rot,
             image_width, image_height)
        VALUES
            (%(image_file)s, %(local_path)s, %(split)s, %(instance_id)s,
             %(level_0)s, %(level_1)s, %(level_2)s, %(level_3)s,
             %(ship_size)s, %(ship_location)s,
             %(env_normal)s, %(env_fog)s, %(env_glint)s, %(env_dark)s,
             %(is_crowd)s, %(difficult)s,
             %(cx)s, %(cy)s, %(bbox_w)s, %(bbox_h)s, %(bbox_rot)s,
             %(image_width)s, %(image_height)s)
    """

    total = 0
    batch = []
    for i, xml_path in enumerate(xml_files):
        instances = parse_xml(xml_path, splits, cat_maps)
        batch.extend(instances)
        if len(batch) >= 500:
            psycopg2.extras.execute_batch(cur, INSERT, batch)
            conn.commit()
            total += len(batch)
            batch = []
        if (i + 1) % 500 == 0:
            print(f'  {i+1}/{len(xml_files)} images processed, {total} instances inserted')

    if batch:
        psycopg2.extras.execute_batch(cur, INSERT, batch)
        conn.commit()
        total += len(batch)

    conn.close()
    print(f'\nDone. {total} ship instances inserted into defense.shiprs_images.')

    # Summary
    conn = psycopg2.connect(**DB_CONFIG)
    cur  = conn.cursor()
    cur.execute('''
        SELECT split, level_1, count(*) as n
        FROM defense.shiprs_images
        GROUP BY split, level_1
        ORDER BY split, level_1
    ''')
    print(f'\n{"Split":<8} {"Level 1":<20} {"Count":>7}')
    print('-' * 38)
    for row in cur.fetchall():
        print(f'{row[0]:<8} {str(row[1]):<20} {row[2]:>7}')
    conn.close()


if __name__ == '__main__':
    main()
