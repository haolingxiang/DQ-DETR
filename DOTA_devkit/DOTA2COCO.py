from math import pi

import numpy as np

import dota_utils as util
import os
import cv2
import json



wordname_18 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship',
               'tennis-court',
               'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool',
               'helicopter',
               'airport', 'container-crane', 'helipad']


def regular_theta(theta, mode='180', start=-pi / 2):
    """
    limit theta  ∈ [-pi/2, pi/2)
    """
    assert mode in ['360', '180']
    cycle = 2 * pi if mode == '360' else pi

    theta = theta - start
    theta = theta % cycle
    return theta + start


def poly2angle(poly):
    poly = np.array(poly)
    poly = np.float32(poly.reshape(4, 2))
    (x, y), (w, h), angle = cv2.minAreaRect(poly)  # θ ∈ [0， 90]
    angle = -angle  # θ ∈ [-90， 0]
    theta = angle / 180 * pi  # 转为pi制

    # trans opencv format to longedge format θ ∈ [-pi/2， pi/2]
    if w != max(w, h):
        w, h = h, w
        theta += pi / 2
    theta = regular_theta(theta)  # limit theta ∈ [-pi/2, pi/2)
    angle = (theta * 180 / pi) + 90  # θ ∈ [0， 180)
    return x, y, w, h, angle


def DOTA2COCO(srcpath, destfile):
    imageparent = os.path.join(srcpath, 'images')
    labelparent = os.path.join(srcpath, 'labelTxt')

    data_dict = {}
    info = {'contributor': 'sean hao',
            'data_created': '2025',
            'description': 'This is 1.0 version of DOTA dataset.',
            'url': '',
            'version': '1.0',
            'year': 2025}
    data_dict['info'] = info
    data_dict['images'] = []
    data_dict['categories'] = []
    data_dict['annotations'] = []
    for idex, name in enumerate(wordname_18):
        single_cat = {'id': idex + 1, 'name': name, 'supercategory': name}
        data_dict['categories'].append(single_cat)

    inst_count = 1
    image_id = 1
    with open(destfile, 'w') as f_out:
        filenames = util.GetFileFromThisRootDir(labelparent)
        for file in filenames:
            basename = util.custombasename(file)
            # image_id = int(basename[1:])

            imagepath = os.path.join(imageparent, basename + '.png')
            img = cv2.imread(imagepath)
            height, width, c = img.shape

            single_image = {}
            single_image['file_name'] = basename + '.png'
            single_image['id'] = image_id
            single_image['width'] = width
            single_image['height'] = height
            data_dict['images'].append(single_image)

            # annotations
            objects = util.parse_dota_poly2(file)
            for obj in objects:
                single_obj = {}
                single_obj['area'] = obj['area']
                single_obj['category_id'] = wordname_18.index(obj['name']) + 1
                single_obj['segmentation'] = []
                # single_obj['segmentation'].append(obj['poly'])
                single_obj['iscrowd'] = 0
                # xmin, ymin, xmax, ymax = min(obj['poly'][0::2]), min(obj['poly'][1::2]), \
                #     max(obj['poly'][0::2]), max(obj['poly'][1::2])

                # width, height = xmax - xmin, ymax - ymin
                x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float,obj['poly']))
                x_center, y_center, w, h, angle = poly2angle([x1, y1, x2, y2, x3, y3, x4, y4])
                single_obj['bbox'] = [x1, y1, x2, y2, x3, y3, x4, y4]
                single_obj['rbox'] = [x_center, y_center, w, h, angle]
                single_obj['image_id'] = image_id
                data_dict['annotations'].append(single_obj)
                single_obj['id'] = inst_count
                inst_count = inst_count + 1
            image_id = image_id + 1
        json.dump(data_dict, f_out)


if __name__ == '__main__':
    DOTA2COCO(r'E:\RSimages_objectdetection_transformer\DQ-DETR\Dataset\dota\val',
              r'E:\RSimages_objectdetection_transformer\DQ-DETR\Dataset\dota\annotations\instances_val.json')
