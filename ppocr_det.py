import os
import sys
import cv2
import numpy as np
import argparse
import utils.operators
from utils.db_postprocess import DBPostProcess, DetPostProcess

# add path
realpath = os.path.abspath(__file__)
_sep = os.path.sep
realpath = realpath.split(_sep)
sys.path.append(os.path.join(realpath[0]+_sep, *realpath[1:realpath.index('rknn_model_zoo')+1]))


DET_INPUT_SHAPE = [480, 480] # h,w

ONNX_PRE_PROCESS_CONFIG = [
        {
            'DetResizeForTest': 
            {
                'limit_side_len': 480,
                'limit_type': 'max',
            }
        }, 
        {
            'NormalizeImage': {
                'std': [0.229, 0.224, 0.225],
                'mean': [0.485, 0.456, 0.406],
                'scale': '1./255.',
                'order': 'hwc'
            }
        }, 
        ]

RKNN_PRE_PROCESS_CONFIG = [
        {
            'DetResizeForTest': {
                    'image_shape': DET_INPUT_SHAPE
                }
         }, 
        {
            'NormalizeImage': 
            {
                    'std': [1., 1., 1.],
                    'mean': [0., 0., 0.],
                    'scale': '1.',
                    'order': 'hwc'
            }
        }
        ]

POSTPROCESS_CONFIG = {
    'DBPostProcess':{
        'thresh': 0.3,
        'box_thresh': 0.6,
        'max_candidates': 1000,
        'unclip_ratio': 1.5,
        'use_dilation': False,
        'score_mode': 'fast',
    }
}

class TextDetector:
    def __init__(self, args) -> None:
        self.model, self.framework = setup_model(args)
        self.preprocess_funct = []
        PRE_PROCESS_CONFIG = ONNX_PRE_PROCESS_CONFIG if self.framework == 'onnx' else RKNN_PRE_PROCESS_CONFIG
        for item in PRE_PROCESS_CONFIG:
            for key in item:
                pclass = getattr(utils.operators, key)
                p = pclass(**item[key])
                self.preprocess_funct.append(p)

        self.db_postprocess = DBPostProcess(**POSTPROCESS_CONFIG['DBPostProcess'])
        self.det_postprocess = DetPostProcess()

    def preprocess(self, img):
        for p in self.preprocess_funct:
            img = p(img)

        if self.framework == 'onnx':
            image_input = img['image']
            image_input = image_input.reshape(1, *image_input.shape)
            image_input = image_input.transpose(0, 3, 1, 2)
            img['image'] = image_input
        return img

    def run(self, img):
        model_input = self.preprocess({'image':img})
        output = self.model.run([model_input['image']])

        preds = {'maps' : output[0].astype(np.float32)}
        result = self.db_postprocess(preds, model_input['shape'])

        output = self.det_postprocess.filter_tag_det_res(result[0]['points'], img.shape)
        return output

def setup_model(args):
    model_path = args.det_model_path
    if model_path.endswith('.rknn'):
        platform = 'rknn'
        from py_utils.rknn_executor import RKNN_model_container 
        model = RKNN_model_container(model_path, args.target, args.device_id)
    elif model_path.endswith('onnx'):
        platform = 'onnx'
        from py_utils.onnx_executor import ONNX_model_container
        model = ONNX_model_container(model_path)
    else:
        assert False, "{} is not rknn/onnx model".format(model_path)
    print('Model-{} is {} model, starting val'.format(model_path, platform))
    return model, platform
