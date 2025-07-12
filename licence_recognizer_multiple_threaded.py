import os
import sys
import cv2
import copy
import numpy as np
import argparse
import ppocr_rec as predict_rec
import ppocr_det as predict_det




os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

DET_INPUT_SHAPE = [860, 860] # h,w

class TextSystem(object):
    def __init__(self, args):
        self.text_detector = predict_det.TextDetector(args)
        self.text_recognizer = predict_rec.TextRecognizer(args)
        self.drop_score = 0.5

    def run(self, img):
        # 1. TextDetector
        ori_im = img.copy()
        dt_boxes = self.text_detector.run(img)
        if dt_boxes is None:
            return None, None

        img_crop_list = []
        dt_boxes = sorted_boxes(dt_boxes)
        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = get_rotate_crop_image(ori_im, tmp_box)
            img_crop_list.append(img_crop)

        # 2. TextRecognizer
        rec_res = self.text_recognizer.run(img_crop_list)
        
        # 3. Filter
        filter_boxes, filter_rec_res = [], []
        for box, rec_result in zip(dt_boxes, rec_res):
            text, score = rec_result[0]
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_result)

        return filter_boxes, filter_rec_res

def get_rotate_crop_image(img, points):
    '''
    img_height, img_width = img.shape[0:2]
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    img_crop = img[top:bottom, left:right, :].copy()
    points[:, 0] = points[:, 0] - left
    points[:, 1] = points[:, 1] - top
    '''
    assert len(points) == 4, "shape of points must be 4*2"
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32([[0, 0], [img_crop_width, 0],
                          [img_crop_width, img_crop_height],
                          [0, img_crop_height]])
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M, (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC)
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img

def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        for j in range(i, -1, -1):
            if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and \
                    (_boxes[j + 1][0][0] < _boxes[j][0][0]):
                tmp = _boxes[j]
                _boxes[j] = _boxes[j + 1]
                _boxes[j + 1] = tmp
            else:
                break
    return _boxes


def init_args():
    parser = argparse.ArgumentParser(description='PPOCR-System Python Demo')
    # basic params
    parser.add_argument('--det_model_path', type=str, required= True, help='model path, could be .onnx or .rknn file')
    parser.add_argument('--rec_model_path', type=str, required= True, help='model path, could be .onnx or .rknn file')
    parser.add_argument('--target', type=str, default='rk3566', help='target RKNPU platform')
    parser.add_argument('--device_id', type=str, default=None, help='device id')
    # parser.add_argument('--vis_font_path', type=str, default='../model/simfang.ttf', help='vis font path')
    return parser
    
    
cap = cv2.VideoCapture(20)

    # Init model
parser = init_args()
args =  parser.parse_args()
system_model = TextSystem(args)
    
  
    # Set inputs
 #   img_path = '../model/test.jpg'

while(cap.isOpened()):
    
    # Capture frame-by-frame
    ret, img = cap.read()
    if not ret:
        break
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (DET_INPUT_SHAPE[1], DET_INPUT_SHAPE[0]))

    # Inference
    filter_boxes, filter_rec_res = system_model.run(img)

    print(filter_rec_res)
    cv2.namedWindow('Visor', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('Visor', img)
        # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
 
# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()



