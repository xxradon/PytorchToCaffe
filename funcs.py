from __future__ import print_function
import numpy as np
import time

class Logger():
    def __init__(self,file_name=None,show=True):
        self.show=show
        self.file_name=file_name

    def __call__(self,str):
        str='%s  '%(time.strftime('%H:%M:%S'),)+str
        if self.file_name:
            with open(self.file_name,'a+') as f:
                f.write(str+'\n')
        if self.show:
            print(str)

def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]

def get_iou(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
        The box should be [x1,y1,x2,y2]
    Args:
        box_a: Single numpy bounding box, Shape: [4] or Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single numpy bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    if box_a.ndim==1:
        box_a=box_a.reshape([1,-1])
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]

def nms(bboxs,scores,thresh):
    """
    The box should be [x1,y1,x2,y2]
    :param bboxs: multiple bounding boxes, Shape: [num_boxes,4]
    :param scores: The score for the corresponding box
    :return: keep inds
    """
    if len(bboxs)==0:
        return []
    order=scores.argsort()[::-1]
    keep=[]
    while order.size>0:
        i=order[0]
        keep.append(i)
        ious=get_iou(bboxs[order],bboxs[i])
        order=order[ious<=thresh]
    return keep
