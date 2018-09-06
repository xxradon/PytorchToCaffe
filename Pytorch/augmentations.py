# driven from https://github.com/amdegroot/ssd.pytorch
import torch
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random

def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
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


class SSD_Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        # >>> augmentations.Compose([
        # >>>     transforms.CenterCrop(10),
        # >>>     transforms.ToTensor(),
        # >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels


class SSD_Lambda(object):
    """Applies a lambda as a transform."""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, boxes=None, labels=None):
        return self.lambd(img, boxes, labels)

class ConvertFromInts(object):
    def __call__(self, image, *args):
        image=image.astype(np.float32)
        if len(args):
            return (image, *args)
        else:
            return image


class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, *args):
        image = image.astype(np.float32)
        image -= self.mean
        image=image.astype(np.float32)
        if len(args):
            return (image,)
        else:
            return image


class SSD_ToAbsoluteCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height

        return image, boxes, labels


class SSD_ToPercentCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        return image, boxes, labels


class BGR_2_HSV(object):
    def __call__(self, image, *args):
        HSV_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        if len(args):
            return (HSV_img, *args)
        else:
            return HSV_img

class HSV_2_BGR(object):
    def __call__(self, image, *args):
        BGR_img = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        if len(args):
            return (BGR_img, *args)
        else:
            return BGR_img

class Resize(object):
    def __init__(self, size=300):
        self.size = size

    def __call__(self, image, *args):
        image = cv2.resize(image, (self.size,
                                 self.size))
        if len(args):
            return (image,)
        else:
            return image


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, *args):
        if random.randint(2):
            image[:, :, 1] = (image[:,:,1]*random.uniform(self.lower, self.upper)).astype(np.uint8)
        if len(args):
            return (image, *args)
        else:
            return image

class RandomHue(object):
    def __init__(self, delta=15.0):
        assert delta >= 0.0 and delta <= 255
        self.delta = delta

    def __call__(self, image, *args):
        if random.randint(2):
            image[:, :, 0] += np.uint8(random.uniform(-self.delta, self.delta))
            image[:, :, 0][image[:, :, 0] > 255.0] -= 255
            image[:, :, 0][image[:, :, 0] < 0.0] += 255
        if len(args):
            return (image,)
        else:
            return image

class RandomChannel(object):
    def __init__(self, delta=15.0):
        assert delta >= 0.0 and delta <= 255
        self.delta = delta

    def __call__(self, image, *args):
        channels=image.shape[-1]
        random_switch=np.random.randint(0,2,channels)
        for i in range(channels):
            if random_switch[i]:
                image[:, :, i] += np.uint8(random.uniform(-self.delta, self.delta))
                image[image > 255.0] = 255
                image[image < 0.0] = 0
        if len(args):
            return (image,)
        else:
            return image

class RandomValue(object):
    def __init__(self, delta=15.0):
        # random add or sub a random value in hsv mode
        assert delta>=0.0 and delta<=255.0
        self.delta = int(delta)
    def __call__(self, image, *args):
        if random.randint(2):
            image[:, :, 2] += np.uint8(random.randint(-self.delta, self.delta))
            image[:, :, 2][image[:, :, 2] > 255.0] = 255
            image[:, :, 2][image[:, :, 2] < 0.0] = 0
        if len(args):
            return (image, *args)
        else:
            return image

class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, *args):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        if len(args):
            return (image, *args)
        else:
            return image


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, *args):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        if len(args):
            return (image, *args)
        else:
            return image


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image,*args):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
            image=image.astype(np.uint8)
        if len(args):
            return (image, )
        else:
            return image


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, *args):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        if len(args):
            return (image, *args)
        else:
            return image

class RandomNoise(object):
    def __init__(self,max_noise=3):
        self.max_noise=max_noise
    def __call__(self,image,*args):
        if np.random.randint(2):
            noise=np.random.randint(-self.max_noise,self.max_noise,image.shape)
            image=np.uint8(noise+image)
        if len(args):
            return (image,*args)
        else:
            return image

class ToTensor(object):
    def __call__(self, cvimage, *args):
        image=torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1)
        if len(args):
            return (image, *args)
        else:
            return image


class SSD_RandomSampleCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """
    def __init__(self,sample_options=None):
        if sample_options is None:
            self.sample_options = (
                # using entire original input image
                None,
                # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
                (0.1, None),
                (0.3, None),
                (0.7, None),
                (0.9, None),
                # randomly sample a patch
                (None, None),
            )
        else:
            self.sample_options=sample_options

    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:
                return image, boxes, labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(boxes, rect)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],
                                              :]

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                current_labels = labels[mask]

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                  rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                  rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]

                return current_image, current_boxes, current_labels

class Padding(object):
    def __init__(self, size=1,type='constant',constant_color=(255,255,255)):
        self.size = size
        self.type = type
        self.constant_color=list(constant_color)

    def __call__(self, image, *args):
        size=self.size
        if self.type=='constant':
            image = cv2.copyMakeBorder(image, size, size, size, size, cv2.BORDER_CONSTANT, value=self.constant_color)
        elif self.type=='reflect':
            image = cv2.copyMakeBorder(image, size, size, size, size, cv2.BORDER_REFLECT)
        elif self.type=='replicate':
            image = cv2.copyMakeBorder(image, size, size, size, size, cv2.BORDER_REPLICATE)

        if len(args):
            return (image, *args)
        else:
            return image

class RandomCrop(object):
    def __init__(self,padding):
        self.padding=padding

    def __call__(self,image,*args):
        xi=random.randint(0,self.padding*2)
        yi=random.randint(0,self.padding*2)
        image=image[xi:-(2*self.padding-xi),yi:-(self.padding*2-yi),:]
        if len(args):
            return (image, *args)
        else:
            return image

class Scale(object):
    def __init__(self,dim):
        self.dim=dim
    def __call__(self,image,*args):
        image=cv2.resize(image,self.dim)
        if len(args):
            return (image, *args)
        else:
            return image

class Flip(object):
    def __init__(self,dim=1,percentage=0.5):
        """1 for horizontal flip
        0 for vertical flip
        -1 for both flip"""
        self.dim=dim
        self.percentage=percentage

    def __call__(self, image, *args):
        if np.random.rand()<self.percentage:
            cv2.flip(image,self.dim,image)
        #inplace flip
        if len(args):
            return (image, *args)
        else:
            return image

class SSD_Expand(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, boxes, labels):
        if random.randint(2):
            return image, boxes, labels

        height, width, depth = image.shape
        ratio = random.uniform(1, 4)
        left = random.uniform(0, width*ratio - width)
        top = random.uniform(0, height*ratio - height)

        expand_image = np.zeros(
            (int(height*ratio), int(width*ratio), depth),
            dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[int(top):int(top + height),
                     int(left):int(left + width)] = image
        image = expand_image

        boxes = boxes.copy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))

        return image, boxes, labels


class SSD_RandomMirror(object):
    def __call__(self, image, boxes, classes):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]
        return image, boxes, classes


class SwapChannels(object):
    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image, *args):
        """
        Arguments:
            image (Tensor): image tensor to be transformed
        Returns:
            a tensor with channels swapped according to swap
        """
        temp = image.clone()
        for i in range(3):
            temp[i] = image[self.swaps[i]]
        if len(args):
            return (image, *args)
        else:
            return image


class SSD_PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, boxes, labels):
        im = image.copy()
        im, boxes, labels = self.rand_brightness(im, boxes, labels)
        if random.randint(2):
            distort = SSD_Compose(self.pd[:-1])
        else:
            distort = SSD_Compose(self.pd[1:])
        im, boxes, labels = distort(im, boxes, labels)
        return self.rand_light_noise(im, boxes, labels)


class SSDAugmentation(object):
    def __init__(self, size=300, mean=(104, 117, 123)):
        self.mean = mean
        self.size = size
        self.augment = SSD_Compose([
            ConvertFromInts(),
            SSD_ToAbsoluteCoords(),
            SSD_PhotometricDistort(),
            SSD_Expand(self.mean),
            SSD_RandomSampleCrop(),
            SSD_RandomMirror(),
            SSD_ToPercentCoords(),
            Resize(self.size),
            SubtractMeans(self.mean)
        ])

    def __call__(self, img, boxes, labels):
        return self.augment(img, boxes, labels)

def get_advanced_transform(dim,padding=5,random_crop=5,hue=True,
                           saturation=True,value=True,horizontal_flip=True,
                           random_noise=0,mean=(127.5,127.5,127.5),std=(127.5,127.5,127.5),
                           other_functions=()):
    # loader must be cv2 loader
    # other functions will be added after the HSV transform
    trans=[]
    if dim!=None:
        trans.append(Scale(dim))
    if padding:
        trans.append(Padding(int(padding)))
    if random_crop:
        trans.append(RandomCrop(int(random_crop)))
    if hue or saturation or value:
        trans.append(BGR_2_HSV())
    if hue:
        trans.append(RandomHue())
    if saturation:
        trans.append(RandomSaturation())
    if value:
        trans.append(RandomValue())
    if hue or saturation or value:
        trans.append(HSV_2_BGR())
    for func in other_functions:
        trans.append(func)
    if horizontal_flip:
        trans.append(Flip(1))
    if random_noise:
        trans.append(RandomNoise(random_noise))
    trans.append(ToTensor())
    # normalize method: (x_channel-mean)/std
    trans.append(transforms.Normalize(mean,std))
    return transforms.Compose(trans)

def get_advanced_transform_test(dim,mean=(127.5,127.5,127.5),std=(127.5,127.5,127.5)):
    # loader must be cv2 loader
    return transforms.Compose([
        Scale(dim),
        ToTensor(),
        transforms.Normalize(mean,std),
    ])

def cv2_loader(path):
    image=cv2.imread(path)
    if image is None:
        pass
    return image