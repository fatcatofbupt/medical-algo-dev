from __future__ import division
from math import sqrt as sqrt
from itertools import product as product
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from typing import List, Tuple, Dict
import torch.nn.init as init
from torch.autograd import Function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms
from math import sqrt
import numpy as np
# TODO:这边的工厂虽然不知道为什么单独运行这个文件会报错，先注释，等下整体一起运行的时候还是要的
from backend.lib.models.factory import ModelFactory
import cv2
# SSD300 CONFIGS
ssd300_configs_clip = True
ssd300_configs_min_dim = 300
ssd300_configs_variance = [0.1, 0.2]
ssd300_configs_aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
ssd300_configs_max_sizes = [60, 111, 162, 213, 264, 315]
ssd300_configs_min_sizes = [30, 60, 111, 162, 213, 264]
ssd300_configs_steps = [8, 16, 32, 64, 100, 300]
ssd300_configs_feature_maps = [38, 19, 10, 5, 3, 1]


# Label map
voc_labels = ('background','nodules')
label_map = {k: v for v, k in enumerate(voc_labels)}
rev_label_map = {v: k for k, v in label_map.items()}  # Inverse mapping


# 图片的预处理所需要的
class BaseTransform:
    def __init__(self, size, mean = (104, 117, 123)):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)
    def base_transform(self,image, size, mean):
        x = cv2.resize(image, (size, size)).astype(np.float32)
        x -= mean
        x = x.astype(np.float32)
        return x
    def __call__(self, image):
        return self.base_transform(image, self.size, self.mean)


# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(
            loc:torch.Tensor,
            priors:torch.Tensor, 
            variances:List[float]
            ) -> torch.Tensor:
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        boxes(tensor):decoded bounding box predictions.
            Shape:[num_priors,4]
    """
    variances_0 = variances[0]
    variances_1 = variances[1]
    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances_0 * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances_1)), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

# Original author: Francisco Massa:
# https://github.com/fmassa/object-detection.torch
# Ported to PyTorch by Max deGroot (02/01/2017)
def nms(
    boxes:torch.Tensor,
    scores:torch.Tensor,
    overlap:float,
    top_k:int) -> Tuple[torch.Tensor,int]:
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """
    keep = torch.zeros_like(scores).long()
    if boxes.numel() == 0:
        return keep,0
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    _ , idx = scores.sort(0)  # sort in ascending order

    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = torch.tensor([],dtype = boxes.dtype,device = boxes.device)
    yy1 = torch.tensor([],dtype = boxes.dtype,device = boxes.device)
    xx2 = torch.tensor([],dtype = boxes.dtype,device = boxes.device)
    yy2 = torch.tensor([],dtype = boxes.dtype,device = boxes.device)
    w = torch.tensor([],dtype = boxes.dtype,device = boxes.device)
    h = torch.tensor([],dtype = boxes.dtype,device = boxes.device)

    
    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view

        # load bboxes of next highest vals
        idx = idx.detach()
        x1 = x1.detach()
        y1 = y1.detach()
        x2 = x2.detach()
        y2 = y2.detach()
        # TODO:这边貌似有warning，需要解决
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)

        # store element-wise max with next highest score
        
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count


class Detect(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    def __init__(self,
                num_classes:int,
                bkg_label:int, 
                top_k:int, 
                conf_thresh:float, 
                nms_thresh:float,
                ssd300_configs_variance:List[float]):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.variance = ssd300_configs_variance

    def forward(self, 
                loc_data:torch.Tensor,
                conf_data:torch.Tensor,
                prior_data:torch.Tensor):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: 
            conf_data: (tensor) Conf preds from conf layers
                Shape: 
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: 
        Return:
            output:(tensor) conf and loc
        """
        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)
        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        conf_preds = conf_data.view(num, num_priors,
                                    self.num_classes).transpose(2, 1)

        # Decode predictions into bboxes.
        for i in range(num):
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()
            for index,cs in enumerate(conf_scores):
                if 0 < index < self.num_classes:
                    c_mask = cs.gt(self.conf_thresh)
                    scores = cs[c_mask]
                    if scores.size(0) == 0:
                        continue
                    l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                    boxes = decoded_boxes[l_mask].view(-1, 4)
                    # idx of highest scoring and non-overlapping boxes per class
                    ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                    output[i, index , :count] = \
                        torch.cat((scores[ids[:count]].unsqueeze(1),
                                boxes[ids[:count]]), 1)
        return output

class L2Norm(nn.Module):
    def __init__(self,n_channels:int, scale:int):
        super(L2Norm,self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight,self.gamma)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        x = torch.div(x,norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out

class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """
    def __init__(self):
        super(PriorBox, self).__init__()
        self.image_size = ssd300_configs_min_dim
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = ssd300_configs_aspect_ratios
        self.variance = ssd300_configs_variance
        self.feature_maps = ssd300_configs_feature_maps
        self.min_sizes = ssd300_configs_min_sizes
        self.max_sizes = ssd300_configs_max_sizes
        self.steps = ssd300_configs_steps
        self.aspect_ratios = ssd300_configs_aspect_ratios
        self.clip = ssd300_configs_clip
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')
    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                f_k = self.image_size / self.steps[k]
                # unit center x,y
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # aspect_ratio: 1
                # rel size: min_size
                s_k = self.min_sizes[k]/self.image_size
                mean += [cx, cy, s_k, s_k]

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output



@ModelFactory.register('model_lungssd_basic')
class LungSsdBasic(nn.Module):
    params = {
        'model_path':{
            "description":"已训练的模型参数的路径，文件以.pth结尾",
            "value_type":"string",
            "value_range":"正确路径即可",
            "default":"models/lung_ssd_jit.pth"
        },
        'device':{
            "description":"算法运行的设备",
            "value_type":"enum",
            "value_range":["cpu","cuda"],
            "default":"cpu"
        }
    }
    def __init__(self,
                path ,
                size = 300,
                num_classes=2,
                device='cpu'):
        super(LungSsdBasic, self).__init__()
        self.model = torch.jit.load(path,map_location=device).to(device)
        self.device = device
        self.size = size
        self.num_classes = num_classes
        self.variance  = ssd300_configs_variance
        self.rev_label_map = rev_label_map
    def forward(self,x):
        output = self.model(x)
        return output
    def detect(self,x):
        detect2 = Detect(self.num_classes, 0, 200, 0.01, 0.45,self.variance)
        loc_preds,conf_preds,prior_preds = self.forward(x)
        output = detect2.forward(loc_preds,conf_preds,prior_preds)
        return output
    def detect_a_img(self,img):
        # 这边考虑到医院肺结节一般都是长宽相等的，resize时候只考虑了正方形
        w = img.shape[0]
        # 一些预处理和反预处理的类方法继承
        # resize 尺寸到300，并且归一化
        transform = BaseTransform(self.size, (104, 117, 123))
        # 图片预处理以及一些数据的处理
        pre_treated_image = torch.from_numpy(transform(img)).permute(2, 0, 1).unsqueeze(0)
       
        # Move to default device
        pre_treated_image = pre_treated_image.to(self.device)
        y = self.detect(pre_treated_image)
        detections = y.detach()
        # scale each detection back up to the image
        scale = torch.Tensor([img.shape[1], img.shape[0],
                            img.shape[1], img.shape[0]])
        pred_num = 0

        scores = list()
        coords = list()
        labels = list()
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.6:
                score = detections[0, i, j, 0]
                scores.append(score)
                pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
                coord = (int(pt[0]), int(pt[1]), int(pt[2]), int(pt[3]))
                coords.append(coord)
                label = i
                labels.append(label)
                pred_num += 1
                j+=1
        detections_output = {'scores':scores,'coords':coords,'labels':labels}
        return detections_output
    def prd_conf(self,x):
        _,output,_ = self.forward(x)
        return output
    def compute_object_loss(self,x):
        pre_conf = torch.squeeze(self.prd_conf(x))
        target = torch.zeros(pre_conf.shape[0],dtype=torch.long)
        loss_fun = torch.nn.CrossEntropyLoss()
        loss = loss_fun(pre_conf,target)
        return loss
    def compute_object_vanishing_gradient(self,x):
        x.requires_grad = True
        loss = self.compute_object_loss(x)
        loss.backward()
        gradient = x.grad
        x.requires_grad = False
        return gradient
