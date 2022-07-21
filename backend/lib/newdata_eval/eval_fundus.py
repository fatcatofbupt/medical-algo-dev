import os
import json
from typing import Callable
from pathlib import Path

import numpy as np
import skimage.io
from skimage.metrics import structural_similarity as ssim_score
# from obs import ObsClient

from .factory import NewdataEvalFactory

@NewdataEvalFactory.register('eval_fundus')
class SSIM(object):
    def __init__(self, *args, **kargs):
        super(SSIM, self).__init__()
        self.args = args
        self.kargs = kargs


    def single(self, input_ori, label_ori, meta_ori, input_new, label_new, meta_new):
        input_ori = ssim_score(input_ori,input_new,*self.args,**self.kargs,multichannel=True) 

        score = {
            'ssim': input_ori,
        }
        print
        return score


    def summary(cls, score_list):
        ssim_list = [score['ssim'] for score in score_list]
        preds_ori = np.array([score['pred_ori'] for score in score_list])
        preds_new = np.array([score['pred_new'] for score in score_list])
        labels = np.array([score['label'] for score in score_list])

        correct_ori = preds_ori[preds_ori == labels]
        correct_new = preds_new[preds_new == labels]
        correct_num_orig = len(correct_ori)
        correct_num_new = len(correct_new)
        correct2wrong_num = correct_num_orig - correct_num_new
        attack_rate = correct2wrong_num / (correct_num_orig + 1e-5)
        accuracy_ori = len(correct_ori) / len(labels)
        accuracy_new = len(correct_new) / len(labels)
        accuracy_loss = (accuracy_ori - accuracy_new) / accuracy_ori
        score = {
            'ssim': np.array(ssim_list).mean(),
            'attack_rate': attack_rate,
            'acc_ori': accuracy_ori,
            'acc_new': accuracy_new,
            'acc_loss': accuracy_loss
        }
        return score


