import numpy as np
from skimage.metrics import structural_similarity as ssim_score
import cv2
# from obs import ObsClient

from .factory import NewdataEvalFactory

@NewdataEvalFactory.register('eval_lungssd')
class Lungssd_Basic(object):
    def __init__(self, *args, **kargs):
        super(Lungssd_Basic, self).__init__()
        self.args = args
        self.kargs = kargs

    def single(self, input_ori, label_ori, meta_ori, input_new, label_new, meta_new):
        ssim = ssim_score(input_ori,input_new,*self.args,**self.kargs,multichannel=True) 

        score = {
            "ssim": ssim,
        }

        return score


    def summary(cls, score_list):
        ssim_list = [score["ssim"] for score in score_list]
        score = {
            "ssim": np.array(ssim_list).mean(),
        }
        return score


