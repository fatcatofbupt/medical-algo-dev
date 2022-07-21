import numpy as np
from skimage.metrics import structural_similarity as ssim_score
# from obs import ObsClient

from .factory import NewdataEvalFactory

@NewdataEvalFactory.register('eval_lungseg')
class Lungseg_Basic(object):
    def __init__(self, *args, **kargs):
        super(Lungseg_Basic, self).__init__()
        self.args = args
        self.kargs = kargs

    def single(self, input_ori, label_ori, meta_ori, input_new, label_new, meta_new):
        input_ori = ssim_score(input_ori,input_new,*self.args,**self.kargs,multichannel=True) 

        score = {
            'ssim': input_ori,
        }

        return score


    def summary(cls, score_list):
        ssim_list = [score['ssim'] for score in score_list]
        preds_ori = np.array([score.pop('pred_ori') for score in score_list])
        preds_new = np.array([score.pop('pred_new') for score in score_list])
        masks = np.array([score.pop('label') for score in score_list])

        correct_ori = preds_ori[preds_ori == masks]
        correct_new = preds_new[preds_new == masks]
        correct_num_orig = correct_ori.sum()
        correct_num_new = correct_new.sum()
        correct2wrong_num = correct_num_orig - correct_num_new
        attack_rate = correct2wrong_num / correct_num_orig
        accuracy_ori = correct_ori.sum() / len(masks.flatten())
        accuracy_new = correct_new.sum() / len(masks.flatten())
        accuracy_loss = (accuracy_ori - accuracy_new) / accuracy_ori
        score = {
            'ssim': np.array(ssim_list).mean(),
            'attack_rate': attack_rate,
            'acc_ori': accuracy_ori,
            'acc_new': accuracy_new,
            'acc_loss': accuracy_loss
        }
        return score


