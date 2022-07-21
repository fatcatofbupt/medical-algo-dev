import numpy as np

from .factory import NewdataEvalFactory

@NewdataEvalFactory.register('eval_ecg')
class ECG_new_data_eval(object):
    def __init__(self, *args, **kargs):
        super(ECG_new_data_eval, self).__init__()
        self.args = args
        self.kargs = kargs
    
    #TODO: 怎么去评价新生成的数据集和原有数据集和的差异，然后替换掉score就行
    def single(self, input_ori, label_ori, meta_ori, input_new, label_new, meta_new):
        score = 0
        score = {
            'score': score,
        }
        return score


    def summary(cls, score_list):
        score_list = [score['score'] for score in score_list]
        score = {
            'score': np.array(score_list).mean(),
        }
        return score