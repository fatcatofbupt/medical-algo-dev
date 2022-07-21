import numpy as np

from .factory import NewdataEvalFactory

def success_rate(score):
    a = 0
    for i in range(len(score)):
        if score[i] == 1:
            a = a + 1
    rate = a / len(score)
    return rate
def compare(label_ori,label_new):
    num_ori = 0
    num_new = 0
    print('label_ori',label_ori,'label_new[0]',label_new[0],'label_new[1]',label_new[1])
    if label_ori == label_new[0]:
        num_ori = 1
    if label_ori == label_new[1]:
        num_new = 1
    return num_ori,num_new
@NewdataEvalFactory.register('eval_ecg')
class ECG_new_data_eval(object):
    def __init__(self, *args, **kargs):
        super(ECG_new_data_eval, self).__init__()
        self.args = args
        self.kargs = kargs
    
    #TODO: 怎么去评价新生成的数据集和原有数据集和的差异，然后替换掉score就行
    def single(self, input_ori, label_ori, meta_ori, input_new, label_new, meta_new):
        rate = 0
        ori,new = compare(label_ori,label_new)
        diff = input_ori - input_new
        for i in range(len(diff[0][0])):
            rate = rate + abs(diff[0][0][i])
        diff_rate = rate / len(diff[0][0])
        score = {
            'acc_ori':ori,
            'acc_new':new,
            'diff':diff_rate
        }
        return score


    def summary(cls, score_list):
        acc_ori = [score['acc_ori'] for score in score_list]
        acc_ori = success_rate(acc_ori)
        acc_new = [score['acc_new'] for score in score_list]
        acc_new = success_rate(acc_new)
        rate = 0
        diff = [score['diff']for score in score_list]
        for i in range(len(diff)):
            rate = rate + diff[i]
        difference = rate / len(diff)
        # print('rate',rate)
        score = {
            'acc_ori':acc_ori,
            'acc_new':acc_new,
            'difference':difference.item()
        }
        return score