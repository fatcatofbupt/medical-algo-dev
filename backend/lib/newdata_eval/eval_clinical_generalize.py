from cProfile import label
from unittest.result import failfast
import numpy as np

from .factory import NewdataEvalFactory

@NewdataEvalFactory.register('eval_clinical_generalize')
class Clinical_data_generalize(object):
    def __init__(self, *args, **kargs):
        super(Clinical_data_generalize, self).__init__()
        self.args = args
        self.kargs = kargs
    
    def single(self, input_ori, label_ori, meta_ori, input_new, label_new, meta_new):

        raw_logits = meta_new["raw_logits"]
        raw_prediction = meta_new["raw_prediction"]
        adv_logits = meta_new["adv_logits"]
        adv_prediction = meta_new["adv_label"]
        num_label = len(raw_prediction[0])
        
        attacked_num = sum(raw_prediction[0]) - sum(adv_prediction[0])
        raw_score = 1 - (sum(raw_logits[0]) / num_label)
        adv_score = 1 - (sum(adv_logits[0]) / num_label)
        score = {'F1': meta_new['F1'], 
                'L2_loss': meta_new['L2_loss'], 
                'Perturbation_num': meta_new['Perturbation_num'], 
                'fail': meta_new['fail'], 
                "raw_score": raw_score, 
                "adv_score": adv_score, 
                "attacked_num": attacked_num
                }
        return score

    # 每个分数都这样处理
    def summary(cls, score_list):
        # read
        instance_num = len(score_list)
        F1_list = [score['F1'] for score in score_list]
        L2_loss_list = [score['L2_loss'] for score in score_list]
        Perturbation_num_list = [score['Perturbation_num'] for score in score_list]
        fail_list = [score['fail'] for score in score_list]
        
        raw_score_list = [score['raw_score'] for score in score_list]
        adv_score_list = [score['adv_score'] for score in score_list]
        raw_score = np.mean(raw_score_list)
        adv_score = np.mean(adv_score_list)
        score_change = adv_score - raw_score
        
        attacked_num_list = [score['attacked_num'] for score in score_list]
        attacked_label_mean = sum(attacked_num_list) / 20
        # count
        SR = 1 - float(np.array(fail_list).mean())
        L2_sum = float(np.array(L2_loss_list).mean())
        Perturbation_num = float(np.array(Perturbation_num_list).mean())
        fail_num = int(np.array(fail_list).sum())

        F1_sum = []
        for a in range(len(F1_list[0])): # 20为label的长度
            F1_sum.append({"TP": 0, "FP": 0, "FN": 0, "TN": 0})
        for F1 in F1_list:
            for a in range(len(F1_sum)):
                F1_sum[a]["TP"] += F1[a]["TP"]
                F1_sum[a]["FP"] += F1[a]["FP"]
                F1_sum[a]["TN"] += F1[a]["TN"]
                F1_sum[a]["FN"] += F1[a]["FN"]
    
        F1_result = gen_micro_macro_result(F1_sum)
        macro_F1 = F1_result["macro_f1"]
        mirco_F1 = F1_result["micro_f1"]

        # score
        score = {
            'SR': SR, 
            'L2_sum': L2_sum, 
            'Perturbation_num': Perturbation_num, 
            'macro': macro_F1, 
            'micro': mirco_F1, 
            'instance_num': instance_num, 
            'success': instance_num - fail_num, 
            # 'fail': fail_num, 
            'score_change': score_change, 
            'attacked_label_num': attacked_label_mean
        }
        return score



def gen_micro_macro_result(res):
    precision = []
    recall = []
    f1 = []
    total = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    for a in range(0, len(res)):
        total["TP"] += res[a]["TP"]
        total["FP"] += res[a]["FP"]
        total["FN"] += res[a]["FN"]
        total["TN"] += res[a]["TN"]

        p, r, f = get_prf(res[a])
        precision.append(p)
        recall.append(r)
        f1.append(f)

    micro_precision, micro_recall, micro_f1 = get_prf(total)

    macro_precision = 0
    macro_recall = 0
    macro_f1 = 0
    for a in range(0, len(f1)):
        macro_precision += precision[a]
        macro_recall += recall[a]
        macro_f1 += f1[a]

    macro_precision /= len(f1)
    macro_recall /= len(f1)
    macro_f1 /= len(f1)

    return {
        "micro_precision": round(micro_precision, 3),
        "micro_recall": round(micro_recall, 3),
        "micro_f1": round(micro_f1, 3),
        "macro_precision": round(macro_precision, 3),
        "macro_recall": round(macro_recall, 3),
        "macro_f1": round(macro_f1, 3)
    }

def get_prf(res):
    # According to https://github.com/dice-group/gerbil/wiki/Precision,-Recall-and-F1-measure
    if res["TP"] == 0:
        if res["FP"] == 0 and res["FN"] == 0:
            precision = 1.0
            recall = 1.0
            f1 = 1.0
        else:
            precision = 0.0
            recall = 0.0
            f1 = 0.0
    else:
        precision = 1.0 * res["TP"] / (res["TP"] + res["FP"])
        recall = 1.0 * res["TP"] / (res["TP"] + res["FN"])
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1

        