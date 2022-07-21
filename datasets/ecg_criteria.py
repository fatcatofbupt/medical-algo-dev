import numpy as np
from sklearn import metrics
import sklearn
import pandas as pd
import matplotlib.pyplot as plt
class MultiClassificationEvaluation(object):
    '''
    二分类测试
    '''
    def __init__(self):
        super(MultiClassificationEvaluation, self).__init__()
    

    def predict(self, input_raw, label, meta, model):
        '''
        Inputs:
        --------------------
        数据集。需要能够迭代 (input_raw, label, meta)
        - for input_raw, label, meta in dataset

        其中:
        - input_raw: 原始输入 (如图像np.array, 文本str)
        - label: 标签 (如标签int, 目标检测list[(w,h,x,y)], 图像分割标签np.array, ...)
        - meta: dict 附带样本相关信息

        model: Model
            待检测模型。需要提供接口:
            - pred, prob = model.predict(input)

            其中:
            - pred: int 表示类别
            - probs: float 置信度

        Outputs:
        --------------------
        predict_list: list[dict]
            dict[{'label':金标准, 'pred':预测分类, 'prob':预测置信度, 'path': path, ...}]
        '''
        pred, prob = model.predict(input_raw)
        record = meta.copy()
        record.update({
            'label': label,
            'pred': pred,
            'prob': prob,
        })
        predict = record
        print('predict',predict)
        return predict


    def criteria(self, predict_list):
        """
        根据糖尿病视网膜病变分类算法指标，写的评估函数
        :param predict_list: 输入列表 [{'label': 金标准, 'pred': 预测分类, 'prob': 预测置信度, 'path': 图片名, ...}, ...]
        :return: 需计算的指标 {'sen': 敏感性, 'spe': 特异性, 'acc': 准确率, 'kappa': kappa系数, 'auc': auc值}
        """

        
        df = pd.DataFrame(predict_list)

        try:
            
            y_true = np.array(df['label'])
            a = []
            for i in range(len(y_true)):
                a.append(int(y_true[i]))
            y_true = np.array(a)

            y_pred = np.array(df['pred'])
    
            cm = metrics.confusion_matrix(y_true,y_pred)
            FP = cm.sum(axis=0) - np.diag(cm)  
            FN = cm.sum(axis=1) - np.diag(cm)
            TP = np.diag(cm)
            TN = cm.sum()-(FP + FN + TP)
            TPR = TP/(TP+FN) # Sensitivity/ hit rate/ recall/ true positive rate
            TNR = TN/(TN+FP) # Specificity/ true negative rate
            PPV = TP/(TP+FP) # Precision/ positive predictive value
            NPV = TN/(TN+FN) # Negative predictive value
            FPR = FP/(FP+TN) # Fall out/ false positive rate
            FNR = FN/(TP+FN) # False negative rate
            FDR = FP/(TP+FP) # False discovery rate
            ACC = TP/(TP+FN) # accuracy of each class
            kappa = metrics.cohen_kappa_score(y_true,y_pred)
            # auc = metrics.roc_auc_score(y_true,y_pred,multi_class='ovo')
            # fpr,tpr,thres = metrics.roc_curve(y_true,y_pred)
            # plt.figure()
            # plt.plot(fpr,tpr,color='red',linewidth=2.0)
            # plt.savefig('example.jpg')
            result = {'sen': TPR.mean(), 'spe': TNR.mean(), 'acc': ACC.mean(), 'kappa': kappa, 
            # 'auc': auc, 'thres': thres.tolist(), 
            'tpr': TPR.tolist(), 'fpr': FPR.tolist()}
        except Exception as e:
            print("Exception:",e)
            result = {}
        print("Result1:",result)
        return result