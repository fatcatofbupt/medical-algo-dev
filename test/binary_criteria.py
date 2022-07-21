class BinaryClassificationEvaluation(object):
    '''
    二分类测试
    '''
    def __init__(self):
        super(BinaryClassificationEvaluation, self).__init__()
    

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
        return predict


    def criteria(self, predict_list):
        """
        根据糖尿病视网膜病变分类算法指标，写的评估函数
        :param predict_list: 输入列表 [{'label': 金标准, 'pred': 预测分类, 'prob': 预测置信度, 'path': 图片名, ...}, ...]
        :return: 需计算的指标 {'sen': 敏感性, 'spe': 特异性, 'acc': 准确率, 'kappa': kappa系数, 'auc': auc值}
        """
        from sklearn import metrics
        import pandas as pd
        import matplotlib.pyplot as plt
        
        df = pd.DataFrame(predict_list)
        # df = pd.DataFrame()
        # for filename in predict_dict:
        #     data_arr = predict_dict[filename]
        #     df.loc[filename, 'label'] = int(data_arr[0])
        #     df.loc[filename, 'pred'] = int(data_arr[1])
        #     df.loc[filename, 'prob'] = float(data_arr[2])
        try:
            df['label'][df['label'] > 1] = 1
            # print("Labels:",df['label'])
            # print("Probs:",df['prob'])
            cm = metrics.confusion_matrix(df['label'], df['pred'])
            TP = cm[1,1]
            TN = cm[0,0]
            FP = cm[0,1]
            FN = cm[1,0]
            sen = TP/(TP+FN)
            spe = TN/(TN+FP)
            acc = (TP+TN)/(TP+TN+FP+FN)
            kappa = metrics.cohen_kappa_score(df['label'], df['pred'])
            auc = metrics.roc_auc_score(df['label'], df['prob'])
            fpr,tpr,thres = metrics.roc_curve(df['label'],df['prob'])
            plt.figure()
            plt.plot(fpr,tpr,color='red',linewidth=2.0)
            plt.savefig('example.jpg')
            result = {'sen': sen, 'spe': spe, 'acc': acc, 'kappa': kappa, 'auc': auc, 'thres': thres.tolist(), 'tpr': tpr.tolist(), 'fpr': fpr.tolist()}
        except Exception as e:
            print("Exception:",e)
            result = {}
        print("Result1:",result)
        return result
