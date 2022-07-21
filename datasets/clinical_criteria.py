from re import S
import numpy as np
import torch
import jieba.posseg as posseg
from torch.autograd import Variable
import torch.nn as nn
import json


def multi_label_accuracy(outputs, label, config = None, result=None):
    if len(label[0]) != len(outputs[0]):
        raise ValueError('Input dimensions of labels and outputs must match.')

    outputs = outputs.data
    labels = label.data

    if result is None:
        result = []
        # result: list(Dict[str, Any]) = []

    total = 0
    nr_classes = outputs.size(1)

    while len(result) < nr_classes:
        # one: torch.tensor(Dict[str, Any]) = {"TP": 0, "FN": 0, "FP": 0, "TN": 0}
        # input = [result, one] 
        # result = torch.cat(input, dim=0)

        result.append({"TP": 0, "FN": 0, "FP": 0, "TN": 0})

        # one: torch.tensor(Dict[str, Any]) = {"TP": 0, "FN": 0, "FP": 0, "TN": 0}
        # result += one

    for i in range(nr_classes):
        outputs1 = (outputs[:, i] >= 0.5).long()
        labels1 = (labels[:, i].float() >= 0.5).long()
        total += int((labels1 * outputs1).sum())
        total += int(((1 - labels1) * (1 - outputs1)).sum())

        if result is None:
            continue

        # if len(result) < i:
        #    result.append({"TP": 0, "FN": 0, "FP": 0, "TN": 0})

        result[i]["TP"] += int((labels1 * outputs1).sum())
        result[i]["FN"] += int((labels1 * (1 - outputs1)).sum())
        result[i]["FP"] += int(((1 - labels1) * outputs1).sum())
        result[i]["TN"] += int(((1 - labels1) * (1 - outputs1)).sum())
    return result


def cal_acc(logits,label):
    if len(label.shape) == 1:
        label = label.unsqueeze(0)
    acc_result = multi_label_accuracy(outputs = logits, label = label, config = None ,result = None)
    return acc_result




class BinaryClassificationEvaluation(object):
    '''
    二分类测试
    '''
    def __init__(self):
        super(BinaryClassificationEvaluation, self).__init__()


        self.w2v_path = '/data/sjy21/old_data/pot/dict/w2v.npy'
        self.word2id_path = '/data/sjy21/old_data/pot/dict/word2id_v2.json'
        # self.id2word_path = '/data/sjy21/old_data/pot/dict/id2word_v2.json'
        # self.word_neighbor_path = '/data/sjy21/old_data/pot/dict/neighbor.json'
        # self.concept_path = '/data/sjy21/old_data/pot/dict/concept.json'
        self.w2v = np.load(self.w2v_path)
        self.word2id = json.load(open(self.word2id_path, "r"))
        # self.id2word = json.load(open(self.id2word_path, "r"))
        # self.word_neighbor = json.load(open(self.word_neighbor_path, "r"))
        # self.concept = json.load(open(self.concept_path, "r"))
        # self.lam = 0.01
        # self.k = 20
        self.max_len = 512
        # self.task = 3
        # self.gen_result = True                     #gen answer
        self.label_num = 20 # label数目
        self.output_dim = 20
        self.embedding1 = nn.Embedding.from_pretrained(
            embeddings=torch.from_numpy(self.w2v),
            freeze = True
            ).cpu()
    
    def process(self, data):
        input = []
        label = []
        property = []
        medical_pos = []
        CUI = []
        length = []
        raw_texts = []

        for item in data:
            process_res = posseg.cut(item["text"])  # jieba,分割，并有对分词的词性进行标注
            begin_points = {} #统计所有起点    
            length_cnt = 0
            TEXT, PROPERTY = [], []
            RAWTEXT = []
            for idx, piece_res in enumerate(process_res):
                TEXT.append(piece_res.word)
                RAWTEXT.append(piece_res.word)
                PROPERTY.append(piece_res.flag)
                #处理端点
                begin_points[length_cnt] = idx
                length_cnt += len(piece_res.word)

            length.append(len(TEXT))
            raw_texts.append(TEXT)
            token = []
            pos_tag = []
            for word in TEXT:
                word = word.lower()
                if word in self.word2id.keys():
                    token.append(self.word2id[word])
                else:
                    token.append(1)  #UNK 1
            #词性
            for tag in PROPERTY:
                if tag in ['a', 'ad', 'an', 'Ag']:
                    pos_tag.append(1)
                elif tag in ['dg', 'd']:
                    pos_tag.append(0)
                else:
                    pos_tag.append(-1)

            while len(token) < self.max_len:
                token.append(0)
                pos_tag.append(-1)
            token = token[0:self.max_len]
            pos_tag = pos_tag[0:self.max_len]

            input.append(token)
            item_label = [0] * self.output_dim
            for disease in item["label"]:
                item_label[disease] = 1

            label.append(item_label)
            property.append(pos_tag)

            item_medical_pos, item_CUI = self.process_entity(item["entity"], RAWTEXT, begin_points)
            medical_pos.append(item_medical_pos)
            CUI.append(item_CUI)


        input = torch.from_numpy(np.array(input)).long()
        label = torch.from_numpy(np.array(label)).long()
        property = np.array(property)
        return {'input': input, 'label': label, "property": property, "medical_pos": medical_pos, "CUI": CUI, "length": length, "rawtext": raw_texts}


    def process_entity(self, entities, text, begin_points):
        #生成medical_pos和CUI，每个位置对应的CUI以最后一次覆盖为准
        medical_pos = {}
        CUI = {}
        for item in entities:
            start_pos, end_pos = item["left"], item["right"]   #左闭右开
            if (start_pos in begin_points.keys()) and (end_pos in begin_points.keys()):
                left = begin_points[start_pos]
                right = begin_points[end_pos]
                CUI[item["CUI"]] = [left, right - 1]
                for idx in range(left, right):
                    medical_pos[idx] = item["CUI"]
        return medical_pos, CUI


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
        note = meta['raw']
        data = note
        data = self.process([data])
        label = data["label"]
        input_seq = data["input"][0].numpy().tolist()
        input_embed = self.embedding1(Variable(torch.from_numpy(np.array(input_seq)).unsqueeze(0)).cuda().long().cpu()).requires_grad_(True)
        label = Variable(label).cuda().float().cpu()
        result = model.predict(data = {'input':input_embed, "label": label})
        logits, prediction = result["logits"], result["prediction"].long() # prob, pred
        pred = prediction
        prob = logits
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
            # labels
            y_true = np.array(df['label'])
            a = []
            for i in range(len(y_true)):
                a.append((y_true[i].squeeze(0).detach().numpy()))
            y_true = np.array(a)
            # preds
            y_pred = np.array(df['pred'])
            b = []
            for i in range(len(y_pred)):
                b.append((y_pred[i].squeeze(0).detach().numpy()))
            y_pred = np.array(b)
            # probs
            y_prob = np.array(df['prob'])
            c = []
            for i in range(len(y_prob)):
                c.append((y_prob[i].squeeze(0).detach().numpy()))
            y_prob = np.array(c)
            # 评价指标
            sen = []
            spe = []
            acc = []
            kappa = []
            auc = []
            for i in range(len(y_true)):
                cm = metrics.confusion_matrix(y_true[i].tolist(), y_pred[i].tolist())
                TP = cm[1,1]
                TN = cm[0,0]
                FP = cm[0,1]
                FN = cm[1,0]
                sen.append(TP/(TP+FN))
                spe.append(TN/(TN+FP))
                acc.append((TP+TN)/(TP+TN+FP+FN))
                kappa.append(metrics.cohen_kappa_score(y_true[i].tolist(), y_pred[i].tolist()))
                auc.append(metrics.roc_auc_score(y_true[i].tolist(), y_prob[i].tolist()))
                # fpr,tpr,thres = metrics.roc_curve(df['label'],df['prob'])
                # plt.figure()
                # plt.plot(fpr,tpr,color='red',linewidth=2.0)
                # plt.savefig('example%s.jpg', %s(i))
            # result = {'sen': sen, 'spe': spe, 'acc': acc, 'kappa': kappa, 'auc': auc, 'thres': thres.tolist(), 'tpr': tpr.tolist(), 'fpr': fpr.tolist()}
            result = {'sen': sen, 'spe': spe, 'acc': acc, 'kappa': kappa, 'auc': auc}
        except Exception as e:
            print("Exception:",e)
            result = {}
        print("Result1:",result)
        return result




