from curses import meta
import albumentations
import torchvision.transforms as transforms
import random

import jieba.posseg as posseg
from cmath import nan
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import json
import os
import numpy as np
import jieba
from .factory import TestFactory


def multi_label_accuracy(outputs, label, config = None, result=None):
    if len(label[0]) != len(outputs[0]):
        raise ValueError('Input dimensions of labels and outputs must match.')

    outputs = outputs.data
    labels = label.data

    if result is None:
        result = []
        

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


@TestFactory.register('algorithm_clinical')
class Attack_Clinical(object):
    params = {

    }


    def __init__(self, 
                attack_types='white',
                attack_levels=2, 
                w2v_path=None, 
                word2id_path=None, 
                id2word_path=None, 
                word_neighbor_path=None, 
                concept_path=None, 
                label_num=20, 
                output_dim=20
                ):
        super(Attack_Clinical, self).__init__()
        self.attack_types = attack_types
        self.attack_levels = attack_levels
        self.w2v_path = w2v_path
        self.word2id_path = word2id_path
        self.id2word_path = id2word_path
        self.word_neighbor_path = word_neighbor_path
        self.concept_path = concept_path
        self.w2v = np.load(self.w2v_path)
        self.word2id = json.load(open(self.word2id_path, "r"))
        self.id2word = json.load(open(self.id2word_path, "r"))
        self.word_neighbor = json.load(open(self.word_neighbor_path, "r"))
        self.concept = json.load(open(self.concept_path, "r"))
        self.lam = 0.01
        self.k = 20
        self.max_len = 512
        self.task = self.attack_levels
        self.gen_result = True                     #gen answer
        self.label_num = 20 # label数目
        self.output_dim = 20
        self.embedding1 = nn.Embedding.from_pretrained(
                            embeddings=torch.from_numpy(self.w2v),
                            freeze = True
                            ).cpu()
        
    def run(self, clin_data, label, meta, model, device):
        note = meta['raw']
        if self.attack_types== 'white':
            input_new, label_new, meta_new = self.attack(model, note, meta)
        return input_new, label_new, meta_new

    def attack(self, model, note, meta):
        # 要输出SR，PR(记录一下总次数)，L2 sum(记录一下)，mic和mac(维护一个F1)
        Perturbation_num = 0
        L2_sum = 0
        F1 = []
        for a in range(self.label_num): # 从10改为20(因为label数变了)
            F1.append({"TP": 0, "FP": 0, "FN": 0, "TN": 0})
        success = 0
        fail = 0
        instance_num = 500
        # lam是Perturbation Distance Score的权重
        # NOTE:model即从attack_demo.py传过来的self.model
        #model.eval()   #这句话不加坑死我了
        # model.train()
        out_data = {}

        candidate_max_num = 200



        # 开始
        data = note
        data = self.process([data])
        text_length = data["length"]
        target_label = Variable(1 - data["label"]).float().cpu()
        label = Variable(data["label"]).float().cpu()
        input_seq = data["input"][0].numpy().tolist()
        property_seq = data["property"][0].tolist()
        medical_pos = data["medical_pos"][0]
        CUI_pos = data["CUI"][0]
        iter_num = 0
        iter_L2_sum = 0
        raw_seq, mutable_raw_seq = data["rawtext"][0], data["rawtext"][0]
        # 跳转到这里
        while True:
            iter_num += 1
            if iter_num >= 32:
                #print("fail for exceed iteration num!")
                fail = 1
                break
            #localize k candidates
            #if len(input_seq) == 0:
            #    print("shit #1")
            input_embed = self.embedding1(Variable(torch.from_numpy(np.array(input_seq)).unsqueeze(0)).long().cpu()).requires_grad_(True)
            result = model.predict(data = {'input':input_embed, "label": label})
            old_logits, prediction = result["logits"], result["prediction"].long() # prob, pred
            if self.check_leave(prediction, label, self.task):
                #完成全部修改，出口一
                #print("exit #1")
                success = 1
                Perturbation_num = float(iter_num / text_length[0])
                L2_sum = iter_L2_sum
                if self.gen_result:
                    break
            loss = self.calc_loss(old_logits, target_label)[0].cpu()
            #print("loss:  ", loss)
            gradients = torch.autograd.grad(outputs=loss,
                        # inputs = model.embeded,
                        inputs = input_embed, 
                        grad_outputs=None,
                        retain_graph=True,
                        create_graph=False,
                        only_inputs=True)[0].cpu().numpy()
            gradients = np.linalg.norm(gradients, axis=2, keepdims=False)
            gradients = np.argsort(-gradients, axis=1)
            candidate = []  #{pos, oldvec, newvec, type}, pos指的是修改的起点
            #填充candidate
            candi_flag = 0
            # NOTE:词替换
            for k in range(self.k):
                location = gradients[0][k]
                #print("location: ", location)
                #先判断是不是医学词汇
                if location in medical_pos.keys():
                    try:
                        CUI = medical_pos[location]
                        start_pos, end_pos = CUI_pos[CUI]
                    except:
                        candi_flag = 1
                        break
                    try:
                        oldvec = [input_seq[idx] for idx in range(start_pos, end_pos + 1)]
                    except:
                        continue
                    for atom in self.concept[CUI]:
                        atom = jieba.lcut(atom)
                        newvec = []
                        rawwords = []
                        for atom_word in atom:
                            rawwords.append(atom_word)
                            if atom_word in self.word2id.keys():
                                newvec.append(self.word2id[atom_word])
                            else:
                                newvec.append(self.word2id["UNK"])
                        candidate.append({"pos": start_pos, "oldvec": oldvec, "newvec": newvec, "type": "med", "rawword": rawwords})
                else:
                    #先进行替换，如果是副词考虑删去
                    word = self.id2word[str(input_seq[location])]
                    if word in self.word_neighbor.keys():
                        for neighbor in self.word_neighbor[word]:
                            neighbor_id = neighbor
                            candidate.append({"pos": location, "oldvec": [input_seq[location]], "newvec": [neighbor_id], "type": "rep",
                                            "rawword": [self.id2word[str(neighbor_id)]]
                            })
                    #删去副词的操作
                    if property_seq[location] == 0:
                        if location > 0 and property_seq[location - 1] == 1:
                            old_vec = input_seq[location - 1 : location + 1]
                            new_vec = input_seq[location - 1]
                            candidate.append({"pos": location - 1, "oldvec": old_vec, "newvec": new_vec, "type": "rem",
                                                "rawword": [self.id2word[str(new_vec)]]
                                                })
                        elif location < self.max_len - 1 and property_seq[location + 1] == 1:
                            old_vec = input_seq[location: location + 2]
                            new_vec = input_seq[location + 1]
                            candidate.append({"pos": location, "oldvec": old_vec, "newvec": new_vec, "type": "rem",
                                                "rawword": [self.id2word[str(new_vec)]]
                            })
                if candi_flag:
                    break
            #candidate组成batch来跑
            candidate_input = []
            for candi in candidate:
                if not isinstance(candi["oldvec"], list):
                    candi["oldvec"] = [candi["oldvec"]]
                if not isinstance(candi["newvec"], list):
                    candi["newvec"] = [candi["newvec"]]
                pos = candi["pos"]
                oldvec = candi["oldvec"]
                newvec = candi["newvec"]
                new_seq = input_seq[:pos] + newvec + input_seq[pos + len(oldvec):]
                while len(new_seq) < self.max_len:
                    new_seq.append(0)
                new_seq = new_seq[:self.max_len]
                candidate_input.append(new_seq)
            if len(candidate_input) == 0:
                # 负分，出口#2
                fail = 1
                #print("fail for no candidate!")
                break
            if len(candidate_input) > candidate_max_num:         #取前200个candidate
                candidate_input = candidate_input[:candidate_max_num]
            candidate_input = Variable(torch.from_numpy(np.array(candidate_input))).long().cpu()
            candidate_embed = self.embedding1(candidate_input)
            logits = model.predict(data = {"input": candidate_embed, "label": label})["logits"]
            candidate_loss = self.calc_loss(logits, target_label).detach().cpu().numpy()
            del candidate_input, logits, candidate_embed
            candidate_score = []
            for idx in range(min(len(candidate), candidate_max_num)):
                candidate_score.append(self.calc_score(loss.item(), candidate_loss[idx], candidate[idx]["oldvec"], candidate[idx]["newvec"]))
            #选取最高的结果
            highest_candidate = candidate_score.index(max(candidate_score))
            if candidate_score[highest_candidate] <= 0 or candidate_score[highest_candidate] != candidate_score[highest_candidate]:
                #负分，出口#2
                fail = 1
                break
            #更新input_seq, property_seq, medical_pos, CUI_pos
            pos, oldvec = candidate[highest_candidate]["pos"], candidate[highest_candidate]["oldvec"]
            type, newvec = candidate[highest_candidate]["type"], candidate[highest_candidate]["newvec"]
            iter_L2_sum += np.linalg.norm(self.get_avg_vec(oldvec) - self.get_avg_vec(newvec))
            new_input_seq = input_seq[:pos] + newvec + input_seq[pos + len(oldvec):]
            candidate[highest_candidate]["rawword"][0] = "##" + candidate[highest_candidate]["rawword"][0]
            candidate[highest_candidate]["rawword"][-1] = candidate[highest_candidate]["rawword"][-1] + "##"
            mutable_raw_seq = mutable_raw_seq[:pos] + candidate[highest_candidate]["rawword"] + mutable_raw_seq[pos + len(oldvec):]
            while len(new_input_seq) < self.max_len:
                new_input_seq.append(0)
            new_input_seq = new_input_seq[:self.max_len]
            input_seq = new_input_seq
            if type == "med":
                property_seq = property_seq[:pos] + [0] * len(newvec) + property_seq[pos + len(oldvec):]
            elif type == "rep":
                property_seq = property_seq[:pos] + [property_seq[pos]] + property_seq[pos + len(oldvec):]
            elif type == "rem":
                property_seq = property_seq[:pos] + [1] + property_seq[pos + len(oldvec):]
            while len(property_seq) < self.max_len:
                property_seq.append(0)
            property_seq = property_seq[:self.max_len]
            new_medical_pos = {}
            for key in medical_pos.keys():
                if key < pos:
                    new_medical_pos[key] = medical_pos[key]
                elif key == pos:
                    if type != "med":
                        continue
                    for key_idx in range(len(newvec)):
                        new_medical_pos[key + key_idx] = medical_pos[key]
                else:
                    new_medical_pos[key + len(newvec) - len(oldvec)] = medical_pos[key]
            new_CUI_pos = {}
            break_flag = 0
            for key in CUI_pos.keys():
                start_pos, end_pos = CUI_pos[key]
                if end_pos < pos:
                    new_CUI_pos[key] = [start_pos, end_pos]
                elif start_pos == pos and type == "med":
                    new_CUI_pos[key] = [pos, pos + len(newvec) - 1]
                elif start_pos > pos:
                    bias = len(newvec) - len(oldvec)
                    new_CUI_pos[key] = [start_pos + bias, end_pos + bias]
                else:
                    break_flag = 1
                    break
                if break_flag:
                    break
            medical_pos = new_medical_pos
            CUI_pos = new_CUI_pos
            
        #print("finish one data!")
        #再跑一个最终结果
        input_embed = self.embedding1(Variable(torch.from_numpy(np.array(input_seq)).unsqueeze(0)).long().cpu()).requires_grad_(True)
        output = model.predict(data = {"input": input_embed, "label": label})
        temp_F1 = cal_acc(logits=output['logits'], label=label)
        for a in range(self.label_num):
            F1[a]["TP"] += temp_F1[a]["TP"]
            F1[a]["FP"] += temp_F1[a]["FP"]
            F1[a]["TN"] += temp_F1[a]["TN"]
            F1[a]["FN"] += temp_F1[a]["FN"]
        def list2str(x):
            ret = ""
            for word in x:
                ret += word
            return ret
        out_data = {
                # "id": dataset_idx, 
                "label_num": self.label_num, 
                "raw_seq": list2str(raw_seq),
                "adv_seq": list2str(mutable_raw_seq),
                "raw_label": data["label"].numpy().tolist(),
                "adv_label": output["prediction"].long().cpu().numpy().tolist(),
                "L2_loss": iter_L2_sum,
                "Perturbation_num": float(iter_num / text_length[0]),
                "fail": int(fail)
            }
        out_data['F1'] = F1
        new_meta = meta.copy()
        new_meta.update(out_data)
        return out_data["adv_seq"], out_data["adv_label"], new_meta



    def calc_loss(self, logits, labels):
        #用于loss的计算，考虑batch个样本
        #relu-like optimization function
        if len(logits.shape) == 1:
            logits = logits.unsqueeze(0)
        if len(labels.shape) == 1:
            labels = labels.unsqueeze(0)
            labels = labels.repeat(int(logits.shape[0] / labels.shape[0]), 1)
        pre_loss = torch.abs(logits - labels)
        pre_loss = (1 - torch.ge(pre_loss, 0.5).float()).float() * 0.4 * pre_loss + torch.ge(pre_loss, 0.5).float() * (1.6 * pre_loss - 0.6)
        loss = torch.sum(pre_loss, dim = 1)
        return loss

    def calc_score(self, old_loss, candidate_loss, old_vec, new_vec):
        # Perturbation Saliency Score,  Perturbation Distance Score(L2 norm)
        #old_vec和new_vec是list，里面存的编号
        saliency = self.calc_saliency_score(old_loss, candidate_loss)
        distance = np.linalg.norm(self.get_avg_vec(old_vec) - self.get_avg_vec(new_vec))
        return saliency - self.lam * distance


    def calc_saliency_score(self, old_loss, candidate_loss):
        #loss做差
        return old_loss - candidate_loss


    def get_avg_vec(self, vec_list):
        #计算平均的词向量
        avg_vec = []
        for vec in vec_list:
            avg_vec.append(self.w2v[vec])
        avg_vec = np.mean(np.array(avg_vec), axis=0)
        return avg_vec

    def check_leave(self, prediction, label, task):
        prediction = prediction.squeeze().detach().cpu().numpy()
        label = label.squeeze().detach().cpu().numpy()
        #print(prediction)
        #print(label)
        if np.sum(prediction != label) >= task:
            return True
        else:
            return False

    # def black_box(self, model, input_seq, k, label):
    #     #black-box candidate selection
    #     top_k = []
    #     now_seq = input_seq
    #     while len(top_k < k):
    #         left, right = 0, len(input_seq)
    #         origin_result = model({"input": Variable(torch.from_numpy(np.array(now_seq))).cuda()})
    #         score_origin = self.calc_loss(origin_result["logits"], label).detach().cpu().numpy()
    #         del origin_result
    #         while left != right:
    #             mid = int((left + right)/ 2)
    #             x_left = self.mask(now_seq, left, mid)
    #             x_right = self.mask(now_seq, mid+1, right)
    #             result = model({"input": Variable(torch.from_numpy(np.array([x_left, x_right]))).cuda()})
    #             logits = result["logits"]
    #             score_left, score_right = self.calc_loss(logits[0], label).detach().cpu().numpy(), self.calc_loss(logits[1], label).detach().cpu().numpy()
    #             del result, logits
    #             if abs(score_left - score_origin) > abs(score_right - score_origin):
    #                 right = mid
    #             else:
    #                 left = mid + 1
    #         top_k.append(left)
    #         now_seq[left] = 0
    #     return top_k


    def mask(self, input_seq, begin, end):
        #闭区间
        pass

    def test_transfer(self, model, data_dir):
        model.eval()
        success_num = 0
        data = json.load(open(data_dir, "r"))
        for item in data:
            prediction = model({"input": Variable(torch.from_numpy(np.array(item["input"])))})["prediction"].detach().cpu().numpy()
            if np.sum(np.array(item["label"]) != prediction) >= 3:
                success_num += 1
        print("success:", success_num)


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
