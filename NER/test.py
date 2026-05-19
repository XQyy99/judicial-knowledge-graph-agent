import json
import os
import torch
from typing import List, Dict
from torch.utils.data import Dataset
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, AlbertForTokenClassification,AutoModelForTokenClassification, AutoTokenizer
import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd 
from NERmodel import BertCRFModel
from safetensors.torch import load_file

os.environ["WANDB_MODE"] = "disabled"

# ========== 参数设定 ==========
usemodel='pert'

TEST_DATA_PATH = r'D:\dl\毕业设计\xxcq\test.json'
MODEL_DIR = rf"D:\dl\毕业设计\xxcq\o1\NER\ner_output\{usemodel}\lr_3e-05_epo_20_bs_16_XL_512_final\checkpoint-3640"
MAX_LENGTH = 512
#TOKENIZER_NAME = r"D:\dl\bert-crf-token_classification_ner-master\pretrain\chinese-roberta-wwm-ext"
TOKENIZER_NAME= r"D:\dl\bert-crf-token_classification_ner-master\pretrain\chinese-pert-base"
PREDICT_PATH=rf"D:\dl\毕业设计\xxcq\o1\NER\ner_output\{usemodel}\lr_3e-05_epo_20_bs_16_XL_512_final"

# 定义标签集，与训练时保持一致
entity_types = ["Nh", "Ns", "NT", "NDR", "NW"]
labels = ["O"] + ["B-"+t for t in entity_types] + ["I-"+t for t in entity_types]
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}

def load_data(data_path: str):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            data.append(item)
        return data

def get_entity_positions(entityMentions):
    entities = []
    for ent in entityMentions:
        entities.append((ent["start"], ent["end"], ent["label"]))
    return entities

def align_labels_with_tokens(tokens, entities, offset_mapping):
    labels_out = ["O"] * len(tokens)
    for (start_char, end_char, ent_label) in entities:
        b_label = "B-" + ent_label
        i_label = "I-" + ent_label
        entity_token_indices = []
        for i, (start, end) in enumerate(offset_mapping):
            if start_char < end and end_char > start:
                entity_token_indices.append(i)
        if len(entity_token_indices) > 0:
            labels_out[entity_token_indices[0]] = b_label
            for idx in entity_token_indices[1:]:
                labels_out[idx] = i_label
    return labels_out

def build_ner_samples(data: List[Dict], tokenizer: BertTokenizerFast, max_length: int = 128):
    samples = []
    for item in data:
        sentence = item["sentText"]
        entityMentions = item.get("entityMentions", [])
        entities = [(ent["start"], ent["end"], ent["label"]) for ent in entityMentions]

        encoding = tokenizer(sentence, return_offsets_mapping=True, truncation=True, padding="max_length", max_length=max_length)
        tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])
        offset_mapping = encoding["offset_mapping"]

        # 找到SEP位置
        sep_index = encoding["input_ids"].index(tokenizer.sep_token_id)
        valid_start, valid_end = 1, sep_index - 1

        raw_labels = align_labels_with_tokens(tokens, entities, offset_mapping)
        labels_out = ["O"] * len(tokens)
        for i in range(len(tokens)):
            if i < valid_start or i > valid_end:
                labels_out[i] = "O"
            else:
                labels_out[i] = raw_labels[i]

        label_ids = [label2id[l] if l in label2id else label2id["O"] for l in labels_out]

        samples.append({
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": label_ids
        })
    return samples

class NERDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v) for k, v in self.samples[idx].items()}
        return item

if __name__ == "__main__":
    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
    # model=BertCRFModel()
    # model.load_state_dict(load_file(r"D:\dl\毕业设计\xxcq\o1\NER\ner_output\mymodel\lr_3e-05_epo_10_bs_16_XL_256\checkpoint-710\model.safetensors", device="cuda"))  # 加载权重
    # model.eval()  
    # 加载测试数据
    test_data = load_data(TEST_DATA_PATH)
    test_samples = build_ner_samples(test_data, tokenizer, max_length=MAX_LENGTH)
    test_dataset = NERDataset(test_samples)

    # 创建 Trainer 用于预测
    trainer = Trainer(model=model, tokenizer=tokenizer)
    predictions = trainer.predict(test_dataset)

    preds = np.argmax(predictions.predictions, axis=2)
    label_ids = predictions.label_ids

    # 将预测和真实标签转换为seqeval需要的格式
    true_labels = []
    true_preds = []
    results=[]
    for pred, lab in zip(preds, label_ids):
        pred_labels = [id2label[p] for p in pred]
        true_lbls = [id2label[l] for l in lab]
        # 去除PAD/CLS/SEP影响，这里简单保持全长，seqeval在计算F1时只对非O的标签进行评估
        true_labels.append(true_lbls)
        true_preds.append(pred_labels)

        results.append({
            "gold_entities": true_lbls,
            "predicted_entities": pred_labels
        })
    csv_path = os.path.join(PREDICT_PATH, "test_results_csv.csv")
    results_df=pd.DataFrame(results)
    results_df.to_csv(csv_path,index=False)
    precision = precision_score(true_labels, true_preds)
    recall = recall_score(true_labels, true_preds)
    f1 = f1_score(true_labels, true_preds)
    cls_report = classification_report(true_labels, true_preds, digits=4)
    print("Evaluation on Test set:\n")
    print(f"Precision: {precision:.4f}\n")
    print(f"Recall:    {recall:.4f}\n")
    print(f"F1-score:  {f1:.4f}\n\n")
    print("Classification Report:\n")
    print(cls_report)
    # 将结果保存到txt文件
    result_path = os.path.join(PREDICT_PATH, "test_results.txt")
    with open(result_path, "w", encoding='utf-8') as f:
        f.write("Evaluation on Test set:\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"F1-score:  {f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(cls_report)

    print("测试集指标已写入:", result_path)

    # 绘制混淆矩阵（对所有标签，包括O）
    # 首先将所有预测和真实标签展平
    flat_true = []
    flat_pred = []
    for tl, pl in zip(true_labels, true_preds):
        flat_true.extend(tl)
        flat_pred.extend(pl)

    # 构建混淆矩阵
    # 注意：O标签很多，会占据主要比例。您可根据需要只对实体标签构建CM
    # 这里直接对全部标签构建
    cm = confusion_matrix(flat_true, flat_pred, labels=labels)

    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    thresh = cm.max() / 2  # 设置阈值，调整文本颜色
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'), 
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()

    cm_path = os.path.join(PREDICT_PATH, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print("混淆矩阵已保存至:", cm_path)