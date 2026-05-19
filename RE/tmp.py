import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModelForTokenClassification,BertTokenizerFast
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict
import matplotlib.pyplot as plt
import json
from sklearn.metrics import classification_report
os.environ["WANDB_MODE"] = "disabled"
# ==============================
# 参数设定
# ==============================
usemodel="roberta"
MODEL_NAME = rf"D:\dl\毕业设计\xxcq\o1\RE\relation_extraction_output\{usemodel}\lr_3e-05_epo_15_bs_16_ML_512_final\checkpoint-9510"
TEST_DATA_PATH = r"D:/dl/毕业设计/xxcq/test_data.json"  # 测试集路径
OUTPUT_DIR = rf'D:\dl\毕业设计\xxcq\o1\RE\relation_extraction_output\{usemodel}\lr_3e-05_epo_15_bs_16_ML_512_final'
MAX_LENGTH = 512
BATCH_SIZE = 16
token = r"D:\dl\bert-crf-token_classification_ner-master\pretrain\chinese-roberta-wwm-ext" 
RELATION_LABELS = ["sell_drugs_to", "traffic_in", "possess", "provide_shelter_for", "NA"]
label2id = {label: i for i, label in enumerate(RELATION_LABELS)}
id2label = {i: label for label, i in label2id.items()}

# ==============================
# 数据加载与预处理
# ==============================
def get_entity_positions(entityMentions):
    # 根据起始index映射实体
    # 返回以 (start_idx, end_idx, text, label) 的列表
    entities = []
    for ent in entityMentions:
        entities.append((ent["start"], ent["end"], ent["text"], ent["label"]))
    return entities

def build_samples(data: List[Dict]):
    samples = []
    for item in data:
        sentence = item["sentText"]
        entityMentions = item["entityMentions"]
        relationMentions = item.get("relationMentions", [])

        entities = get_entity_positions(entityMentions)

        # 构建已知关系样本
        for rm in relationMentions:
            e1_text = rm["em1Text"]
            e2_text = rm["em2Text"]
            relation_label = rm["label"]
            if relation_label not in label2id:
                # 若数据中有未知关系，可选跳过或统一为NA
                relation_label = "NA"

            # 将该关系对加入samples中
            samples.append({
                "sentence": sentence,
                "entity1": e1_text,
                "entity2": e2_text,
                "label": relation_label
            })

        # 如需构建NA样本(无关系实体对)，请在此添加逻辑
        # 例如：对所有实体对，如果不在relationMentions中，则标记为NA
        # 这里仅为示意，不一定需要
        # 已有标注则略过此步骤
        # entity_texts = [e[2] for e in entities]
        # for i in range(len(entities)):
        #     for j in range(i+1, len(entities)):
        #         e1_text = entities[i][2]
        #         e2_text = entities[j][2]
        #         # 检查是否在relationMentions出现
        #         paired = False
        #         for rm in relationMentions:
        #             if (rm["em1Text"] == e1_text and rm["em2Text"] == e2_text) or (rm["em1Text"] == e2_text and rm["em2Text"] == e1_text):
        #                 paired = True
        #                 break
        #         if not paired:
        #             samples.append({
        #                 "sentence": sentence,
        #                 "entity1": e1_text,
        #                 "entity2": e2_text,
        #                 "label": "NA"
        #             })

    return samples
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

data = load_data(TEST_DATA_PATH)

# ==============================
# 数据集定义
# ==============================

class RelationDataset(Dataset):
    def __init__(self, samples: List[Dict], tokenizer, max_length: int = 128):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encodings = self._encode_samples(self.samples)

    def _encode_samples(self, samples: List[Dict]):
        sentences = []
        labels = []
        for s in samples:
            text = s["sentence"]
            e1 = s["entity1"]
            e2 = s["entity2"]
            input_text = text + " [SEP] " + e1 + " [SEP] " + e2
            sentences.append(input_text)

            label = s["label"]
            label_ids = [label2id[label]] * self.max_length
            labels.append(label_ids)

        encodings = self.tokenizer(
            sentences,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        encodings['labels'] = torch.tensor(labels)
        return encodings

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        return item

#tokenizer = BertTokenizerFast.from_pretrained(r"D:\dl\bert-crf-token_classification_ner-master\pretrain\bert-base-chinese")
# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(token)

# 构建测试数据集
test_samples = build_samples(data)
test_dataset = RelationDataset(test_samples, tokenizer, max_length=MAX_LENGTH)

# ==============================
# 加载模型
# ==============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, num_labels=len(RELATION_LABELS))
model.to(device)  # 将模型移动到GPU（或CPU）

# ==============================
# 测试并输出评价指标
# ==============================

# def evaluate_model(model, test_dataset, batch_size):
#     dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
#     all_preds = []
#     all_labels = []

#     with torch.no_grad():
#         for batch in dataloader:
#             input_ids = batch['input_ids'].to(device)  # 将数据移动到GPU
#             attention_mask = batch['attention_mask'].to(device)  # 将数据移动到GPU
#             labels = batch['labels'].to(device)  # 将数据移动到GPU

#             outputs = model(input_ids, attention_mask=attention_mask)
#             logits = outputs.logits
#             preds = torch.argmax(logits, dim=-1)

#             # 收集预测值和实际标签
#             all_preds.extend(preds.cpu().numpy().flatten())  # 将预测结果移回CPU
#             all_labels.extend(labels.cpu().numpy().flatten())  # 将标签移回CPU

#     return all_preds, all_labels
def evaluate_model(model, test_dataset, batch_size):
    dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)  # 将数据移动到GPU
            attention_mask = batch['attention_mask'].to(device)  # 将数据移动到GPU
            labels = batch['labels'].to(device)  # 将数据移动到GPU

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)

            # 忽略填充部分 (PAD token)，根据 attention_mask 来过滤
            for i in range(input_ids.size(0)):  # 遍历批次中的每个样本
                mask = attention_mask[i].cpu().numpy()  # 获取当前样本的 attention_mask
                pred = preds[i].cpu().numpy()  # 获取当前样本的预测值
                label = labels[i].cpu().numpy()  # 获取当前样本的标签

                # 仅保留有效的标签（非填充部分）
                valid_indices = np.where(mask != 0)[0]  # 找到非PAD的索引
                all_preds.extend(pred[valid_indices])  # 将有效的预测值添加到列表
                all_labels.extend(label[valid_indices])  # 将有效的真实标签添加到列表

    return all_preds, all_labels

# 获取模型预测结果与实际标签
predictions, actual_labels = evaluate_model(model, test_dataset, BATCH_SIZE)

# 计算混淆矩阵和其他评价指标
conf_matrix = confusion_matrix(actual_labels, predictions)
accuracy = accuracy_score(actual_labels, predictions)
precision = precision_score(actual_labels, predictions, average='weighted')
recall = recall_score(actual_labels, predictions, average='weighted')
f1 = f1_score(actual_labels, predictions, average='weighted')
class_report = classification_report(actual_labels, predictions, target_names=RELATION_LABELS)
# 输出评价指标到txt文件
metrics_output_path = os.path.join(OUTPUT_DIR, "evaluation_metrics.txt")
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(metrics_output_path, 'w') as f:
    f.write(f"Accuracy: {accuracy}\n")
    f.write(f"Precision: {precision}\n")
    f.write(f"Recall: {recall}\n")
    f.write(f"F1 Score: {f1}\n")
    f.write("\nConfusion Matrix:\n")
    f.write(np.array2string(conf_matrix))
    f.write("\n\nClassification Report:\n")
    f.write(class_report)

# 绘制并保存混淆矩阵图
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(RELATION_LABELS))
plt.xticks(tick_marks, RELATION_LABELS, rotation=45)
plt.yticks(tick_marks, RELATION_LABELS)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))

# 保存预测结果和实际标签到CSV
predictions_labels_df = pd.DataFrame({
    "Prediction": [id2label.get(label, "NA") for label in predictions],
    "Actual": [id2label.get(label, "NA") for label in actual_labels]
})
predictions_labels_df.to_csv(os.path.join(OUTPUT_DIR, "predictions_vs_actual.csv"), index=False)

print(f"Evaluation metrics and confusion matrix saved to {OUTPUT_DIR}")