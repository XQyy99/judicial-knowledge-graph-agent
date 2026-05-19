from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, AutoModelForTokenClassification,AutoTokenizer
import torch
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import json
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Dict

os.environ["WANDB_MODE"] = "disabled"
usemodel="macbert"
MODEL_DIR = rf"D:\dl\毕业设计\xxcq\o1\RE\relation_extraction_output\{usemodel}\lr_3e-05_epo_15_bs_16_ML_512_final\checkpoint-9510"
TEST_DATA_PATH = r'D:\dl\毕业设计\xxcq\test.json'
TOKENIZER_NAME = r"D:\dl\bert-crf-token_classification_ner-master\pretrain\chinese-macbert-base"
PREDICT_PATH=rf'D:\dl\毕业设计\xxcq\o1\RE\relation_extraction_output\{usemodel}\lr_3e-05_epo_15_bs_16_ML_512_final'
RELATION_LABELS = ["sell_drugs_to", "traffic_in", "possess", "provide_shelter_for", "NA"]
label2id = {label: i for i, label in enumerate(RELATION_LABELS)}
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
        entities.append((ent["start"], ent["end"], ent["text"], ent["label"]))
    return entities

def build_samples(data):
    samples = []
    for item in data:
        sentence = item["sentText"]
        relationMentions = item.get("relationMentions", [])
        for rm in relationMentions:
            e1_text = rm["em1Text"]
            e2_text = rm["em2Text"]
            relation_label = rm["label"]
            if relation_label not in label2id:
                relation_label = "NA"
            samples.append({
                "sentence": sentence,
                "entity1": e1_text,
                "entity2": e2_text,
                "label": relation_label
            })
    return samples

class RelationDataset(Dataset):
    def __init__(self, samples: List[Dict], tokenizer: BertTokenizerFast, max_length: int = 128):
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
            # 构建输入
            input_text = text + " [SEP] " + e1 + " [SEP] " + e2
            sentences.append(input_text)

            # 这里使用NA标签，你需要构建与输入句子长度匹配的标签序列
            label = s["label"]
            label_ids = [label2id[label]] * self.max_length  # 将标签扩展到max_length

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

if __name__ == "__main__":
    # 加载已训练好的模型和分词器
    #tokenizer = BertTokenizerFast.from_pretrained(r"D:\dl\bert-crf-token_classification_ner-master\pretrain\bert-base-chinese")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_DIR,
        num_labels=len(RELATION_LABELS)
    )

    # 加载测试数据集
    test_data = load_data(TEST_DATA_PATH)
    test_samples = build_samples(test_data)
    test_dataset = RelationDataset(test_samples, tokenizer)

    # 创建 Trainer，只为预测用，不传入 train_dataset，不调用 train()
    trainer = Trainer(model=model, tokenizer=tokenizer)

    # 对测试集进行预测
    predictions = trainer.predict(test_dataset)
    preds = predictions.predictions.argmax(-1)
    labels = predictions.label_ids

    # 扁平化labels和preds
    labels_flat = labels.flatten()
    preds_flat = preds.flatten()

    # 计算准确率
    accuracy = accuracy_score(labels_flat, preds_flat)

    # 计算其他评估指标
    precision, recall, f1, _ = precision_recall_fscore_support(labels_flat, preds_flat, average='macro')
    cls_report = classification_report(labels_flat, preds_flat, digits=4)

    data = {
        'True': labels_flat.tolist(),
        'Pred': preds_flat.tolist()
    }
    import pandas as pd
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(PREDICT_PATH, "test_results_csv.csv"), index=False)

    print("Evaluation on Test set:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(cls_report)

    save_txt_path = os.path.join(PREDICT_PATH, "results.txt")
    with open(save_txt_path, 'w') as f:
        f.write("Evaluation on Test set:\n")
        f.write(f"Accuracy:  {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"F1-score:  {f1:.4f}\n")
        f.write(cls_report)

    # 计算并绘制混淆矩阵
    cm = confusion_matrix(labels_flat, preds_flat)

    # 开始绘图
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(RELATION_LABELS))
    plt.xticks(tick_marks, RELATION_LABELS, rotation=45, ha='right')
    plt.yticks(tick_marks, RELATION_LABELS)

    thresh = cm.max() / 2  # 设置阈值，调整文本颜色
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    # 保存混淆矩阵图片
    confusion_matrix_path = os.path.join(PREDICT_PATH, "confusion_matrix.png")
    plt.savefig(confusion_matrix_path)

    print(f"混淆矩阵已保存至 {confusion_matrix_path}")