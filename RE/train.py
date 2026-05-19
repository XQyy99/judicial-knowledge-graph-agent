import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments,AutoModelForTokenClassification, AutoTokenizer
from typing import List, Dict
os.environ["WANDB_MODE"] = "disabled"
# ==============================
# 参数设定
# ==============================
DATA_PATH = r"D:\dl\毕业设计\xxcq\final_train.json"  # 您的数据文件路径
MODEL_NAME = r"D:\dl\bert-crf-token_classification_ner-master\pretrain\chinese-electra-base-discriminator" 
MAX_LENGTH = 512
BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 3e-5
usemodel="electra"
OUTPUT_DIR = f"D:/dl/毕业设计/xxcq/o1/RE/relation_extraction_output/{usemodel}/lr_{LEARNING_RATE}_epo_{EPOCHS}_bs_{BATCH_SIZE}_ML_{MAX_LENGTH}_final"

# 定义关系集合（包括NA即无关系分类，若需要）
RELATION_LABELS = ["sell_drugs_to", "traffic_in", "possess", "provide_shelter_for", "NA"]
label2id = {label: i for i, label in enumerate(RELATION_LABELS)}
id2label = {i: label for label, i in label2id.items()}

# ==============================
# 数据加载与预处理
# 假设数据为:
# [
#   {
#     "articleId": 2,
#     "sentId": 20,
#     "entityMentions": [
#       {"end": 20, "start": 17, "text": "林某某", "label": "Nh"},
#       ...
#     ],
#     "sentText": "湛江市麻章区人民检察院指控，...",
#     "relationMentions": [
#       {"e1start": 17, "e21start": 48, "em1Text": "林某某", "em2Text": "海洛因", "label": "traffic_in"},
#       ...
#     ]
#   },
#   ...
# ]
# ==============================

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

data = load_data(DATA_PATH)

# ==============================
# 数据转换逻辑
# 将每个句子转换为多条训练样本：对每个relationMention
# 都构建一个 (sentence, entity1, entity2, label) 的实例
# 如果需要负样本（NA），可以对未标注关系的实体对构建NA样本
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

samples = build_samples(data)

# ==============================
# 数据集定义
# 将 (sentence, entity1, entity2, label) 编码成BERT输入
# 为了帮助模型聚焦实体，可考虑在实体周围加上特殊标记，但这里简化处理
# 仅在句子中进行编码，然后让模型学会区分关系
# ==============================

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

#tokenizer = BertTokenizerFast.from_pretrained(r"D:\dl\bert-crf-token_classification_ner-master\pretrain\bert-base-chinese")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 将全部数据分成训练集与验证集（简单分割）
train_size = int(0.8 * len(samples))
train_samples = samples[:train_size]
val_samples = samples[train_size:]

train_dataset = RelationDataset(train_samples, tokenizer, max_length=MAX_LENGTH)
val_dataset = RelationDataset(val_samples, tokenizer, max_length=MAX_LENGTH)

# ==============================
# 模型及训练器定义
# ==============================

model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(RELATION_LABELS)
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",  # 确保与evaluation_strategy保持一致
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    num_train_epochs=EPOCHS,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

# 定义评价函数
def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids
    accuracy = (preds == labels).mean()
    return {"accuracy": accuracy}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# ==============================
# 开始训练
# ==============================
trainer.train()

# 训练完成后可保存模型
trainer.save_model(OUTPUT_DIR)
import matplotlib.pyplot as plt
# ==============================
# 训练完成后，绘制训练过程的loss和acc曲线
# ==============================
log_history = trainer.state.log_history

train_loss_values = []
eval_loss_values = []
eval_acc_values = []
train_steps = []
eval_epochs = []

for log in log_history:
    # 训练过程的loss日志 (在 logging_steps 或每步记录)
    if "loss" in log and "learning_rate" in log:
        train_loss_values.append(log["loss"])
        if "step" in log:
            train_steps.append(log["step"])

    # 验证过程的loss和acc日志 (在 epoch 结束时)
    if "eval_loss" in log:
        eval_loss_values.append(log["eval_loss"])
        if "eval_accuracy" in log:
            eval_acc_values.append(log["eval_accuracy"])
        if "epoch" in log:
            eval_epochs.append(log["epoch"])

# 创建输出目录（如果不存在）
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 绘制训练loss曲线
if train_loss_values and train_steps:
    plt.figure(figsize=(10,5))
    plt.title("Training Loss")
    plt.plot(train_steps, train_loss_values, label="train_loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_loss.png"))
    plt.close()

# 绘制验证loss曲线
if eval_loss_values and eval_epochs:
    plt.figure(figsize=(10,5))
    plt.title("Evaluation Loss")
    plt.plot(eval_epochs, eval_loss_values, label="eval_loss", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "eval_loss.png"))
    plt.close()

# 绘制验证accuracy曲线
if eval_acc_values and eval_epochs:
    plt.figure(figsize=(10,5))
    plt.title("Evaluation Accuracy")
    plt.plot(eval_epochs, eval_acc_values, label="eval_accuracy", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "eval_accuracy.png"))
    plt.close()

print("训练过程的loss和acc曲线已保存在:", OUTPUT_DIR)