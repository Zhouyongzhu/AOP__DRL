import random
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoConfig
import os
import pandas as pd
from transformers import AutoTokenizer, TrainingArguments, Trainer, EarlyStoppingCallback
import datasets
import Getmetrics
import getDataset
import logging
import csv
from datasets import load_dataset
from sklearn.model_selection import GroupKFold  # 使用分组交叉验证
class TEXTModel(AutoModelForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 2
        self.hidden_size = config.hidden_size
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config._name_or_path, num_labels=self.num_labels
        )
        self.num_filters = 64  # 减少卷积核的数量
        self.kernel_sizes = [3]  # 只使用一个卷积核大小
        self.convs = nn.ModuleList([nn.Conv2d(1, self.num_filters, (kernel_size, self.hidden_size)) for kernel_size in self.kernel_sizes])
        self.dropout = nn.Dropout(0.2)  # 减少Dropout率
        self.fc = nn.Linear(len(self.kernel_sizes) * self.num_filters, self.num_labels)

    def forward(self, **kwargs):
        outputs = self.model(**kwargs)
        model_output = outputs.hidden_states[-1]
        model_output = model_output.unsqueeze(1)  
        conv_outputs = [torch.relu(conv(model_output)).squeeze(3) for conv in self.convs]
        pooled_outputs = [torch.max(conv_output, dim=2).values for conv_output in conv_outputs]
        feature_vector = torch.cat(pooled_outputs, dim=1)  
        logits = self.fc(self.dropout(feature_vector))
        return (logits,)
class LstmModel(AutoModelForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 2
        self.hidden_size = config.hidden_size
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config._name_or_path, num_labels=self.num_labels
        )
        self.num_layers = 2  # 减少LSTM层的数量
        self.bidirectional = False  # 使用单向LSTM
        self.dropout = nn.Dropout(0.5)  # 增加Dropout率
        self.lstm = nn.LSTM(
            self.hidden_size,
            self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            batch_first=True,
            dropout=0.5,
        )
        self.classifier = nn.Linear(
            self.hidden_size * 2 if self.bidirectional else self.hidden_size,
            self.num_labels,
        )

    def forward(self, **kwargs):
        outputs = self.model(**kwargs)
        hidden_states = outputs.hidden_states
        pooled_output = torch.stack(hidden_states, dim=1).mean(dim=1)  
        seq_lens = kwargs["attention_mask"].sum(dim=1).cpu()
        seq_lens, perm_idx = seq_lens.sort(0, descending=True)
        pooled_output = pooled_output[perm_idx]
        packed_input = nn.utils.rnn.pack_padded_sequence(
            pooled_output, seq_lens, batch_first=True
        )
        packed_output, (hn, cn) = self.lstm(packed_input)
        # 对于单向LSTM，只使用 hn[-1]
        forward_hidden_state = hn[-1]  # 取单向LSTM的最后一个隐藏状态
        hidden_states = forward_hidden_state  # 不需要拼接，因为是单向的
        _, unperm_idx = perm_idx.sort(0)
        hidden_states = hidden_states[unperm_idx]
        logits = self.classifier(self.dropout(hidden_states))
        return logits
class FCModel(AutoModelForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 2
        self.hidden_size = config.hidden_size
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config._name_or_path, num_labels=self.num_labels
        )
        self.dropout = nn.Dropout(0.5)  # 增加Dropout率
        self.fc = nn.Linear(self.hidden_size, self.num_labels)  

    def forward(self, **kwargs):
        outputs = self.model(**kwargs)
        model_output = outputs.hidden_states[-1]  
        cls_output = model_output[:, 0, :]  
        logits = self.fc(self.dropout(cls_output))  
        return (logits,)
class BiLSTMModel(AutoModelForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 2
        self.hidden_size = config.hidden_size
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config._name_or_path, num_labels=self.num_labels
        )
        self.num_layers = 2  # 减少LSTM层的数量
        self.bidirectional = True  # 使用双向LSTM
        self.dropout = nn.Dropout(0.5)  # 增加Dropout率
        self.lstm = nn.LSTM(
            self.hidden_size,
            self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            batch_first=True,
            dropout=0.5,
        )
        self.classifier = nn.Linear(
            self.hidden_size * 2 if self.bidirectional else self.hidden_size,
            self.num_labels,
        )

    def forward(self, **kwargs):
        outputs = self.model(**kwargs)
        hidden_states = outputs.hidden_states
        pooled_output = torch.stack(hidden_states, dim=1).mean(dim=1)  
        seq_lens = kwargs["attention_mask"].sum(dim=1).cpu()
        seq_lens, perm_idx = seq_lens.sort(0, descending=True)
        pooled_output = pooled_output[perm_idx]
        packed_input = nn.utils.rnn.pack_padded_sequence(
            pooled_output, seq_lens, batch_first=True
        )
        packed_output, (hn, cn) = self.lstm(packed_input)
        forward_hidden_state = hn[-2] if self.bidirectional else hn[-1]
        hidden_states = torch.cat((forward_hidden_state, hn[-1]), dim=1)  
        _, unperm_idx = perm_idx.sort(0)
        hidden_states = hidden_states[unperm_idx]
        logits = self.classifier(self.dropout(hidden_states))
        return logits
#types = ['LSTM', 'BiLSTM','FC','Text-CNN']
types = ['LSTM', 'BiLSTM','FC','Text-CNN']

# 加载模型和分词器
model_path = "/home/ubuntu/zhouyongzhu/model/esm2"
# 数据集路径
data_paths = [
    '/home/ubuntu/zhouyongzhu/data/p60.csv',
    '/home/ubuntu/zhouyongzhu/data/p70.csv',
    '/home/ubuntu/zhouyongzhu/data/p80.csv',
    '/home/ubuntu/zhouyongzhu/data/p90.csv'
]
for type in types:
    print(f"Start training {type} model")

    for data_path in data_paths:
        # 加载数据集
        dataset = load_dataset(
            "csv",
            cache_dir="/home/ubuntu/zhouyongzhu/data/cache",
            data_files=data_path,
        )
        
        # 获取训练集
        train_data = dataset["train"].shuffle(seed=702)
        
        # 获取partition列作为分组依据
        partition_values = train_data["partition"]

        # 五重分组交叉验证
        group_kfold = GroupKFold(n_splits=5)

        # 结果保存
        results = []

        # 遍历每一轮交叉验证
        for fold, (train_idx, val_idx) in enumerate(
            group_kfold.split(train_data, groups=partition_values)  # 使用分组划分
        ):
            if type == "LSTM":
                model = LstmModel.from_pretrained(model_path)
            elif type == "BiLSTM":
                model = BiLSTMModel.from_pretrained(model_path)
            elif type == "Text-CNN":
                model = TEXTModel.from_pretrained(model_path)
            elif type == "FC":
                model = FCModel.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)

            # 如果有多个GPU, 将模型分配到GPU 0 和 GPU 1上
            if torch.cuda.device_count() > 1:
                print(f"Using {torch.cuda.device_count()} GPUs!")
                model = model.to('cuda:0')


            # 根据索引分割数据集
            train_fold_data = train_data.select(train_idx)
            val_fold_data = train_data.select(val_idx)

            # 分词处理函数
            def tokenize_function(examples):
                return tokenizer(
                    examples["sequence"],
                    max_length=512,        # 缩短序列长度
                    padding="max_length",
                    truncation=True,
                )

            # 并行处理数据
            tokenized_train = train_fold_data.map(
                tokenize_function,
                batched=True,
                num_proc=4
            )

            tokenized_val = val_fold_data.map(
                tokenize_function,
                batched=True,
                num_proc=4
            )

            # 设置训练参数
            outPutDir = f"/home/ubuntu/zhouyongzhu/result/{type}/{os.path.basename(data_path)}_fold{fold + 1}"
            os.makedirs(outPutDir, exist_ok=True)

            training_args = TrainingArguments(
                output_dir=outPutDir,
                num_train_epochs=20,  # 设置较少的训练轮数，避免过拟合
                per_device_train_batch_size=8,  # 降低批量大小
                warmup_ratio=0.1 if type == "BiLSTM" else 0.15 if type == "Text-CNN" else 0.1 if type == "LSTM" else 0.05,  # 更合理的warmup比例
                weight_decay=0.01,
                learning_rate=3e-5 if type == "BiLSTM" else 2e-5 if type == "Text-CNN" else 1e-5 if type == "LSTM" else 5e-6,  # 更低的学习率
                lr_scheduler_type="linear",  # 使用线性学习率衰减
                load_best_model_at_end=False,  # 启用“在结束时加载最佳模型”
                save_strategy="no",  # 不保存模型
                save_steps=0,  # 不保存检查点
                metric_for_best_model="eval_accuracy",
                greater_is_better=True,
                logging_dir="./logs",
                logging_steps=50,  # 增加日志频率，监控训练过程
            )
         
            # 初始化Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_train,
                eval_dataset=tokenized_val,
                compute_metrics=Getmetrics.getMetrics,
            )

            # 训练并评估
            trainer.train()

            # 预测验证集
            predictions = trainer.predict(tokenized_val)

            # 结果处理
            predicted_scores = torch.sigmoid(torch.tensor(predictions.predictions))
            
            res_dict = {
                "sequence": list(val_fold_data["sequence"]),
                "input_ids": list(tokenized_val["input_ids"]),
                "score": list(predicted_scores[:, 1].numpy()),
                "predict_label": list(Getmetrics.getPredictLabel(predicted_scores.numpy())),
                "y_label": list(predictions.label_ids),
                "check_label": list(val_fold_data["label"]),
            }

            # 保存结果到CSV文件
            res_df = pd.DataFrame(res_dict)
            res_df.to_csv(os.path.join(outPutDir, "predict_res.csv"), index=None)
            res_met_df = pd.DataFrame(predictions[2], index=[1]).T
            res_met_df.rename(columns={1: "Model"}, inplace=True)
            res_met_df.to_csv(os.path.join(outPutDir, "metrics_res.csv"))

            print(f"Fold {fold + 1} completed. Best eval loss: {trainer.state.best_metric}")