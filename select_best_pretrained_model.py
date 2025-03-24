import os
import random
import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForSequenceClassification, AutoConfig

import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import datasets
import Getmetrics
import getDataset
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoConfig
class CustomModel(AutoModelForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 2  # 分类标签数量，二分类任务
        self.hidden_size = config.hidden_size
        self.model = AutoModelForSequenceClassification.from_pretrained(config._name_or_path, num_labels=self.num_labels)  # 加载预训练分类模型
        self.dropout = nn.Dropout(0.1)  # Dropout 层，用于防止过拟合
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)  # 分类器

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # 通过模型获取输出
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        # 获取模型的输出
        model_output = outputs[0]  # [batch_size, sequence_length, hidden_size]

        # 使用 [CLS] token 的输出进行分类
        logits = self.classifier(self.dropout(model_output[:, 0, :]))  # [CLS] token output

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # 分类任务使用交叉熵损失
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))  # 计算损失
            outputs = (loss,) + (logits,)  # 返回损失和logits
        
        return outputs  # 返回损失和预测结果


# 定义模型路径和名称
model_paths = [
    '/home/ubuntu/zhouyongzhu/model/esm2',
    '/home/ubuntu/zhouyongzhu/model/OntoProtein',
    '/home/ubuntu/zhouyongzhu/model/bert_bfd'
]
model_names = [ 'esm2','OntoProtein','bert_bfd']

for model_path, model_name in zip(model_paths, model_names):
    print('model_name:', model_name)
    print('model_path:', model_path)

    # 加载对应模型的 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print('Tokenizer has been loaded.')

    # 加载对应模型
    # 替换为具体的模型路径
    model = CustomModel.from_pretrained(model_path)

    print('Model has been loaded.')

    # 获取数据集
    test_dataset = getDataset.getDataset(modelpath=model_path, tkpath=model_path, datapath="/home/ubuntu/zhouyongzhu/data/test_data.csv", max_len=512)
    dataset = datasets.load_dataset("csv", cache_dir='/home/ubuntu/zhouyongzhu/data/cache',data_files="/home/ubuntu/zhouyongzhu/data/train_data.csv")
    dataset = dataset.shuffle(seed=702)
    
    # 分词序列
    tokenized_dataset = dataset.map(lambda x: tokenizer(
        ' '.join(x["sequence"]), max_length=512,
        padding="max_length", truncation=True))
    train_dataset = tokenized_dataset['train']

    # 输出目录
    outPutDir = f'output/{model_name}'
    os.makedirs(outPutDir, exist_ok=True)

    # 训练参数
    training_args = TrainingArguments(
        output_dir=str(outPutDir),
        num_train_epochs=20,
        per_device_train_batch_size=4,
        warmup_ratio=0.1,
        weight_decay=0.01,
        learning_rate=(2e-5),
        logging_dir=str(outPutDir) + os.sep + 'logs',
        logging_steps=50,  # 每50步记录一次日志
        save_steps=200,    # 每200步保存一次模型
        save_total_limit=3,
        seed=1020,
        remove_unused_columns=True,
        lr_scheduler_type="cosine_with_restarts",
        optim='adafactor',
        dataloader_drop_last=False,
        evaluation_strategy="steps",  # 按步数评估
        eval_steps=200,  # 每 200 步评估一次
        load_best_model_at_end=True,  # 训练结束时加载最优模型
        #metric_for_best_model="eval_loss",  # 最优模型的指标依据   不是准确率，没保存准确率最高的模型，你要准确率最高的模型，就要改成eval_accuracy
        metric_for_best_model="eval_accuracy",  # 最优模型的指标依据
        greater_is_better=True  # 指标越大越好
        )

    # 使用 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=Getmetrics.getMetrics,
    )

    trainer.train()
    # 显式保存最终的模型和分词器
    model.save_pretrained(outPutDir)
    tokenizer.save_pretrained(outPutDir)
    print(f"Final model and tokenizer saved to {outPutDir}")

    # 在训练过程中进行预测
    predictions = trainer.predict(test_dataset)
    res_dict = {
        'sequence': list(test_dataset['sequence']),
        'input_ids': list(test_dataset['input_ids']),
        'score': list(Getmetrics.getScore(predictions[0])[:, 1]),
        'predict_label': list(Getmetrics.getPredictLabel(predictions[0])),
        'y_label': list(predictions[1]),
        'check_label': list(test_dataset['label']),
    }

    # 保存预测结果和评估结果
    res_df = pd.DataFrame(res_dict)
    res_df.to_csv(outPutDir + os.sep + 'predict_res.csv', index=None)
    res_met_df = pd.DataFrame(predictions[2], index=[1])
    res_met_df = res_met_df.T
    res_met_df = res_met_df.rename(columns={1: 'Model'})
    res_met_df.to_csv(outPutDir + os.sep + 'metrics_res.csv')

    
