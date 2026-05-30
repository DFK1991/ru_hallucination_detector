from transformers import TrainingArguments, Trainer
from dataset import load_dataset_splits
from model import HallucDetector
from metrics import compute_metrics
from dataclasses import dataclass, field
from transformers import AutoTokenizer
from transformers import set_seed
from transformers import DataCollatorWithPadding
from preprocess import preprocess
import torch

set_seed(42)

# Config
@dataclass
class Config:
    bs: int = 16
    MODEL_NAME: str = 'cointegrated/rubert-tiny2'
    lr: float = 3e-4
    lrmf: float = 0.2
    max_length: int = 384
    lossft: str = "FL"
    cl_layers: int = 3
    dor: float = 0.4
    wd: float = 0.001
    wur: float = 0.1
    epochs: int = 10
    lr_sch: str = "linear"
    pools_list: list['str'] = field(default_factory=lambda: ['attn'])
    use_CLS: bool = False
    use_numfeatslin: bool = False
    output_dir: str = './hallucination_model'

TConfig = Config()

# loading dataset
ds_filename = '../data/ds_v0'

dataset, dict_idx2label, dict_label2idx = load_dataset_splits(ds_filename)

def add_labels(example):
    example["labels"] = (dict_label2idx[example["aug_type"]])
    return example

dataset = dataset.map(add_labels)

# preprocess dataset
tokenizer = AutoTokenizer.from_pretrained(TConfig.MODEL_NAME)
dataset = dataset.map(lambda x: preprocess(x, tokenizer, TConfig.max_length), batched=False)
dataset = dataset.remove_columns(["context", "evidence_aug", "aug_type"])

# init the model
model = HallucDetector(lossft=TConfig.lossft, pools_list=TConfig.pools_list, \
                       cl_layers=TConfig.cl_layers, dor=TConfig.dor, model_name=TConfig.MODEL_NAME, num_labels=len(dict_label2idx))

# setting lr
param_groups = [
    {"params": [p for n, p in model.named_parameters() if ("classifier." in n)], "lr": TConfig.lr, "weight_decay": TConfig.wd},
    {"params": [p for n, p in model.named_parameters() if (".encoder." in n or ".pooler." in n or "attn" in n)], "lr": TConfig.lr*TConfig.lrmf, "weight_decay": TConfig.wd},
    {"params": [p for n, p in model.named_parameters() if (".embeddings." in n)], "lr": TConfig.lr*(TConfig.lrmf)**2, "weight_decay": TConfig.wd}
]

optimizer = torch.optim.AdamW(param_groups)

# set up the trainer
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir=TConfig.output_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=TConfig.lr,
    per_device_train_batch_size=TConfig.bs,
    per_device_eval_batch_size=TConfig.bs,
    num_train_epochs=TConfig.epochs,
    weight_decay=TConfig.wd,
    warmup_ratio=TConfig.wur,
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1_macro",  
    greater_is_better=True,
    report_to="none",
    lr_scheduler_type=TConfig.lr_sch
    )

data_collator = DataCollatorWithPadding(tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    optimizers=(optimizer, None),
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=lambda x: compute_metrics(x, dict_idx2label),
    data_collator=data_collator
    )

# training
trainer.train()

# test eval
test_metrics = trainer.evaluate(dataset["test"])

print("\n============ TEST METRICS ============")
for k, v in test_metrics.items():
    print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")