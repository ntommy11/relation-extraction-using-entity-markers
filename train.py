# import machine learning modules
from sklearn.metrics import f1_score, confusion_matrix

import numpy as np
import random
import os

# import torch and its applications
import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

# import transformers and its applications
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    AdamW,
    get_cosine_schedule_with_warmup,
)

# custom library
from dataset import load_train_dataset
from model import StartTokenModel
from metrics import compute_metrics


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministric = True
    torch.backends.cudnn.benchmark = True


def train(Trainer):
    # load model
    model = StartTokenModel()

    # load dataset
    train_dir = "./data/train/train.csv"
    train_dataset, eval_dataset = load_train_dataset(
        train_dir=train_dir, tokenizer=model.tokenizer
    )

    # device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device)

    training_args = TrainingArguments(
        output_dir="./results",  # output directory
        save_total_limit=5,  # number of total save model.
        save_steps=500,  # model saving step.
        num_train_epochs=5,  # total number of training epochs
        learning_rate=5e-5,  # learning_rate
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=16,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir="./logs",  # directory for storing logs
        logging_steps=100,  # log saving step.
        evaluation_strategy="steps",  # evaluation strategy to adopt during training
        # `no`: No evaluation during training.
        # `steps`: Evaluate every `eval_steps`.
        # `epoch`: Evaluate every end of epoch.
        eval_steps=500,  # evaluation step.
        load_best_model_at_end=True,
    )
    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=eval_dataset,  # evaluation dataset
        compute_metrics=compute_metrics,  # define metrics function
    )

    # train model
    trainer.train()
    model.save_pretrained("./best_model")


def main():
    train()


if __name__ == "__main__":
    main()
