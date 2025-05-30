from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from typing import Callable

# api inference
from openai import OpenAI
from tiktoken import encoding_for_model

# finetuning
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)
from trl import (
    SFTConfig,
    SFTTrainer,
    DataCollatorForCompletionOnlyLM,
)
from peft import LoraConfig
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score
import torch
from transformers import Trainer, TrainingArguments
from transformers.integrations import WandbCallback
import wandb

from llm_survey_prediction.prompt import LLAMA_INSTRUCT_CHAT_TEMPLATE

class ModelWrapper(ABC):
    @abstractmethod
    def get_params(self):
        pass

    @abstractmethod
    def fit(self, *args, **kwargs):
        pass


class ClassificationWrapper(ModelWrapper):
    def __init__(self):
        self.classes_: list | None = None

    def predict(self, X: pd.DataFrame) -> pd.Series:
        predict_proba = self.predict_proba(X)
        y_proba_df = pd.DataFrame(predict_proba, columns=self.classes_)
        y_pred = list(y_proba_df.idxmax(axis=1))
        return pd.Series(y_pred)

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        pass


class XGBoostClassifier(ClassificationWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = XGBClassifier(*args, **kwargs)
        self.le = LabelEncoder()

    def get_params(self):
        return self.model.get_params()

    def fit(self, X: pd.DataFrame, y: pd.Series):
        y_encoded = self.le.fit_transform(y)
        self.classes_ = self.le.classes_
        return self.model.fit(X, y_encoded)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)


class OpenAIClassifier(ClassificationWrapper):
    def __init__(
        self, client: OpenAI, model: str, prompt_func: Callable, random_state: int = 24
    ):
        self.classes_: list | None = None
        self.client = client
        self.model = model
        self.prompt_func = prompt_func
        self.random_state = random_state

    def get_params(self):
        return {
            "openai_model": self.model,
            "prompt_func": self.prompt_func.func.__name__
            if hasattr(self.prompt_func, "func")
            else self.prompt_func.__name__,
            "random_state": self.random_state,
        }

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.classes_ = y.unique()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        encoder = encoding_for_model(self.model)
        label_tokens = {
            encoder.decode([encoder.encode(label)[0]]): label for label in self.classes_
        }

        prompts = self.prompt_func(X=X, y=None)

        # Initialize df_results with the correct shape and columns
        df_results = pd.DataFrame(
            0, index=range(len(X)), columns=self.classes_, dtype=float
        )

        for i, prompt in tqdm(enumerate(prompts), total=len(prompts)):
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=prompt,
                logprobs=True,
                top_logprobs=20,
                max_completion_tokens=1,
                seed=self.random_state,
            )

            probas = {
                label_tokens[logprob.token]: np.exp(logprob.logprob)
                for logprob in resp.choices[0].logprobs.content[0].top_logprobs
                if logprob.token in label_tokens
            }
            for label, proba in probas.items():
                df_results.loc[i, label] = proba

        return np.array(df_results)


# NOTE: per_device_train_batch_size assumes 40GB GPU RAM
MODEL_SPECIFIC_CONFIG = {
    # LLAMA
    "meta-llama/Llama-3.2-1B-Instruct": {
        "per_device_train_batch_size": 1,
        "instruction_template": "<|start_header_id|>user<|end_header_id|>\n\n",
        "response_template": "<|start_header_id|>assistant<|end_header_id|>\n\n",
    },
    "meta-llama/Llama-3.2-3B-Instruct": {
        "per_device_train_batch_size": 1,
        "instruction_template": "<|start_header_id|>user<|end_header_id|>\n\n",
        "response_template": "<|start_header_id|>assistant<|end_header_id|>\n\n",
    },
    "meta-llama/Llama-3.1-8B-Instruct": {
        "per_device_train_batch_size": 1,
        "instruction_template": "<|start_header_id|>user<|end_header_id|>\n\n",
        "response_template": "<|start_header_id|>assistant<|end_header_id|>\n\n",
    },
    # QWEN
    "Qwen/Qwen2.5-0.5B-Instruct": {
        "per_device_train_batch_size": 1,
        "instruction_template": "<|im_start|>user\n",
        "response_template": "<|im_start|>assistant\n",
    },
    "Qwen/Qwen2.5-1.5B-Instruct": {
        "per_device_train_batch_size": 1,
        "instruction_template": "<|im_start|>user\n",
        "response_template": "<|im_start|>assistant\n",
    },
    "Qwen/Qwen2.5-3B-Instruct": {
        "per_device_train_batch_size": 1,
        "instruction_template": "<|im_start|>user\n",
        "response_template": "<|im_start|>assistant\n",
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "per_device_train_batch_size": 1,
        "instruction_template": "<|im_start|>user\n",
        "response_template": "<|im_start|>assistant\n",
    },
}


class FinetuningClassifier(ClassificationWrapper):
    def __init__(
        self,
        model_id: str,
        prompt_func: Callable,
        batch_size: int = 8,
        n_epochs: int = 3,
        quantization_config: BitsAndBytesConfig | None = None,
        lora_config: LoraConfig | None = None,
        random_state: int = 24,
    ):
        assert model_id in MODEL_SPECIFIC_CONFIG, f"Model {model_id} not supported"

        self.classes_: list | None = None
        self.model_id = model_id
        self.prompt_func = prompt_func
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.lora_config = lora_config
        self.quantization_config = quantization_config
        self.random_state = random_state

        transformers.set_seed(self.random_state)

    def get_params(self):
        return {
            "model_id": self.model_id,
            "prompt_func": self.prompt_func.func.__name__
            if hasattr(self.prompt_func, "func")
            else self.prompt_func.__name__,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "quantization_config": self.quantization_config.to_dict()
            if self.quantization_config
            else {},
            "lora_config": self.lora_config.to_dict() if self.lora_config else {},
            "random_state": self.random_state,
        }

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.classes_ = y.unique()
        self.X_train = X
        self.y_train = y

    def predict_proba(self, X: pd.DataFrame, y: pd.Series | None = None) -> np.ndarray:
        # y is needed for evaluation logging
        self.X_test = X
        self.y_test = y

        text_train = self.prompt_func(X=self.X_train, y=self.y_train)
        text_test = self.prompt_func(X=self.X_test, y=self.y_test)

        dataset = DatasetDict(
            {
                "train": Dataset.from_dict({"messages": text_train}),
                "test": Dataset.from_dict({"messages": text_test}),
            }
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if getattr(tokenizer, "pad_token_id") is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        if self.quantization_config:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=self.quantization_config,
            )
        else:
            # load in bfloat16 if no quantization is specified
            model = AutoModelForCausalLM.from_pretrained(
                
                self.model_id,
                torch_dtype=torch.bfloat16,
            )
        if getattr(model.config, "pad_token_id") is None:
            model.config.pad_token_id = tokenizer.pad_token_id

        first_token_ids = [
            tokenizer.encode(label, add_special_tokens=False)[0]
            for label in self.classes_
        ]

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            # Return dist of relevant tokens
            return logits[:, :, first_token_ids]

        y_probas = []

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            labels = labels[:, 1:]
            logits = logits[:, :-1]

            logits_first_token_dist = np.array(
                [
                    [
                        logit_id
                        for logit_id, label_id in zip(logit_ids, label_ids)
                        if label_id != -100
                    ][0]
                    for logit_ids, label_ids in zip(logits, labels)
                ]
            )

            logits_first_token_dist_exp = np.exp(logits_first_token_dist)
            normalized_logits_first_token_dist = (
                logits_first_token_dist_exp
                / logits_first_token_dist_exp.sum(axis=1, keepdims=True)
            )

            y_probas.append(normalized_logits_first_token_dist)

            df_y_proba = pd.DataFrame(
                normalized_logits_first_token_dist, columns=self.classes_
            )

            wandb.log(
                data={
                    "y_proba": wandb.Table(data=df_y_proba),
                }
            )

            # -100 is a default value for ignore_index used by DataCollatorForCompletionOnlyLM
            mask = labels == -100
            labels[mask] = tokenizer.pad_token_id

            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            if self.model_id.startswith(
                "Qwen/"
            ):  # NOTE: Qwen models have \n at the end of each label (probably bug?)
                decoded_labels = [label.strip("\n") for label in decoded_labels]

            decoded_preds = list(df_y_proba.idxmax(axis=1))

            return {
                "accuracy": accuracy_score(y_pred=decoded_preds, y_true=decoded_labels)
            }

        collator = DataCollatorForCompletionOnlyLM(
            tokenizer=tokenizer,
            instruction_template=MODEL_SPECIFIC_CONFIG[self.model_id][
                "instruction_template"
            ],
            response_template=MODEL_SPECIFIC_CONFIG[self.model_id]["response_template"],
        )

        max_seq_length = (
            max(
                [
                    len(tokenizer.apply_chat_template(messages, tokenize=True))
                    for messages in text_train + text_test
                ]
            )
            + 10  # NOTE: 10 is a buffer and might not be needed
        )
        print(f"Setting `max_seq_length` to {max_seq_length}")

        per_device_train_batch_size = MODEL_SPECIFIC_CONFIG[self.model_id][
            "per_device_train_batch_size"
        ]

        gradient_accumulation_steps = self.batch_size // per_device_train_batch_size
        assert (
            per_device_train_batch_size * gradient_accumulation_steps == self.batch_size
        ), (
            f"Batch size {self.batch_size} is not divisible by per_device_train_batch_size {per_device_train_batch_size}"
        )

        training_args = SFTConfig(
            # batch size
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            # eval
            do_eval=True,
            eval_strategy="epoch",
            # saving
            save_strategy="epoch",
            save_total_limit=1,
            # other
            max_seq_length=max_seq_length,
            output_dir="./results",
            num_train_epochs=self.n_epochs,
        )

        trainer = SFTTrainer(
            tokenizer=tokenizer,
            model=model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            args=training_args,
            peft_config=self.lora_config,
            compute_metrics=compute_metrics,
            data_collator=collator,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        print(tokenizer.apply_chat_template(dataset["train"][0]["messages"], tokenize=False))

        trainer.evaluate()  # for zero-shot evaluation
        trainer.train()

        return y_probas[-1]


class BERTClassifier(ClassificationWrapper):
    def __init__(self, model_id: str, batch_size: int = 8, random_state: int = 24):
        self.model_id = model_id
        self.batch_size = batch_size
        self.random_state = random_state

        transformers.set_seed(self.random_state)

    def get_params(self):
        return {
            "model_id": self.model_id,
            "batch_size": self.batch_size,
            "random_state": self.random_state,
        }

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.classes_ = y.unique()
        self.X_train = X
        self.y_train = y

    def predict_proba(self, X: pd.DataFrame, y: pd.Series | None = None) -> np.ndarray:
        # y is needed for evaluation logging
        self.X_test = X
        self.y_test = y

        text_train = self._create_prompt(self.X_train, self.y_train)
        text_test = self._create_prompt(self.X_test, self.y_test)

        dataset = DatasetDict(
            {
                "train": Dataset.from_list(text_train),
                "test": Dataset.from_list(text_test),
            }
        )

        id2label = {i: label for i, label in enumerate(self.classes_)}
        label2id = {label: i for i, label in enumerate(self.classes_)}

        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if getattr(tokenizer, "pad_token_id") is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        def preprocess(examples):
            return tokenizer(
                examples["text"], padding="max_length", truncation=False
            ) | {"label": [label2id[label] for label in examples["label"]]}

        dataset = dataset.map(preprocess, batched=True)

        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id,
            num_labels=len(self.classes_),
            id2label=id2label,
            label2id=label2id,
            torch_dtype=torch.bfloat16,
        )
        if getattr(model.config, "pad_token_id") is None:
            model.config.pad_token_id = tokenizer.pad_token_id

        training_args = TrainingArguments(
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=1,
            # eval
            do_eval=True,
            eval_strategy="epoch",
            # saving
            save_strategy="epoch",
            save_total_limit=1,
            # dtype
            bf16=True,
            bf16_full_eval=True,
            # other
            output_dir="./results",
            num_train_epochs=3,
            push_to_hub=False,
        )

        y_probas = []

        def compute_metrics(eval_pred):
            pred_logits, labels = eval_pred
            predictions = np.argmax(pred_logits, axis=1)
            preds_exp = np.exp(pred_logits)
            normalized_preds = preds_exp / preds_exp.sum(axis=1, keepdims=True)
            y_probas.append(normalized_preds)
            wandb.log(
                data={
                    "y_proba": wandb.Table(
                        data=pd.DataFrame(normalized_preds, columns=self.classes_)
                    ),
                }
            )
            return {"accuracy": accuracy_score(y_pred=predictions, y_true=labels)}

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            compute_metrics=compute_metrics,
            data_collator=transformers.DataCollatorWithPadding(
                tokenizer=tokenizer,
                padding="longest",
            ),
        )
        trainer.train()
        return y_probas[-1]

    def _create_prompt(self, X: pd.DataFrame, y: pd.Series):
        prompts = []
        print("Creating prompts...")
        for (_, row), (_, label) in tqdm(zip(X.iterrows(), y.items()), total=len(X)):
            prompt = {
                "text": "\n".join([f"{k} {v}" for k, v in row.items()]),
                "label": label,
            }
            prompts.append(prompt)
        return prompts
