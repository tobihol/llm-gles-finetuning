from helpers.eval_pipeline import classification_experiment
from helpers.data_preprocessing import DatasetGLES2017NoPartyId
from llm_survey_prediction.model_wrappers import (
    XGBoostClassifier,
    OpenAIClassifier,
    FinetuningClassifier,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, BaseCrossValidator
from pathlib import Path
from datetime import datetime
import itertools
import os
from openai import OpenAI
from peft import LoraConfig, TaskType
from transformers import BitsAndBytesConfig
from functools import partial
from llm_survey_prediction.prompt import prompt_w_system, InstructionPromptGLES2017

# CONFIG

SEED = 24

WANDB_CONFIG = {
    # INSERT WANDB ACCOUNT
    # "entity": "username",
    # "project": "project-name",
}

DATASET = DatasetGLES2017NoPartyId(
    path="datasets/ZA6835_v1-0-0.sav",
)
TARGET_COL = "Wahlentscheidung"

CROSS_VALIDATOR = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED,
)

DEBUG = False
# CONFIG END

assert isinstance(CROSS_VALIDATOR, BaseCrossValidator)

# experiment config for logging. (Models are logged separately.)
exp_config = {
    "experiment": f"{Path(__file__).stem}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",  # name of this file
    "cross_validation_config": {
        "validator_name": CROSS_VALIDATOR.__class__.__name__,
        **CROSS_VALIDATOR.__getstate__(),
    },
    "seed": SEED,
    "debug": DEBUG,
}
# if DEBUG:
#     DATASET._df_raw = DATASET._df_raw[:20]  # small sample for testing

splits_gen = DATASET.classification_splits(
    target_col=TARGET_COL,
    splits=CROSS_VALIDATOR,
    train_mask=(
        lambda df: df["Erwerbstätigkeit"].isin(
            [
                "Student",
                "Schüler",
            ]
        )
    ),
)

def init_models():
    # LLM models
    finetuning_models = []
    for (
        model_id,
        batch_size,
        n_epochs,
        quantization_config,
        lora_config,
    ) in itertools.product(
        # models
        [
            "meta-llama/Llama-3.2-1B-Instruct",
            "meta-llama/Llama-3.2-3B-Instruct",
            "meta-llama/Llama-3.1-8B-Instruct",
            "Qwen/Qwen2.5-0.5B-Instruct",
            "Qwen/Qwen2.5-1.5B-Instruct",
            "Qwen/Qwen2.5-3B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct",
        ],
        # batch sizes
        [
            1,
        ],
        # n_epochs
        [
            3,
        ],
        # quantization config
        [
            None
        ],
        # lora config
        [
            LoraConfig(
                r=256,  # change to the max that is in vram capacity
                lora_alpha=8,
                use_rslora=True,  # see https://arxiv.org/abs/2312.03732
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                target_modules="all-linear",
            ),
        ],
    ):
        finetuning_model = FinetuningClassifier(
            model_id=model_id,
            prompt_func=partial(prompt_w_system, system_prompt=InstructionPromptGLES2017.system.value),
            batch_size=batch_size,
            n_epochs=n_epochs,
            quantization_config=quantization_config,
            lora_config=lora_config,
            random_state=SEED,
        )
        finetuning_models.append(finetuning_model)

    openai_client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
    )
    openai_model = OpenAIClassifier(
        client=openai_client,
        model="gpt-4o-2024-11-20",
        prompt_func=partial(prompt_w_system, system_prompt=InstructionPromptGLES2017.system.value),
        random_state=SEED,
    )

    models_llm = [
        openai_model,
        CatBoostClassifier(random_state=SEED),
    ] 
    + finetuning_models

    # Traditional models
    models_trad = [
        RandomForestClassifier(random_state=SEED),
        LogisticRegression(
            random_state=SEED,
            max_iter=1000,
        ),
        XGBoostClassifier(random_state=SEED),
    ]

    return (models_trad, models_llm)

# EVALUATION
classification_experiment(
    dataset_splits=splits_gen,
    model_init_func=init_models,
    exp_config=exp_config,
    wandb_config=WANDB_CONFIG,
)
