import os
import argparse
from datasets import load_dataset, DatasetDict, Audio, load_from_disk
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

# Set visible GPU devices
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
# Argument parsing
parser = argparse.ArgumentParser(description="Whisper full-parameter fine-tuning configuration")
parser.add_argument(
    "--model_size",
    type=str,
    default="small",
    help="model size: [small, medium, large]"
)
args = parser.parse_args()

# Select pretrained model based on size
model_config = {
    "small": "openai/whisper-small",
    "medium": "openai/whisper-medium",
    "large": "openai/whisper-large-v3"
}
if args.model_size not in model_config:
    print("Invalid model size, please choose from [small, medium, large]")
    exit()

model_name_or_path = model_config[args.model_size]

# Dataset and task configuration
language_abbr = "uk"
task = "translate"
target_language = "english"
dataset_name = "oovword/speech-translation-uk-en"

# Load feature extractor and tokenizer
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name_or_path)
tokenizer = WhisperTokenizer.from_pretrained(
    model_name_or_path, 
    language=target_language,
    task=task
)

# Data preprocessing function
def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(
        audio["array"], 
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    
    batch["labels"] = tokenizer(
        batch["en"],
        padding="max_length", 
        truncation=True, 
        max_length=448
    ).input_ids
    
    return batch

# Load or prepare dataset
data_path = f"./{args.model_size}"
if not os.path.exists(data_path):
    dataset = DatasetDict()
    dataset["train"] = load_dataset(dataset_name, split="train")
    dataset["test"] = load_dataset(dataset_name, split="validation")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    dataset = dataset.map(
        prepare_dataset,
        remove_columns=dataset.column_names["train"],
        num_proc=8
    )
    dataset.save_to_disk(data_path)
else:
    dataset = load_from_disk(data_path)

# Initialize processor and model
processor = WhisperProcessor.from_pretrained(
    model_name_or_path,
    language=target_language,
    task=task
)

model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path,device_map="auto",
)
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

# Print total and trainable parameter counts
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params}, Trainable parameters: {trainable_params} ({trainable_params/total_params:.2%})")

import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union

# Data collator for padding inputs and labels
dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(
        self,
        features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # Prepare input features batch
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # Prepare and pad label features
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            return_tensors="pt",
            padding=True
        )
        # Mask padding tokens for loss computation
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1),
            -100
        )

        # Remove decoder start token if present at first position
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        
        batch["labels"] = labels
        return batch

# Instantiate data collator
data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

# Evaluation metrics function
from jiwer import wer
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer_score = wer(label_str, pred_str)
    return {"wer": wer_score}

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=f"./{args.model_size}/whisper-{args.model_size}-full-finetune",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    warmup_ratio=0.1,
    num_train_epochs=5,
    evaluation_strategy="no",
    fp16=False,
    per_device_eval_batch_size=8,
    generation_max_length=128,
    logging_steps=10,
    report_to=[],
    save_strategy="steps",
    save_steps=500,
    remove_unused_columns=False,
    label_names=["labels"]
)

# Initialize Trainer
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor
)

trainer.train()
