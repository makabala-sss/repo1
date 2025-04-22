# Select CUDA device index
import dataclasses
import os
from pyparsing import Any
from transformers import AutoConfig, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from datasets import load_from_disk, Audio
import evaluate
from transformers import WhisperFeatureExtractor, WhisperTokenizer
import argparse
from dataclasses import dataclass
from typing import Dict, List, Union
import torch

from datasets import load_dataset, DatasetDict
os.environ["CUDA_VISIBLE_DEVICES"] = "0,3"
model_name_or_path = "openai/whisper-large-v3"
language = "french"
language_abbr = "fr"
task = "transcribe"
dataset_name = "facebook/voxpopuli"

# Argument parsing
parser = argparse.ArgumentParser(description="Whisper full-parameter fine-tuning configuration")
parser.add_argument("--use_fp16", type=bool, default=False, help="Whether to use mixed precision training")
parser.add_argument("--model_size", type=str, default="large", help="Size of the model")

args = parser.parse_args()
if args.model_size == "large":
    model_name_or_path = "openai/whisper-large-v3"
elif args.model_size == "medium":
    model_name_or_path = "openai/whisper-medium"
elif args.model_size == "small":
    model_name_or_path = "openai/whisper-small"
else:
    print("wrong model_size, look code!")
    exit()

# Load dataset
common_voice = DatasetDict()
common_voice["train"] = load_dataset(dataset_name, language_abbr, split="train").select(range(11600))
common_voice["test"] = load_dataset(dataset_name, language_abbr, split="validation")

# Remove unnecessary columns
print(common_voice["train"][0])
common_voice = common_voice.remove_columns(
    ["accent", "audio_id", "gender", "speaker_id", "is_gold_transcript"]
)

# Initialize feature extractor and tokenizer
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name_or_path)
tokenizer = WhisperTokenizer.from_pretrained(model_name_or_path, language=language, task=task)
if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
processor = WhisperProcessor.from_pretrained(model_name_or_path, language=language, task=task)

def prepare_dataset(batch):
    # load and resample audio data from 48kHz to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from the input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    # encode target text to label ids
    batch["labels"] = tokenizer(batch["normalized_text"]).input_ids
    return batch

common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

path = "./map_data_fr_default_large"  # replace with your actual path

# If path does not exist, process and save dataset to disk
if 0:
    common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=4)
    common_voice.save_to_disk(path)
else:
    # Otherwise, load dataset from disk
    common_voice = common_voice.load_from_disk(path)

model = WhisperForConditionalGeneration.from_pretrained(
    model_name_or_path, device_map="auto", cache_dir="/data02/wwh/data/hf_cache/"
)
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt", padding=True)
        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's appended later anyway
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

# Evaluation metric
wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with pad token
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # Decode predictions and labels
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Compute WER
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=f"/home/WangYonghe_Grp/Wenhao_54/wwh/code/Reft/ASR/fr/models/{args.model_size}/finetune-full-parameter-epoch",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    warmup_ratio=0.1,
    num_train_epochs=10,
    evaluation_strategy="no",
    fp16=args.use_fp16,
    per_device_eval_batch_size=2,
    predict_with_generate=True,  # important: enable generation for predictions
    generation_max_length=128,
    logging_steps=10,
    remove_unused_columns=False,
    save_strategy="epoch",  # save once per epoch
    save_total_limit=30
)

# Initialize Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor
)

# Start training
trainer.train()
