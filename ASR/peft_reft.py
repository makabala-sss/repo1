# Select CUDA device index
import os
from transformers import AutoConfig, DataCollatorForSeq2Seq, WhisperForConditionalGeneration
from pyreft import ReftSrTrainer
import pyreft
import torch
import numpy as np
from pyreft import ReftAudioDatasetFixed, ReftAudioDataCollator
import argparse
from datasets import DatasetDict
# Set the visible CUDA devices (use GPU 1)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Default model and task settings
model_name_or_path = "openai/whisper-small"
language = "french"
language_abbr = "uk"
task = "translate"  # Set task type to translation
target_language = "english"  # Specify target language
dataset_name = "oovword/speech-translation-uk-en"  # Use CoVoST2 uk-to-English dataset

# Argument parser for training parameters
parser = argparse.ArgumentParser(description="whisper training")
parser.add_argument("--intervention", type=str, default="Loreft", help="Intervention type: [Loreft, Direft, Lora]")
parser.add_argument("--model_size", type=str, default="small", help="Model size: small, medium, large")
parser.add_argument("--layers", type=int, default=3, help="Number of intervention layers")
parser.add_argument("--reft_r", type=int, default=4, help="Low-rank dimension for ReFT")
parser.add_argument("--lora_r", type=int, default=8, help="Rank for LoRA low-rank matrices")
parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA scaling factor")
args = parser.parse_args()
 
# Choose model checkpoint based on model_size argument
if args.model_size == "small":
    model_name_or_path = "openai/whisper-small"
elif args.model_size == "medium":
    model_name_or_path = "openai/whisper-medium"
elif args.model_size == "large":
    model_name_or_path = "openai/whisper-large-v3"
else:
    print("Please choose one of small, medium, large")
    exit()

# Load Whisper feature extractor
from transformers import WhisperFeatureExtractor
from datasets import Audio, load_dataset
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name_or_path)

# Initialize the tokenizer for translation task
from transformers import WhisperTokenizer
tokenizer = WhisperTokenizer.from_pretrained(
    model_name_or_path,
    language=target_language,  # Target language setting
    task=task  # Translation task
)

# Function to prepare dataset examples
def prepare_dataset(batch):
    audio = batch["audio"]
    # Extract input features (resample stays at 16kHz)
    batch["input_features"] = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    # Use English translation text as labels
    batch["input_ids"] = tokenizer(
        batch["en"],  # Use the translated text
        padding="max_length",
        truncation=True,
        max_length=448
    ).input_ids
    batch["labels"] = batch["input_ids"]
    return batch

# Load and preprocess CoVoST2 French-to-English dataset
data_train = load_dataset(
    dataset_name, split="train"
)
data_val = load_dataset(
    dataset_name, split="validation"
)
data_train = data_train.cast_column("audio", Audio(sampling_rate=16000))
data_val = data_val.cast_column("audio", Audio(sampling_rate=16000))
common_voice = DatasetDict()
common_voice["train"] = data_train.map(
    prepare_dataset, remove_columns=data_train.column_names, num_proc=8
)
common_voice["validation"] = data_val.map(
    prepare_dataset, remove_columns=data_val.column_names, num_proc=8
)

# Initialize processor for generation (with correct language and task)
from transformers import WhisperProcessor
processor = WhisperProcessor.from_pretrained(
    model_name_or_path,
    language=target_language,
    task=task
)
layers_to_transform =[10]
# Setup ReFTAudioDataset for training and evaluation
train_dataset = ReftAudioDatasetFixed(
    task="sr",
    data_path=dataset_name,
    dataset=common_voice["train"],
    feature_extractor=feature_extractor,
    language_abbr=language_abbr,
    tokenizer=tokenizer,
    data_split="train",
    seed=42,
    num_interventions=len(layers_to_transform),
    position='f1+l1',  # Intervene on first and last tokens
    share_weights=False,
    test_split="validation"
)

eval_dataset = ReftAudioDatasetFixed(
    task="sr",
    data_path=dataset_name,
    dataset=common_voice["validation"],
    feature_extractor=feature_extractor,
    tokenizer=tokenizer,
    data_split="test",
    seed=42,
    num_interventions=len(layers_to_transform),
    position='f1+l1',
    share_weights=False,
    test_split="validation"
)

# Load the Whisper model (automatically maps devices)
model = WhisperForConditionalGeneration.from_pretrained(
    model_name_or_path,
    device_map="auto"
)
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

# Configure intervention type
include_peft = True
intervention_type = args.intervention
if intervention_type == "Loreft":
    intervention = pyreft.LoreftIntervention
elif intervention_type == "Direft":
    intervention = pyreft.DireftIntervention
elif intervention_type == "AdaDoreft":
    intervention = pyreft.AdareftIntervention
else:
    intervention = pyreft.DireftIntervention
layers_to_transform = [10]
# Build ReFT configuration
reft_config = pyreft.ReftConfig(representations=[{
        "layer": l,
        "component": f"base_model.encoder.layers[{l}].self_attn.out_proj.output",
        "low_rank_dimension": args.reft_r,
        "intervention": intervention(
            embed_dim=model.config.hidden_size,
            low_rank_dimension=args.reft_r,
            dtype=torch.float
        )
    } for l in layers_to_transform]
)

# Optionally apply LoRA adaptation
from peft import LoraConfig, get_peft_model
use_lora = False
if intervention_type == "Lora":
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["self_attn.q_proj", "self_attn.v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    use_lora = True
    model = get_peft_model(model, peft_config)
    reft_config = pyreft.ReftConfig(representations=[])

# Initialize data collator for seq2seq
data_collator_fn = DataCollatorForSeq2Seq(
    model=model,
    tokenizer=tokenizer,
    padding="longest",
)
data_collator = ReftAudioDataCollator(data_collator=data_collator_fn)
model = pyreft.get_reft_model(model, reft_config)
if use_lora:
    model.model.enable_adapter_layers()

model.print_trainable_parameters()
model.model.train()

# Metric computation: WER using jiwer
from transformers import EvalPrediction
from jiwer import wer
import multiprocessing

def decode_batch(tokenizer, preds):
    return tokenizer.batch_decode(preds, skip_special_tokens=True)

def in_training_compute_metrics_factory(tokenizer):
    def in_training_compute_metrics(pred: EvalPrediction, is_regression=False):
        preds = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            pred_str = pool.apply_async(decode_batch, (tokenizer, preds))
            label_str = pool.apply_async(decode_batch, (tokenizer, pred.label_ids))
            pred_str = pred_str.get()
            label_str = label_str.get()
        wer_score = wer(label_str, pred_str)
        print({"wer": wer_score})
        return {"wer": wer_score}
    return in_training_compute_metrics

compute_metrics = in_training_compute_metrics_factory(tokenizer)

# Training arguments for Seq2Seq
from transformers import Seq2SeqTrainingArguments
training_args = Seq2SeqTrainingArguments(
    output_dir=f"./models/{args.model_size}/whisper-{args.model_size}-{args.intervention}-{args.reft_r}-fp32",
    overwrite_output_dir=True,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    learning_rate=1e-3,
    warmup_steps=50,
    num_train_epochs=3,
    evaluation_strategy="no",
    fp16=False,
    per_device_eval_batch_size=2,
    generation_max_length=128,
    logging_steps=10,
    remove_unused_columns=False,
    label_names=["labels"],
)

# Initialize the custom ReFT trainer
trainer = ReftSrTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

# Disable cache for training
model.config.use_cache = False

# Start training
trainer.train()
