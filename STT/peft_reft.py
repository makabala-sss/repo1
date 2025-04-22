import os
from transformers import AutoConfig, DataCollatorForSeq2Seq, WhisperForConditionalGeneration
from pyreft import ReftSrTrainer
import pyreft
import torch
import numpy as np
from pyreft import ReftAudioDatasetFixed, ReftAudioDataCollator
from datasets import load_from_disk, load_dataset, DatasetDict, Audio
import argparse
import torch
import numpy as np
from transformers import EvalPrediction
from jiwer import wer
import multiprocessing
# Select the CUDA device index
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Task and dataset configuration
language = "french"
language_abbr = "uk"
task = "translate"
target_language = "english"
dataset_name = "oovword/speech-translation-uk-en"

# Argument parser for training parameters
description = "whisper-small-LoRa training parameters"
parser = argparse.ArgumentParser(description=description)
parser.add_argument(
    "--intervention",
    type=str,
    default="Loreft",
    help="Type of intervention to use: [Loreft, Direft, Lora]"
)
parser.add_argument(
    "--model_size",
    type=str,
    default="small",
    help="Model size: [small, medium, large]"
)
parser.add_argument(
    "--layers",
    type=int,
    default=3,
    help="Number of layers to intervene"
)
parser.add_argument(
    "--reft_r",
    type=int,
    default=4,
    help="Low-rank dimension for Reft"
)
parser.add_argument(
    "--lora_r",
    type=int,
    default=8,
    help="Rank for LoRA low-rank matrices"
)
parser.add_argument(
    "--lora_alpha",
    type=int,
    default=16,
    help="Scaling factor for LoRA"
)
args = parser.parse_args()

# Select pretrained model based on model_size argument
if args.model_size == "small":
    model_name_or_path = "openai/whisper-small"
elif args.model_size == "medium":
    model_name_or_path = "openai/whisper-medium"
elif args.model_size == "large":
    model_name_or_path = "openai/whisper-large-v3"
else:
    print("Invalid model size, please choose from [small, medium, large]")
    exit()

# Determine which layers to transform based on user input
if args.layers == 3:
    layers_to_transform = [5, 7, 10]
elif args.layers == 10:
    layers_to_transform = list(range(12))  # all 12 encoder layers
else:
    print("Only 3 or 10 layers supported for intervention")
    exit()

# Load CoVoST2 French-to-English translation dataset
# and cast the audio column to the correct sampling rate
dataset = DatasetDict()
from transformers import WhisperFeatureExtractor, WhisperTokenizer,WhisperProcessor
# Feature extractor initialization
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name_or_path)

# Initialize tokenizer for translation task
tokenizer = WhisperTokenizer.from_pretrained(
    model_name_or_path,
    language=target_language,
    task=task
)

# Data preprocessing function
def prepare_dataset(batch):
    # Resample and extract input features for audio
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    
    # Tokenize translation text and use as both input_ids and labels
    batch_input_ids = tokenizer(
        batch["en"],  # translation text
        padding="max_length",
        truncation=True,
        max_length=448
    ).input_ids
    batch["input_ids"] = batch_input_ids
    batch["labels"] = batch_input_ids
    return batch

# Save processed data to disk or load if exists
data_path = "your/path/here"
if not os.path.exists(data_path):
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

print(dataset["train"])

# Initialize processor for translation task
processor = WhisperProcessor.from_pretrained(
    model_name_or_path,
    language=target_language,
    task=task
)


from transformers import WhisperForConditionalGeneration
share_weights=False
layers = "all"
position = 'f1+l1'
test_split = "validation"

model_input_name = feature_extractor.model_input_names[0]
# position str takes the following formats:
# f1 -> first token; f2 -> first two tokens.
# f1+l1 -> first and last tokens; f2+l2 -> first and last two tokens.
# fn or ln shares the same intervention.
if layers.strip() == "":
    layers = []
elif layers != "all":
    layers = [int(l) for l in layers.split(";")]
else:
    temp_config = AutoConfig.from_pretrained(model_name_or_path)
    print(temp_config)
    layers = [l for l in range(temp_config.num_hidden_layers)]
    print(layers)
    
train_dataset = ReftAudioDatasetFixed(
    task="sr",
    data_path=dataset_name,
    dataset=dataset["train"],
    feature_extractor=feature_extractor,  # 传递特征提取器
    language_abbr=language_abbr,
    tokenizer=tokenizer,
    # max_n_example=6000,
    data_split="train",
    seed=42,
    #参数： 设置为“al” -》干预数量就是所有隐藏层的数量
    **{"num_interventions": len(layers), "position": position,
       "share_weights": share_weights, "test_split": test_split,
       }
)

eval_dataset = ReftAudioDatasetFixed(
    task="sr",
    data_path=dataset_name,
    dataset=dataset["test"],
    feature_extractor=feature_extractor,  # 传递特征提取器
    data_split="test",
    tokenizer=tokenizer,
    seed=42,
    **{"num_interventions": len(layers), "position": position,
       "share_weights": share_weights, "test_split": test_split}
)

# Load or reload the pretrained model with device mapping
model = WhisperForConditionalGeneration.from_pretrained(
    model_name_or_path,
    device_map="auto",
)
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

# Setup intervention configuration for ReFT
intervention_type = args.intervention
if intervention_type == "Loreft":
    intervention = pyreft.LoreftIntervention
elif intervention_type == "Direft":
    intervention = pyreft.DireftIntervention
elif intervention_type == "AdaDoreft":
    intervention = pyreft.AdareftIntervention
else:
    intervention = pyreft.DireftIntervention

reft_config = pyreft.ReftConfig(
    representations=[{
        "layer": l,
        "component": (
            f"base_model.encoder.layers[{l}].self_attn.out_proj.output"
            if intervention_type != "Lora"
            else "block_output"
        ),
        "low_rank_dimension": args.reft_r,
        "intervention": intervention(
            embed_dim=model.config.hidden_size,
            low_rank_dimension=args.reft_r,
            dtype=torch.float
        )
    } for l in layers_to_transform]
)

# If using LoRA, wrap the model accordingly
use_lora = False
if args.intervention == "Lora":
    from peft import LoraConfig, get_peft_model
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["self_attn.q_proj", "self_attn.v_proj"],
        lora_dropout=0.05,
        bias="none"
    )
    model = get_peft_model(model, peft_config)
    use_lora = True
    # Clear ReFT representations when using LoRA
    reft_config = pyreft.ReftConfig(representations=[])

# Prepare data collator for Seq2Seq training with ReFT
seq2seq_collator = DataCollatorForSeq2Seq(
    model=model,
    tokenizer=tokenizer,
    padding="longest"
)

data_collator = ReftAudioDataCollator(data_collator=seq2seq_collator)
model = pyreft.get_reft_model(model, reft_config)

if use_lora:
    # Re-enable LoRA adapter layers gradients
    model.model.enable_adapter_layers()

# Print number of trainable parameters
model.print_trainable_parameters()
model.model.train()

# Multiprocessing decode helper function
def decode_batch(tokenizer, preds):
    # Decode a batch of token IDs to strings
    return tokenizer.batch_decode(preds, skip_special_tokens=True)

# Create compute_metrics function for training
def in_training_compute_metrics_factory(tokenizer):
    def in_training_compute_metrics(pred, is_regression=False):
        preds = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)

        # Parallel decode predictions and labels
        with multiprocessing.Pool() as pool:
            pred_str = pool.apply_async(decode_batch, (tokenizer, preds)).get()
            label_str = pool.apply_async(decode_batch, (tokenizer, pred.label_ids)).get()

        wer_score = wer(label_str, pred_str)
        print({"wer": wer_score})
        return {"wer": wer_score}
    return in_training_compute_metrics

compute_metrics = in_training_compute_metrics_factory(tokenizer)

# Training arguments configuration
from transformers import Seq2SeqTrainingArguments
training_args = Seq2SeqTrainingArguments(
    output_dir=f"./{args.model_size}/whisper-{args.model_size}-{args.intervention}-{args.reft_r}-fp32",
    overwrite_output_dir=True,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    learning_rate=1e-3,
    warmup_steps=50,
    num_train_epochs=3,
    evaluation_strategy="no",
    fp16=False,
    per_device_eval_batch_size=8,
    generation_max_length=128,
    logging_steps=10,
    remove_unused_columns=False,
    label_names=["labels"],
)

# Initialize ReftSrTrainer and start training
trainer = ReftSrTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)
model.config.use_cache = False  # Disable caching during training to avoid warnings
trainer.train()
