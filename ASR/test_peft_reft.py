from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperFeatureExtractor,
    WhisperTokenizer,
    AutoConfig,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model
import torch
import pyreft
from datasets import load_dataset, DatasetDict, Audio
import datasets
import os
from pyreft import ReftAudioDatasetFixed, ReftAudioDataCollator
import re
import jiwer
from tqdm import tqdm
import argparse

# Select CUDA device index
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
model_name_or_path = "openai/whisper-large-v3"

language = "french"
language_abbr = "fr"
task = "transcribe"
dataset_name = "facebook/voxpopuli"

dataset = DatasetDict()
# .select(range(10))
parser = argparse.ArgumentParser(description="Reft_model_test_WER")
parser.add_argument(
    "--base_dir",
    type=str,
    default=None,
    help="Path to store model checkpoints"
)
parser.add_argument(
    "--intervention",
    type=str,
    default="Loreft",
    help="Type of intervention to use: [Loreft,Direft,Lora]"
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="If specified, load the test split from HuggingFace online, not from local disk"
)
args = parser.parse_args()

tokenizer = WhisperTokenizer.from_pretrained(
    model_name_or_path, language=language, task=task
)
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name_or_path)
processor = WhisperProcessor.from_pretrained(
    model_name_or_path, language=language, task=task
)
model = WhisperForConditionalGeneration.from_pretrained(
    model_name_or_path, device_map="auto",
)

path = "your_path"  # replace with your actual path

# online load: directly download the test split from HF hub
if args.checkpoint:
    dataset = load_dataset(dataset_name, language_abbr, split="test")
else:
    # default behavior: load from local disk
    dataset = DatasetDict()
    dataset = dataset.load_from_disk(path)

share_weights = False
layers = "all"
position = 'f1+l1'
test_split = "validation"
model_input_name = feature_extractor.model_input_names[0]

# Determine layers list
if layers.strip() == "":
    layers = []
elif layers != "all":
    layers = [int(l) for l in layers.split(";")]
else:
    temp_config = AutoConfig.from_pretrained(model_name_or_path)
    print(temp_config)
    layers = list(range(temp_config.num_hidden_layers))
    print(layers)

print(dataset["test"])

eval_dataset = ReftAudioDatasetFixed(
    task="sr",
    data_path=dataset_name,
    dataset=dataset["test"],
    feature_extractor=feature_extractor,  # pass feature extractor
    data_split="test",
    tokenizer=tokenizer,
    seed=42,
    **{
        "num_interventions": len(layers),
        "position": position,
        "share_weights": share_weights,
        "test_split": test_split
    }
)

data_collator_fn = DataCollatorForSeq2Seq(
    model=model,
    tokenizer=tokenizer,
    padding="longest",
)

data_collator = ReftAudioDataCollator(data_collator=data_collator_fn)

include_peft = True
layers_to_transform = [10]

# Set up intervention
intervention_type = args.intervention
if intervention_type == "Loreft":
    intervention = pyreft.LoreftIntervention
elif intervention_type == "Direft":
    intervention = pyreft.DireftIntervention
else:
    intervention = pyreft.DireftIntervention

reft_config = pyreft.ReftConfig(
    representations=[
        {
            "layer": l,
            "component": (
                f"base_model.encoder.layers[{l}].self_attn.out_proj.output"
                if include_peft
                else "block_output"
            ),
            "low_rank_dimension": 4,
            "intervention": intervention(
                embed_dim=model.config.hidden_size,
                low_rank_dimension=4,
                dtype=torch.float
            )
        }
        for l in layers_to_transform
    ]
)

if intervention_type == "Lora":
    peft_config = LoraConfig(
        r=8,                   # LoRA rank
        lora_alpha=16,         # LoRA scaling factor
        target_modules=[
            "self_attn.k_proj",
            "self_attn.q_proj",
        ],
        lora_dropout=0.05,     # LoRA dropout probability
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    reft_config = pyreft.ReftConfig(representations=[])

base_dir = args.base_dir

from sacremoses import MosesPunctNormalizer

# use MosesPunctNormalizer for French punctuation normalization
punct_norm = MosesPunctNormalizer(lang='fr')

transformation = jiwer.Compose([
    lambda x: punct_norm.normalize(x),
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.Strip(),
    jiwer.RemoveMultipleSpaces()
])


def evaluate_checkpoint(intervention_path):
    """
    Load the model for the given intervention_path, perform inference and compute WER, return values.
    """
    reft_model = pyreft.get_reft_model(model, reft_config)
    reft_model.print_trainable_parameters()

    # 2) load intervention model
    reft_model.load_intervention(
        intervention_path,
        include_model=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for k, v in reft_model.interventions.items():
        _ = v[0].to(device).eval()
    reft_model = reft_model.to(device)

    # count skipped references
    count = 0
    batch_size = 4
    total_samples = 0
    current_wer = 0.0

    # 3) prepare data
    dataloader = pyreft.reft_trainer.make_dataloader(
        eval_dataset,
        batch_size,
        data_collator,
        shuffle=False
    )

    # 4) run inference and compute WER
    with torch.no_grad():
        for step, inputs in tqdm(enumerate(dataloader), desc=f"Evaluating {intervention_path}"):
            # move data to device
            for k, v in inputs.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)

            # handle intervention positions
            intervention_locations = inputs["intervention_locations"].permute(1, 0, 2).tolist()

            # forward pass to obtain outputs
            _, cf_outputs = reft_model(
                {
                    "input_features": inputs["input_features"],
                    "attention_mask": inputs["attention_mask"]
                },
                unit_locations={"sources->base": (None, intervention_locations)},
                labels=inputs["labels"],
                subspaces=(
                    inputs["subspaces"].permute(1, 0, 2).tolist()
                    if "subspaces" in inputs else None
                )
            )

            logits = cf_outputs.logits
            labels = inputs["labels"]

            # take argmax as predictions
            pred_ids = torch.argmax(logits, dim=-1)

            # compute per-sample
            local_wer_sum = 0.0
            valid_samples = 0

            for i in range(pred_ids.size(0)):
                seq_len = pred_ids.size(1)
                pred_seq = pred_ids[i, :seq_len].cpu().tolist()
                label_seq = labels[i, :seq_len].cpu().tolist()

                # filter out -100 values
                label_seq = [token for token in label_seq if token != -100]

                # decode
                pred_text = tokenizer.decode(pred_seq, skip_special_tokens=True)
                ref_text = tokenizer.decode(label_seq, skip_special_tokens=True)
                pred_text = transformation(pred_text)
                ref_text = transformation(ref_text)
                if not ref_text or not pred_text:
                    count += 1
                    continue

                # compute WER (jiwer.wer requires list inputs)
                local_wer_sum += jiwer.wer([ref_text], [pred_text])
                valid_samples += 1

            # accumulate at batch level
            if valid_samples > 0:
                current_wer += (local_wer_sum / valid_samples)
                total_samples += 1

    # 5) compute average WER (averaged per batch)
    final_wer = (current_wer / total_samples) if total_samples > 0 else 0.0

    return final_wer, count


def main():
    # for storing results {step_or_path: WER}
    results = {}
    print("Post-normalization results:")

    # if the user specified --checkpoint, evaluate only that one
    if args.checkpoint:
        checkpoint_path = args.checkpoint
        print(f"Evaluating single checkpoint: {checkpoint_path} ...")
        wer_value, skipped = evaluate_checkpoint(checkpoint_path)
        print(f"Checkpoint {os.path.basename(checkpoint_path)} => WER: {wer_value*100:.2f}%, Skipped: {skipped}")
        return  # exit immediately after processing

    # otherwise, iterate over all checkpoint-xxx folders under base_dir as per original logic
    for dirname in sorted(os.listdir(base_dir), key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else -1):
        match = re.match(r"checkpoint-(\d+)", dirname)
        if not match:
            continue
        step = int(match.group(1))
        checkpoint_path = os.path.join(base_dir, dirname, "intervenable_model")
        if not os.path.isdir(checkpoint_path):
            continue

        print(f"Evaluating checkpoint {step} ...")
        wer_value, skipped = evaluate_checkpoint(checkpoint_path)
        results[step] = wer_value
        print(f"Checkpoint {step} => WER: {wer_value*100:.2f}%, Skipped: {skipped}")

    # print summary of all results
    if results:
        print("\n=== Summary of WER for all checkpoints ===")
        for step, wer_val in sorted(results.items()):
            print(f"{step}: {wer_val*100:.2f}%")


if __name__ == "__main__":
    main()
