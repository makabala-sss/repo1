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
from jiwer import Compose, ToLowerCase, RemovePunctuation, Strip, RemoveMultipleSpaces
from sacremoses import MosesPunctNormalizer
import sacrebleu

# Select CUDA device index
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# specify the pretrained model name
model_name_or_path = "openai/whisper-small"

language = "en"
language_abbr = "uk"
task = "translate"               # set the task type to translation
target_language = "english"      # add target language
dataset_name = "oovword/speech-translation-uk-en"  # switch dataset to CoVoST2

parser = argparse.ArgumentParser(description="whisper-small-Lora training parameters")
parser.add_argument(
    "--intervention",
    type=str,
    default="Loreft",
    help="Type of intervention to use: [Loreft,Direft,Lora]"
)
parser.add_argument(
    "--model_size",
    type=str,
    default="small",
    help="Model parameter size"
)
parser.add_argument(
    "--reft_r",
    type=int,
    default=4,
    help="Rank of the ReFT low-rank matrices"
)
parser.add_argument(
    "--lora_r",
    type=int,
    default=8,
    help="Rank of the LoRA low-rank matrices"
)
parser.add_argument(
    "--lora_alpha",
    type=int,
    default=16,
    help="LoRA scaling factor"
)
parser.add_argument(
    "--base_dir",
    type=str,
    default=None,
    help="Specify path; if none, search based on model size"
)
args = parser.parse_args()

# select model path based on size
if args.model_size == "small":
    model_name_or_path = "openai/whisper-small"
elif args.model_size == "medium":
    model_name_or_path = "openai/whisper-medium"
elif args.model_size == "large":
    model_name_or_path = "openai/whisper-large-v3"
else:
    print("Invalid choice: please select from small, medium, large")
    exit()

dataset = DatasetDict()

# initialize tokenizer, feature extractor, processor, and model
tokenizer = WhisperTokenizer.from_pretrained(
    model_name_or_path, language=language, task=task
)
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name_or_path)
processor = WhisperProcessor.from_pretrained(model_name_or_path)
model = WhisperForConditionalGeneration.from_pretrained(
    model_name_or_path, device_map=device, cache_dir="/data02/wwh/data/hf_cache/"
)

# set data path based on model size
if args.model_size == "small":
    path = "/home/WangYonghe_Grp/Wenhao_54/wwh/data/map_data_translate"
else:
    path = f"/home/WangYonghe_Grp/Wenhao_54/wwh/data/map_data_translate-{args.model_size}"

# load dataset from disk
dataset = dataset.load_from_disk(path)

share_weights = False
layers = "all"
position = 'f1+l1'
test_split = "validation"
model_input_name = feature_extractor.model_input_names[0]

# determine which layers to intervene on
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

# prepare evaluation dataset with interventions
eval_dataset = ReftAudioDatasetFixed(
    task="tr",
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

# choose intervention type
intervention_type = args.intervention
if intervention_type == "Loreft":
    intervention = pyreft.LoreftIntervention
elif intervention_type == "Direft":
    intervention = pyreft.DireftIntervention
else:
    intervention = pyreft.DireftIntervention

# configure ReFT with low-rank interventions
reft_config = pyreft.ReftConfig(
    representations=[
        {
            "layer": l,
            "component": (
                f"base_model.encoder.layers[{l}].self_attn.out_proj.output"
                if include_peft else "block_output"
            ),
            "low_rank_dimension": args.reft_r,
            "intervention": intervention(
                embed_dim=model.config.hidden_size,
                low_rank_dimension=args.reft_r,
                dtype=torch.float
            )
        }
        for l in layers_to_transform
    ]
)

# optionally wrap model with LoRA
if intervention_type == "Lora":
    peft_config = LoraConfig(
        r=args.lora_r,           # LoRA rank
        lora_alpha=args.lora_alpha,  # LoRA scaling factor
        target_modules=[
            "self_attn.q_proj",
            "self_attn.v_proj",
        ],
        lora_dropout=0.05,       # LoRA dropout probability
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    reft_config = pyreft.ReftConfig(representations=[])

# determine base directory for checkpoints
if args.base_dir is None:
    base_dir = f"./{args.model_size}/whisper-{args.model_size}-{args.intervention}-fp32"
else:
    base_dir = args.base_dir

# use MosesPunctNormalizer for punctuation normalization
punct_norm = MosesPunctNormalizer(lang='en')

transformation = jiwer.Compose([
    # Step 1: Moses punctuation normalization
    lambda x: punct_norm.normalize(x),
    # Step 2: special character replacement
    jiwer.SubstituteRegexes({
        r"‘": "'", r"’": "'",
        r"“": '"', r"”": '"',
        r"–": "-", r"—": "-",
        r"…": "...",
    }),
    # Step 3: basic cleaning
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    # Step 4: optional - handle currency symbols etc. (add as needed)
    # jiwer.SubstituteRegexes({r"€": "EUR", r"£": "GBP"}),
])

def evaluate_checkpoint(intervention_path):
    """
    Load the model from the given intervention_path, perform inference, and compute BLEU.
    """
    reft_model = pyreft.get_reft_model(model, reft_config)
    reft_model.print_trainable_parameters()

    # 2) load intervention model
    reft_model.load_intervention(
        intervention_path,
        include_model=True
    )
    for k, v in reft_model.interventions.items():
        _ = v[0].to(device).eval()
    reft_model = reft_model.to(device)

    batch_size = 4
    # 3) prepare data loader
    dataloader = pyreft.reft_trainer.make_dataloader(
        eval_dataset,
        batch_size,
        data_collator,
        shuffle=False
    )

    count = 0             # count of skipped empty-text samples
    all_predictions = []  # collect all predicted texts
    all_references = []   # collect all reference texts

    # 4) run inference and compute BLEU
    with torch.no_grad():
        for step, inputs in tqdm(enumerate(dataloader), desc=f"Evaluating {intervention_path}"):
            # move data to device
            for k, v in inputs.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)

            # handle intervention positions
            intervention_locations = inputs["intervention_locations"].permute(1, 0, 2).tolist()

            # forward pass to get outputs
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
            batch_size = labels.size(0)

            # decode each sample
            for i in range(batch_size):
                seq_len = pred_ids.size(1)
                pred_seq = pred_ids[i, :seq_len].cpu().tolist()
                label_seq = labels[i, :seq_len].cpu().tolist()

                # filter out -100
                label_seq = [token for token in label_seq if token != -100]

                # decode texts
                pred_text = tokenizer.decode(pred_seq, skip_special_tokens=True)
                ref_text = tokenizer.decode(label_seq, skip_special_tokens=True)

                # apply text normalization
                pred_text = transformation(pred_text)
                ref_text = transformation(ref_text)
                if not ref_text or not pred_text:
                    count += 1
                    continue

                all_predictions.append(pred_text)
                all_references.append(ref_text)

    # 5) compute BLEU
    if len(all_references) == 0:
        bleu_score = 0.0
    else:
        # compute BLEU using sacrebleu (handles tokenization automatically)
        bleu = sacrebleu.corpus_bleu(all_predictions, [all_references],
                                     tokenize='13a', smooth_method='exp',
                                     lowercase=False)
        bleu_score = bleu.score

    return bleu_score, count

def main():
    # store results {step: BLEU}
    results = {}
    print("Post-normalization results:")
    # get all checkpoint-xxx folders and sort in reverse order
    for dirname in sorted(
        os.listdir(base_dir),
        key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else -1,
        reverse=True
    ):
        match = re.match(r"checkpoint-(\d+)", dirname)
        if match:
            step = int(match.group(1))
            checkpoint_path = os.path.join(base_dir, dirname, "intervenable_model")
            if os.path.isdir(checkpoint_path):
                print(f"Evaluating checkpoint {step} ...")
                bleu_score, skipped = evaluate_checkpoint(checkpoint_path)
                results[step] = bleu_score
                print(f"Checkpoint {step} => BLEU: {bleu_score:.2f}, Skipped: {skipped}")

    print("\n=== Summary of BLEU for all checkpoints ===")
    for step, bleu_val in sorted(results.items()):
        print(f"{step}: {bleu_val:.2f}")

if __name__ == "__main__":
    main()
