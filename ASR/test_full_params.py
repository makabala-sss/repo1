from transformers import( WhisperForConditionalGeneration,
                         WhisperProcessor,WhisperFeatureExtractor,WhisperTokenizer,
                         AutoConfig,DataCollatorForSeq2Seq)
from peft import LoraConfig,get_peft_model
import torch
import pyreft
from datasets import load_dataset, DatasetDict ,Audio
import datasets
import os
from pyreft import ReftAudioDatasetFixed,ReftAudioDataCollator
from transformers import WhisperForConditionalGeneration, WhisperProcessor
# 指定预训练模型名称
import re
import torch
import jiwer
from tqdm import tqdm
import argparse
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import torch
import jiwer
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
from dataclasses import dataclass
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
model_name_or_path = "openai/whisper-large-v3"

language = "french"
language_abbr = "fr"
task = "transcribe"
dataset_name = "facebook/voxpopuli"



dataset = DatasetDict()
# .select(range(10))
parser = argparse.ArgumentParser(description="finetune_model_test_WER")
parser.add_argument("--model_size",type=str,default="small",help="模型规模")
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


tokenizer = WhisperTokenizer.from_pretrained(model_name_or_path, language=language, task=task)
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name_or_path)
processor = WhisperProcessor.from_pretrained(model_name_or_path)
model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path, device_map="auto",cache_dir="/data02/wwh/data/hf_cache/")
dataset = DatasetDict()

share_weights=False
layers = "all"
position = 'f1+l1'
test_split = "validation"
model_input_name = feature_extractor.model_input_names[0]

path = f"/home/WangYonghe_Grp/Wenhao_54/wwh/data/map_data_fr_default_{args.model_size}"  # 请替换为你的实际路径
    # 如果路径存在，则直接加载数据集
dataset = dataset.load_from_disk(path)

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
        
        label_features = [{
            "input_ids": feature["labels"],
        } for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt",padding=True)
        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        
        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)
eval_dataset = dataset["test"]

base_dir = f"./checkpoint_path/{args.model_size}"

from sacremoses import MosesPunctNormalizer
#use MosesPunctNormalizer 
punct_norm = MosesPunctNormalizer(lang='fr')

transformation = jiwer.Compose([
    lambda x: punct_norm.normalize(x),
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.Strip(),
    jiwer.RemoveMultipleSpaces()
])
def evaluate_checkpoint(model_path):
    """
    compute WER
    """
    # 1) Load the base model and processor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path).to(device)
    model.eval()

    # 2) Prepare data (assuming data_collator and eval_dataset are properly defined)
    batch_size = 4
    dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        shuffle=False
    )

    count = 0           # Count of skipped empty-text samples
    total_samples = 0   # Count of valid samples
    current_wer = 0.0   # Accumulated WER

    # 3) Run inference and compute WER
    with torch.no_grad():
        for step, inputs in tqdm(enumerate(dataloader), desc=f"Evaluating {model_path}"):
            # Move data to device
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)

            # Forward pass through the base model
            outputs = model(
                input_features=inputs["input_features"],
                labels=inputs["labels"]
            )
            
            logits = outputs.logits
            labels = inputs["labels"]

            # Take argmax as predictions
            pred_ids = torch.argmax(logits, dim=-1)

            batch_size = labels.size(0)
            local_wer_sum = 0.0
            valid_samples = 0

            # Compute per-sample WER
            for i in range(batch_size):
                # Retrieve sequence (removing padding)
                pred_seq = pred_ids[i].cpu().tolist()
                label_seq = labels[i].cpu().tolist()

                # Filter out -100 labels (tokens ignored by CTK loss)
                label_seq = [token for token in label_seq if token != -100]

                # Decode text
                pred_text = processor.decode(pred_seq, skip_special_tokens=True)
                ref_text = processor.decode(label_seq, skip_special_tokens=True)
                
                # Apply text normalization (assuming transformation is defined)
                pred_text = transformation(pred_text)
                ref_text = transformation(ref_text)

                # Skip empty reference or prediction
                if not ref_text or not pred_text:
                    count += 1
                    continue

                # Compute WER
                local_wer_sum += jiwer.wer([ref_text], [pred_text])
                valid_samples += 1

            # Accumulate metrics
            if valid_samples > 0:
                current_wer += local_wer_sum
                total_samples += valid_samples

    # 4) Compute final WER
    final_wer = current_wer / total_samples if total_samples > 0 else 0.0
    return final_wer, count

def main():
    # Store results as {step: WER}
    results = {}
    print("规范化后")
    # Get all checkpoint-xxx folders sorted by numeric step
    for dirname in sorted(os.listdir(base_dir), key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else -1):
        match = re.match(r"checkpoint-(\d+)", dirname)
        if match:
            step = int(match.group(1))
            checkpoint_path = os.path.join(base_dir, dirname)
            if os.path.isdir(checkpoint_path):
                print(f"Evaluating checkpoint {step} ...")
                wer_value, skipped = evaluate_checkpoint(checkpoint_path)
                results[step] = wer_value
                print(f"Checkpoint {step} => WER: {wer_value*100:.2f}%, Skipped: {skipped}")
    
    print("\n=== Summary of WER for all checkpoints ===")
    for step, wer_val in sorted(results.items()):
        print(f"{step}: {wer_val*100:.2f}%")

if __name__ == "__main__":
    main()
