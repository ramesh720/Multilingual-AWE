import sys
import os
import copy
import torch
import argparse
from timeit import default_timer as timer
from torch.optim.lr_scheduler import OneCycleLR

from conf.conf import Config
from dataloader.dataloader import create_data_loader
from dataloader.dataset import TripletSpeechDataset
from model.NAW_models import NAW_LSTM_multi_view_model
from utils.steps import train
from utils.transforms import mel_spec_transform
from utils.utils import ensure_dir_exists

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Train Neural Acoustic Word Embedding Model.')
parser.add_argument('--device',        type=str, default='cuda',         help='cuda or cpu')
parser.add_argument('--ckpt_dir_name', type=str, default='overall_run_1', help='Checkpoint dir name')
parser.add_argument('--log_dir_name',  type=str, default='overall_run_1', help='Tensorboard log dir name')
args = parser.parse_args()

device = torch.device(
    args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
)
print(f"🚀 Using device: {device}")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
print("🔄 Loading configuration...")
conf = Config()
conf.data_info_dict['test']['alignments_path'] = (
    '/home/siplabiith/clap_dwd_overall/data/test/test_overall_iv_combined.pkl'
)
print("✅ Configuration loaded. Validation → test_overall_iv_combined.pkl")

# ---------------------------------------------------------------------------
# Feature extractor
# ---------------------------------------------------------------------------
audio_transforms = torch.nn.Sequential(mel_spec_transform(config=conf))

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
print("\n📦 Loading datasets...")
NAWTrainData = TripletSpeechDataset(conf, transform=audio_transforms, type="train")
NAWValData   = TripletSpeechDataset(conf, transform=audio_transforms, type="test")

print("\n=================== 📊 SPLIT INTEGRITY CHECK ===================")
print(f"  Train instances : {len(NAWTrainData)}")
print(f"  Val   instances : {len(NAWValData)}")
print(f"  Train vocab     : {len(NAWTrainData.words)} words")
print(f"  Val   vocab     : {len(NAWValData.words)} words")
# Check coverage of min examples
min_required = conf.number_examples_wer_word
train_enough = sum(1 for w in NAWTrainData.words if len(NAWTrainData.word_dict[w]) >= min_required)
val_enough   = sum(1 for w in NAWValData.words   if len(NAWValData.word_dict[w])   >= min_required)
print(f"  Train words with ≥{min_required} instances: {train_enough}/{len(NAWTrainData.words)}")
print(f"  Val   words with ≥{min_required} instances: {val_enough}/{len(NAWValData.words)}")
print("================================================================\n")

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
print("⚙️  Creating data loaders...")
train_loader = create_data_loader(
    dataset=NAWTrainData, batch_size=conf.batch_size,
    shuffle=True, num_workers=4, verification=False, drop_last=True
)
val_loader = create_data_loader(
    dataset=NAWValData, batch_size=conf.batch_size,
    shuffle=False, num_workers=4, verification=False, drop_last=False
)
print("✅ Data loaders created.\n")

# ---------------------------------------------------------------------------
# Inspect one batch (to verify positives)
# ---------------------------------------------------------------------------
def inspect_batch(dataloader, device):
    batch = next(iter(dataloader))
    if batch[0].numel() == 0:
        print("❗ Empty batch!")
        return
    mels, anc, mel_lens, word_labels, anc_lens = batch
    print("\n🔍 SAMPLE BATCH INSPECTION")
    print(f"  Batch size (embeddings): {mels.size(0)}")
    print(f"  Number of unique words: {len(torch.unique(word_labels))}")
    print(f"  Word label distribution: {torch.bincount(word_labels)}")
    for i in range(min(5, len(word_labels))):
        print(f"    sample {i}: word_label={word_labels[i].item()}, mel_len={mel_lens[i]}, anc_len={anc_lens[i]}")
    print()

inspect_batch(train_loader, device)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
print("🧠 Initialising model...")
model = NAW_LSTM_multi_view_model(
    input_dim_acoustic=conf.n_mels,
    input_dim_text=conf.input_dim_text,
    no_of_tokens=conf.no_of_tokens,
    hidden_dim=conf.hidden_dim,
    embedding_dim=conf.embedding_dim,
    num_layers=conf.num_layers,
    bidirectional=True,
).to(device)
print("✅ Model initialised.\n")

# ---------------------------------------------------------------------------
# Optimiser & scheduler
# ---------------------------------------------------------------------------
print("📉 Setting up optimiser and scheduler...")
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
NUM_EPOCHS = conf.num_epochs

scheduler = OneCycleLR(
    optimizer,
    max_lr=1e-3,
    steps_per_epoch=len(train_loader),
    epochs=NUM_EPOCHS,
    pct_start=0.15,
    anneal_strategy='cos',
    div_factor=10.0,
    final_div_factor=100.0,
)
print("✅ Optimiser ready.\n")

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
cwd = os.getcwd()
CHECKPOINT_DIR = f"exp/ckpt/overall_fixed_logit_1{args.ckpt_dir_name}/"
LOG_DIR        = f"exp/logs/overall_fixed_logit_1/{args.log_dir_name}/"
ensure_dir_exists(os.path.join(cwd, CHECKPOINT_DIR))
ensure_dir_exists(os.path.join(cwd, LOG_DIR))

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
print("🏁 Starting training...\n")
start_time = timer()

train(
    model=model,
    train_dataloader=train_loader,
    validation_dataloader=val_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    epochs=NUM_EPOCHS,
    checkpoint_dir=CHECKPOINT_DIR,
    log_dir=LOG_DIR,
    device=device,
    config=conf,
)

# ---------------------------------------------------------------------------
# Save final checkpoint
# ---------------------------------------------------------------------------
model_save_path = os.path.join(CHECKPOINT_DIR, 'model_last.pt')
torch.save(copy.deepcopy(model).state_dict(), model_save_path)
print(f"\n💾 Final checkpoint saved → {model_save_path}")

end_time = timer()
print(f"🎉 Training complete. Duration: {end_time - start_time:.2f}s")