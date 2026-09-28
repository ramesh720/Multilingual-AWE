"""
Acoustic-View (Audio vs Audio) Pairwise Embedding Similarity Evaluation
-- Per-language trial sets under ind_lang_trials/output/<lang>/ --

Compares audio1 against audio2 directly (both from model.lstm_acoustic
via forward_infer's audio_emb output). The text/phoneme branch is still
computed internally by forward_infer but its output is discarded here --
this measures whether the model's ACOUSTIC embedding space alone
clusters same-word audio together, independent of the text branch.

Directory changes from the combined overall_ version:
  - TRIAL_CSV / FEATURE_DIR now point at
    ind_lang_trials/output/<lang>/trials_<split>_final.csv and
    .../features_<split>/, parameterized by --lang.
  - Words in these trial CSVs are PLAIN (untagged, e.g. "शामिल"), not
    "lang__word" -- split_lang_word() handles this correctly already.
  - PHONES_FILE / LEXICON_FILE still point at the single shared
    "overall" files (needed only because forward_infer requires SOME
    phoneme sequence per call, even though this script never uses the
    resulting text embedding).
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import average_precision_score, roc_curve

sys.path.append(
    "/home/siplabiith/unified_python_parser/unified_python_parser/Unified_Parser_smt_lab_IITM"
)
from uparser import wordparse

from conf.conf import Config
from model.NAW_models import NAW_LSTM_multi_view_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

conf = Config()

IND_LANG_TRIALS_ROOT = "/home/siplabiith/ind_lang_trials_v2/output"


def split_lang_word(tagged_word):
    """'telugu__పదం' -> ('telugu', 'పదం'). Falls back to (None, word)
    if there's no '__' separator -- the case for every word in these
    per-language trial sets."""
    if "__" in tagged_word:
        lang, raw_word = tagged_word.split("__", 1)
        return lang, raw_word
    return None, tagged_word


def main():

    parser = argparse.ArgumentParser(
        description="Acoustic-View (Audio vs Audio) AWE Matching -- per-language trials"
    )

    parser.add_argument("--lang", required=True)
    parser.add_argument("--split", required=True, choices=["iv", "oov"])

    parser.add_argument(
        "--model_path",
        default="/home/siplabiith/clap_dwd_overall/exp/ckpt/overall_fixed_logit_1overall_run_1/NAW_best_epoch9.pt"
    )

    args = parser.parse_args()

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running on {DEVICE}")

    LANG_DIR = os.path.join(IND_LANG_TRIALS_ROOT, args.lang)

    TRIAL_CSV = os.path.join(LANG_DIR, f"trials_{args.split}_final.csv")
    FEATURE_DIR = os.path.join(LANG_DIR, f"features_{args.split}")

    OUTPUT_CSV = f"ind_lang_acoustic_similarity_{args.lang}_{args.split}.csv"

    PHONES_FILE = "/home/siplabiith/clap_dwd_overall/data/lang/overall/phones.txt"
    LEXICON_FILE = "/home/siplabiith/clap_dwd_overall/data/lang/overall/lexicon.txt"

    if not os.path.exists(TRIAL_CSV):
        logger.error(f"Cannot find {TRIAL_CSV}")
        sys.exit(1)

    ###############################################################
    # phones (single shared file covering all languages)
    ###############################################################

    phones_dict = {}
    with open(PHONES_FILE, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                phones_dict[parts[0]] = int(parts[1])

    ###############################################################
    # lexicon (single shared file, keyed by RAW word)
    ###############################################################

    lexicon = {}
    with open(LEXICON_FILE, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                lexicon[parts[0]] = parts[1:]

    ###############################################################
    # model
    ###############################################################

    model = NAW_LSTM_multi_view_model(
        input_dim_acoustic=conf.n_mels,
        input_dim_text=conf.input_dim_text,
        no_of_tokens=conf.no_of_tokens,
        hidden_dim=conf.hidden_dim,
        embedding_dim=conf.embedding_dim,
        num_layers=conf.num_layers,
        bidirectional=True,
    ).to(DEVICE)

    checkpoint = torch.load(args.model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.eval()

    ###############################################################
    # embedding cache
    ###############################################################

    embedding_cache = {}

    CHUNK_SIZE = 100000
    BATCH_SIZE = 2048

    ###############################################################
    # embedding extraction
    ###############################################################

    @torch.no_grad()
    def get_embedding(file_name, word):

        if file_name in embedding_cache:
            return embedding_cache[file_name]

        _lang, raw_word = split_lang_word(str(word))

        if raw_word in lexicon:
            phonemes = lexicon[raw_word]
        else:
            try:
                phonemes = wordparse(raw_word.strip(), 0, 0, 1).split()
            except Exception:
                phonemes = []

        if len(phonemes) == 0:
            phonemes = ["SIL"]

        phoneme_indices = [phones_dict.get(ph, 1) for ph in phonemes]

        anc_seq = torch.tensor(
            phoneme_indices, dtype=torch.int64, device=DEVICE,
        ).unsqueeze(0)

        anc_length = torch.tensor([len(phoneme_indices)], dtype=torch.int64)

        feature_path = os.path.join(FEATURE_DIR, file_name)
        if not os.path.exists(feature_path):
            return None, None

        mel = torch.load(feature_path, map_location=DEVICE).float()
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)

        mel_length = torch.tensor([mel.shape[1]], dtype=torch.int64)

        audio_emb, text_emb = model.forward_infer(
            mel, anc_seq, mel_length, anc_length,
        )

        result = (audio_emb.squeeze(0), text_emb.squeeze(0))
        embedding_cache[file_name] = result
        return result

    ###############################################################
    # batch similarity -- AUDIO vs AUDIO
    ###############################################################

    def compute_similarity_batch(file1_list, file2_list, words1, words2):

        emb1_list, emb2_list, valid_indices = [], [], []

        for idx, (f1, f2, w1, w2) in enumerate(
            zip(file1_list, file2_list, words1, words2)
        ):
            audio1, _ = get_embedding(f1, w1)
            audio2, _ = get_embedding(f2, w2)

            if audio1 is None or audio2 is None:
                continue

            emb1_list.append(audio1)
            emb2_list.append(audio2)
            valid_indices.append(idx)

        if len(emb1_list) == 0:
            return [], []

        emb1 = torch.stack(emb1_list)
        emb2 = torch.stack(emb2_list)

        emb1 = torch.nn.functional.normalize(emb1, dim=-1)
        emb2 = torch.nn.functional.normalize(emb2, dim=-1)

        similarity = (emb1 * emb2).sum(dim=1)

        return similarity.cpu().tolist(), valid_indices

    ###############################################################
    # read trial file
    ###############################################################

    reader = pd.read_csv(TRIAL_CSV, chunksize=CHUNK_SIZE)
    first_write = True

    for chunk_id, chunk in enumerate(tqdm(reader, desc=f"{args.lang}/{args.split} (acoustic)")):

        results = []

        for start in range(0, len(chunk), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(chunk))
            batch = chunk.iloc[start:end]

            sims, valid = compute_similarity_batch(
                batch["file1"], batch["file2"],
                batch["word1"], batch["word2"],
            )

            for sim, idx in zip(sims, valid):
                results.append({
                    "word1": batch.iloc[idx]["word1"],
                    "word2": batch.iloc[idx]["word2"],
                    "file1": batch.iloc[idx]["file1"],
                    "file2": batch.iloc[idx]["file2"],
                    "gt_decision": batch.iloc[idx]["gt_decision"],
                    "similarity": round(sim, 4),
                })

        mode = "w" if first_write else "a"
        pd.DataFrame(results).to_csv(OUTPUT_CSV, mode=mode, header=first_write, index=False)
        first_write = False

    ###############################################################
    # Evaluation
    ###############################################################

    logger.info("Finished extracting similarities.")

    df = pd.read_csv(OUTPUT_CSV)

    y_true = df["gt_decision"].astype(int).values
    y_score = df["similarity"].astype(float).values

    ap = average_precision_score(y_true, y_score)

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr
    eer_index = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_index] + fnr[eer_index]) / 2.0

    logger.info("")
    logger.info("=" * 80)
    logger.info(f"Language           : {args.lang}")
    logger.info(f"Split              : {args.split.upper()}")
    logger.info(f"View               : ACOUSTIC (audio vs audio)")
    logger.info(f"Average Precision  : {ap:.8f}")
    logger.info(f"Equal Error Rate   : {eer:.8f}")
    logger.info("=" * 80)
    logger.info("")

    print(f"RESULT,acoustic,{args.lang},{args.split},{ap:.8f},{eer:.8f}")


if __name__ == "__main__":
    main()