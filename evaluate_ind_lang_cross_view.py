"""
Cross-View (Audio vs Text) Evaluation
-- Per-language trial sets under ind_lang_trials/output/<lang>/ --

Compares audio_emb from file1 against text_emb derived from word2's
phoneme sequence -- this is a genuine test of whether the model's text
(phoneme) representation actually aligns with real audio, unlike the
acoustic-view script which only compares audio-to-audio.

Directory changes from the combined overall_ version:
  - trial_csv / feature_dir now point at
    ind_lang_trials/output/<lang>/trials_<split>_final.csv and
    .../features_<split>/, parameterized by --lang.
  - Words in these trial CSVs are PLAIN (untagged, e.g. "शामिल"), not
    "lang__word" -- split_lang_word() handles this correctly already.
  - PHONES_FILE / LEXICON_FILE still point at the single shared
    "overall" files covering all languages.
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

sys.path.append("/home/siplabiith/unified_python_parser/unified_python_parser/Unified_Parser_smt_lab_IITM")
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

PHONES_FILE = "/home/siplabiith/clap_dwd_overall/data/lang/overall/phones.txt"
LEXICON_FILE = "/home/siplabiith/clap_dwd_overall/data/lang/overall/lexicon.txt"
CHUNK_SIZE = 100000
BATCH_SIZE = 2048
SIMILARITY_THRESHOLD = 0.5


def split_lang_word(tagged_word):
    """'telugu__పదం' -> ('telugu', 'పదం'). Falls back to (None, word)
    if there's no '__' separator -- the case for every word in these
    per-language trial sets."""
    if "__" in tagged_word:
        lang, raw_word = tagged_word.split("__", 1)
        return lang, raw_word
    return None, tagged_word


def load_phones():
    phones_dict = {}
    with open(PHONES_FILE, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                phones_dict[parts[0]] = int(parts[1])
    return phones_dict


def load_lexicon():
    lexicon = {}
    with open(LEXICON_FILE, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                lexicon[parts[0]] = parts[1:]
    return lexicon


def load_model(model_path, device):
    model = NAW_LSTM_multi_view_model(
        input_dim_acoustic=conf.n_mels,
        input_dim_text=conf.input_dim_text,
        no_of_tokens=conf.no_of_tokens,
        hidden_dim=conf.hidden_dim,
        embedding_dim=conf.embedding_dim,
        num_layers=conf.num_layers,
        bidirectional=True,
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    return model


class CrossViewEvaluator:

    def __init__(self, model, feature_dir, phones_dict, lexicon, device):
        self.model = model
        self.feature_dir = feature_dir
        self.phones_dict = phones_dict
        self.lexicon = lexicon
        self.device = device
        self.embedding_cache = {}

    def get_embedding(self, file_name, word):
        """
        `word` here is PLAIN (untagged) for these per-language trial
        sets, but split_lang_word() is still applied for safety/
        consistency with the combined-model version of this script.
        """
        cache_key = f"{file_name}_{word}"
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]

        _lang, raw_word = split_lang_word(str(word))

        if raw_word in self.lexicon:
            phonemes = self.lexicon[raw_word]
        else:
            try:
                phonemes = wordparse(raw_word.strip(), 0, 0, 1).split()
            except Exception:
                phonemes = []

        if len(phonemes) == 0:
            phonemes = ["SIL"]

        phoneme_indices = [self.phones_dict.get(ph, 1) for ph in phonemes]
        anc_seq = torch.tensor(phoneme_indices, dtype=torch.int64, device=self.device).unsqueeze(0)
        anc_length = torch.tensor([len(phoneme_indices)], dtype=torch.int64)

        feature_path = os.path.join(self.feature_dir, file_name)
        if not os.path.exists(feature_path):
            return None, None

        mel = torch.load(feature_path, map_location=self.device).float()
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
        mel_length = torch.tensor([mel.shape[1]], dtype=torch.int64)

        with torch.no_grad():
            audio_emb, text_emb = self.model.forward_infer(
                mel, anc_seq, mel_length, anc_length
            )

        result = (audio_emb.squeeze(0), text_emb.squeeze(0))
        self.embedding_cache[cache_key] = result

        return result

    def compute_cross_similarity_batch(self, file_list, word_list):
        """
        Compare audio embedding from file with text embedding from word
        """
        audio_embeddings = []
        text_embeddings = []
        valid_indices = []

        for idx, (file_name, word) in enumerate(zip(file_list, word_list)):
            audio_emb, text_emb = self.get_embedding(file_name, word)

            if audio_emb is None or text_emb is None:
                continue

            audio_embeddings.append(audio_emb)
            text_embeddings.append(text_emb)
            valid_indices.append(idx)

        if len(audio_embeddings) == 0:
            return [], []

        audio_emb = torch.stack(audio_embeddings)
        text_emb = torch.stack(text_embeddings)

        audio_emb = torch.nn.functional.normalize(audio_emb, dim=-1)
        text_emb = torch.nn.functional.normalize(text_emb, dim=-1)

        similarity = (audio_emb * text_emb).sum(dim=1)

        return similarity.cpu().tolist(), valid_indices

    def evaluate(self, trial_csv, output_csv):
        logger.info(f"Loading trials from: {trial_csv}")

        reader = pd.read_csv(trial_csv, chunksize=CHUNK_SIZE)

        all_results = []
        first_write = True

        for chunk_id, chunk in enumerate(tqdm(reader, desc="Processing Trials")):
            chunk_results = []

            for start in range(0, len(chunk), BATCH_SIZE):
                end = min(start + BATCH_SIZE, len(chunk))
                batch = chunk.iloc[start:end]

                sims, valid_indices = self.compute_cross_similarity_batch(
                    batch["file1"].tolist(),
                    batch["word2"].tolist()  # Audio from file1, Text from word2
                )

                for sim, idx in zip(sims, valid_indices):
                    row = batch.iloc[idx]

                    prediction = 1 if sim > SIMILARITY_THRESHOLD else 0
                    is_correct = (prediction == row["gt_decision"])

                    chunk_results.append({
                        "audio_word": row["word1"],
                        "text_word": row["word2"],
                        "audio_file": row["file1"],
                        "gt_decision": row["gt_decision"],
                        "similarity": round(sim, 4),
                        "prediction": prediction,
                        "correct": is_correct
                    })

            all_results.extend(chunk_results)

            chunk_df = pd.DataFrame(chunk_results)
            mode = "w" if first_write else "a"
            chunk_df.to_csv(output_csv, mode=mode, header=first_write, index=False)
            first_write = False

            logger.info(f"Chunk {chunk_id + 1}: Processed {len(chunk_results)} trials")

        logger.info(f"Completed! Total trials: {len(all_results)}")

        ap, eer = self.calculate_metrics(pd.DataFrame(all_results))

        return all_results, ap, eer

    def calculate_metrics(self, df):
        y_true = df["gt_decision"].astype(int).values
        y_score = df["similarity"].astype(float).values
        y_pred = df["prediction"].astype(int).values

        ap = average_precision_score(y_true, y_score)

        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        fnr = 1.0 - tpr
        eer_index = np.nanargmin(np.abs(fpr - fnr))
        eer = (fpr[eer_index] + fnr[eer_index]) / 2.0

        accuracy = (y_pred == y_true).mean()

        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        logger.info("=" * 80)
        logger.info("CROSS-VIEW EVALUATION RESULTS (AUDIO vs TEXT)")
        logger.info("=" * 80)
        logger.info(f"Total Trials        : {len(df)}")
        logger.info(f"Accuracy            : {accuracy:.4f}")
        logger.info(f"Average Precision   : {ap:.8f}")
        logger.info(f"Equal Error Rate    : {eer:.8f}")
        logger.info(f"Precision           : {precision:.4f}")
        logger.info(f"Recall              : {recall:.4f}")
        logger.info(f"F1 Score            : {f1:.4f}")
        logger.info("=" * 80)

        logger.info("\nConfusion Matrix:")
        logger.info(f"  True Positives  : {tp}")
        logger.info(f"  True Negatives  : {tn}")
        logger.info(f"  False Positives : {fp}")
        logger.info(f"  False Negatives : {fn}")
        logger.info("=" * 80)

        return ap, eer


def main():
    parser = argparse.ArgumentParser(description="Cross-View Evaluation (Audio vs Text) -- per-language trials")
    parser.add_argument("--lang", required=True)
    parser.add_argument(
        "--split",
        required=True,
        choices=["iv", "oov"],
        help="Split to evaluate: iv (in-vocabulary) or oov (out-of-vocabulary)"
    )
    parser.add_argument(
        "--model_path",
        default="/home/siplabiith/clap_dwd_overall/exp/ckpt/overall_fixed_logit_1overall_run_1/NAW_best_epoch9.pt",
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Similarity threshold for predictions"
    )
    parser.add_argument(
        "--output_dir",
        default="./results",
        help="Directory to save results"
    )
    args = parser.parse_args()

    global SIMILARITY_THRESHOLD
    SIMILARITY_THRESHOLD = args.threshold

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    LANG_DIR = os.path.join(IND_LANG_TRIALS_ROOT, args.lang)

    trial_csv = os.path.join(LANG_DIR, f"trials_{args.split}_crossview.csv")
    feature_dir = os.path.join(LANG_DIR, f"features_{args.split}")
    output_csv = os.path.join(
        args.output_dir, f"ind_lang_cross_view_{args.lang}_{args.split}.csv"
    )

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Loading phoneme mapping...")
    phones_dict = load_phones()
    logger.info(f"Loaded {len(phones_dict)} phonemes")

    logger.info("Loading lexicon...")
    lexicon = load_lexicon()
    logger.info(f"Loaded {len(lexicon)} words")

    logger.info(f"Loading model from: {args.model_path}")
    model = load_model(args.model_path, device)
    logger.info("Model loaded successfully")

    evaluator = CrossViewEvaluator(
        model=model,
        feature_dir=feature_dir,
        phones_dict=phones_dict,
        lexicon=lexicon,
        device=device
    )

    logger.info(f"Starting cross-view evaluation for {args.lang}/{args.split.upper()}")
    results, ap, eer = evaluator.evaluate(trial_csv, output_csv)

    logger.info(f"Results saved to: {output_csv}")
    logger.info("Evaluation complete!")

    print(f"RESULT,cross_view,{args.lang},{args.split},{ap:.8f},{eer:.8f}")


if __name__ == "__main__":
    main()