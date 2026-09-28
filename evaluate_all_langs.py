"""
Fast word-discrimination evaluation over all languages, splits and views.

Same trials, scoring and metrics as evaluate_ind_lang_acoustic.py and
evaluate_ind_lang_cross_view.py, but each unique audio segment and each unique
text word is encoded ONCE, in length-sorted batches, and every trial is then
scored with a dot product of cached embeddings.

  acoustic view : trials_<split>_final.csv      cos(audio[file1], audio[file2])
  cross view    : trials_<split>_crossview.csv  cos(audio[file1], text[word2])

Scores are rounded to 4 decimals before AP/EER, as the old scripts did (they
wrote round(sim, 4) to CSV and computed the metrics from it); --round -1 turns
that off. Rows whose feature file is missing are dropped (and counted), also
as before.

The model size (hidden dim, layers, ...) is read from the checkpoint, so old
256x3 and new 512x6 checkpoints both work regardless of conf.py.

Usage:
    python evaluate_all_langs.py --model_path exp/ckpt/.../NAW_best_epoch11.pt --device cuda:1
    python evaluate_all_langs.py --model_path ... --langs telugu tamil --splits iv --views cross
"""
import os
import time
import argparse
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import average_precision_score, roc_curve

from model.NAW_models import NAW_LSTM_multi_view_model

TRIALS_ROOT  = "/home/siplabiith/ind_lang_trials_v2/output"
PHONES_FILE  = "data/lang/overall/phones.txt"
LEXICON_FILE = "data/lang/overall/lexicon.txt"
LANGUAGES = ["bengali", "gujarati", "hindi", "kannada", "malayalam", "marathi",
             "odia", "punjabi", "sanskrit", "tamil", "telugu"]
TRIAL_FILE = {"acoustic": "trials_{split}_final.csv", "cross": "trials_{split}_crossview.csv"}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def load_model(path, device):
    """Builds the model with the sizes stored in the checkpoint."""
    if path.endswith(".ckpt"):   # Lightning checkpoint
        sd = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
        sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    else:
        sd = torch.load(path, map_location="cpu", weights_only=True)
    ih = [k for k in sd if k.startswith("lstm_acoustic.lstm.weight_ih_l") and not k.endswith("_reverse")]
    arch = dict(
        input_dim_acoustic=sd["lstm_acoustic.lstm.weight_ih_l0"].shape[1],
        input_dim_text=sd["lstm_text.embedding.weight"].shape[1],
        no_of_tokens=sd["lstm_text.embedding.weight"].shape[0],
        hidden_dim=sd["lstm_acoustic.lstm.weight_hh_l0"].shape[1],
        embedding_dim=sd["lstm_acoustic.fc.weight"].shape[0],
        num_layers=len(ih),
        bidirectional=any(k.endswith("_reverse") for k in sd),
    )
    model = NAW_LSTM_multi_view_model(**arch)
    model.load_state_dict(sd)
    return model.to(device).eval(), arch


# ---------------------------------------------------------------------------
# Trials: read only the needed columns, dictionary-encode the string columns
# ---------------------------------------------------------------------------
def read_trials(path, str_cols):
    """Returns ({col: (codes int32 array, unique values list)}, gt int8 array)."""
    tbl = pacsv.read_csv(path, convert_options=pacsv.ConvertOptions(
        include_columns=str_cols + ["gt_decision"],
        column_types={**{c: pa.string() for c in str_cols}, "gt_decision": pa.int8()}))
    out = {}
    for c in str_cols:
        enc = tbl[c].combine_chunks().dictionary_encode()
        out[c] = (enc.indices.to_numpy(zero_copy_only=False).astype(np.int32), enc.dictionary.to_pylist())
    return out, tbl["gt_decision"].combine_chunks().to_numpy(zero_copy_only=False)


def to_global(codes, local_values, index):
    """Map per-column dictionary codes to indices in `index` (-1 if absent)."""
    lookup = np.array([index.get(v, -1) for v in local_values], dtype=np.int64)
    return lookup[codes]


# ---------------------------------------------------------------------------
# Embeddings (cached per checkpoint / language / split)
# ---------------------------------------------------------------------------
def _load_mel(path):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x = torch.load(path, map_location="cpu", weights_only=True).float()
        if x.dim() == 3:
            x = x.squeeze(0)
        return x if x.dim() == 2 and x.shape[0] > 0 else None
    except Exception:
        return None


@torch.no_grad()
def encode_audio(model, feature_dir, names, device, batch_size, workers):
    with ThreadPoolExecutor(workers) as pool:
        mels = list(pool.map(_load_mel, [os.path.join(feature_dir, n) for n in names]))
    ok = [i for i, m in enumerate(mels) if m is not None]
    emb = np.zeros((len(names), model.lstm_acoustic.fc.out_features), dtype=np.float32)
    order = sorted(ok, key=lambda i: mels[i].shape[0])        # length-sorted -> little padding
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        xs = [mels[i] for i in idx]
        lengths = torch.tensor([x.shape[0] for x in xs])
        e = model.lstm_acoustic(pad_sequence(xs, batch_first=True).to(device), lengths)
        emb[idx] = torch.nn.functional.normalize(e, dim=-1).cpu().numpy()
    valid = np.zeros(len(names), dtype=bool)
    valid[ok] = True
    return emb, valid


@torch.no_grad()
def encode_text(model, words, phones, lexicon, device, batch_size):
    wordparse = None
    seqs = []
    for w in words:
        ph = lexicon.get(w)
        if ph is None:
            if wordparse is None:
                from indic_unified_parser.uparser import wordparse
            try:
                ph = wordparse(w.strip(), 0, 0, 1).split()
            except Exception:
                ph = []
        seqs.append(torch.tensor([phones.get(p, 1) for p in (ph or ["SIL"])], dtype=torch.int64))
    emb = np.zeros((len(words), model.lstm_text.fc.out_features), dtype=np.float32)
    order = sorted(range(len(words)), key=lambda i: len(seqs[i]))
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        xs = [seqs[i] for i in idx]
        lengths = torch.tensor([len(x) for x in xs])
        e = model.lstm_text(pad_sequence(xs, batch_first=True).to(device), lengths)
        emb[idx] = torch.nn.functional.normalize(e, dim=-1).cpu().numpy()
    return emb


# ---------------------------------------------------------------------------
# Scoring and metrics
# ---------------------------------------------------------------------------
@torch.no_grad()
def pair_scores(A, B, ia, ib, device, chunk=1 << 20):
    A = torch.from_numpy(A).to(device)
    B = torch.from_numpy(B).to(device)
    out = np.empty(len(ia), dtype=np.float32)
    for s in range(0, len(ia), chunk):
        a = A[torch.from_numpy(ia[s:s + chunk]).to(device)]
        b = B[torch.from_numpy(ib[s:s + chunk]).to(device)]
        out[s:s + chunk] = (a * b).sum(1).cpu().numpy()
    return out


def metrics(y_true, y_score):
    ap = average_precision_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr
    i = np.nanargmin(np.abs(fpr - fnr))
    return ap, (fpr[i] + fnr[i]) / 2.0


# ---------------------------------------------------------------------------
def evaluate_lang_split(model, lang, split, views, args, phones, lexicon, device, cache_dir):
    lang_dir = os.path.join(args.trials_root, lang)
    feature_dir = os.path.join(lang_dir, f"features_{split}")
    t0 = time.time()

    trials = {}
    if "acoustic" in views:
        trials["acoustic"] = read_trials(os.path.join(lang_dir, TRIAL_FILE["acoustic"].format(split=split)), ["file1", "file2"])
    if "cross" in views:
        trials["cross"] = read_trials(os.path.join(lang_dir, TRIAL_FILE["cross"].format(split=split)), ["file1", "word2"])
    t_read = time.time() - t0

    # unique audio segments and text words needed by the requested views
    audio_names = sorted({n for cols, _ in trials.values() for c in ("file1", "file2") if c in cols for n in cols[c][1]})
    text_words = sorted(set(trials["cross"][0]["word2"][1])) if "cross" in trials else []

    cache = os.path.join(cache_dir, f"{lang}_{split}.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=False)
        cached_audio = dict(zip(z["audio_names"].tolist(), range(len(z["audio_names"]))))
        cached_text = dict(zip(z["text_words"].tolist(), range(len(z["text_words"]))))
        if all(n in cached_audio for n in audio_names) and all(w in cached_text for w in text_words):
            audio_names, text_words = z["audio_names"].tolist(), z["text_words"].tolist()
            A, valid, T = z["audio_emb"], z["audio_valid"], z["text_emb"]
        else:
            os.remove(cache)
    if not os.path.exists(cache):
        A, valid = encode_audio(model, feature_dir, audio_names, device, args.batch_size, args.workers)
        T = encode_text(model, text_words, phones, lexicon, device, args.batch_size) if text_words \
            else np.zeros((0, A.shape[1]), dtype=np.float32)
        np.savez(cache, audio_names=np.array(audio_names), audio_valid=valid, audio_emb=A,
                 text_words=np.array(text_words, dtype=str), text_emb=T)
    t_enc = time.time() - t0 - t_read

    a_index = {n: i for i, n in enumerate(audio_names)}
    t_index = {w: i for i, w in enumerate(text_words)}
    rows = []
    for view, (cols, gt) in trials.items():
        i1 = to_global(*cols["file1"], a_index)
        if view == "acoustic":
            i2 = to_global(*cols["file2"], a_index)
            keep = (i1 >= 0) & (i2 >= 0)
            keep[keep] &= valid[i1[keep]] & valid[i2[keep]]
            scores = pair_scores(A, A, i1[keep], i2[keep], device)
        else:
            i2 = to_global(*cols["word2"], t_index)
            keep = (i1 >= 0) & (i2 >= 0)
            keep[keep] &= valid[i1[keep]]
            scores = pair_scores(A, T, i1[keep], i2[keep], device)
        if args.round >= 0:
            scores = np.round(scores.astype(np.float64), args.round)
        y = gt[keep].astype(np.int64)
        ap, eer = metrics(y, scores)
        if args.save_scores:
            np.save(os.path.join(cache_dir, f"scores_{view}_{lang}_{split}.npy"), scores)
        rows.append(dict(lang=lang, split=split, view=view, AP=100 * ap, EER=100 * eer,
                         trials=int(keep.sum()), positives=int(y.sum()), dropped=int((~keep).sum())))
        print(f"RESULT,{view},{lang},{split},{ap:.8f},{eer:.8f}", flush=True)
    print(f"  {lang}/{split}: {len(audio_names)} clips, {len(text_words)} words | "
          f"read {t_read:.0f}s, encode {t_enc:.0f}s, score+metrics {time.time() - t0 - t_read - t_enc:.0f}s", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--langs", nargs="+", default=LANGUAGES)
    ap.add_argument("--splits", nargs="+", default=["iv", "oov"], choices=["iv", "oov"])
    ap.add_argument("--views", nargs="+", default=["acoustic", "cross"], choices=["acoustic", "cross"])
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=16, help="threads for loading feature files")
    ap.add_argument("--round", type=int, default=4, help="decimals to round scores to (old scripts: 4); -1 = off")
    ap.add_argument("--trials_root", default=TRIALS_ROOT)
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--save_scores", action="store_true", help="also save per-trial scores (.npy)")
    args = ap.parse_args()

    # exact float32: with TF32 on, batched and single-clip LSTM outputs differ by ~1e-4,
    # enough to change scores at the 4th decimal
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    model, arch = load_model(args.model_path, device)
    ckpt = os.path.abspath(args.model_path)
    tag = f"{os.path.basename(os.path.dirname(ckpt))}__{os.path.splitext(os.path.basename(ckpt))[0]}__{int(os.path.getmtime(ckpt))}"
    cache_dir = os.path.join("exp/eval_cache", tag)
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"model {args.model_path} | {arch} | device {device}\ncache {cache_dir}", flush=True)

    phones = {}
    with open(PHONES_FILE, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) >= 2:
                phones[p[0]] = int(p[1])
    lexicon = {}
    with open(LEXICON_FILE, encoding="utf-8") as f:
        for line in f:
            p = line.strip().split()
            if len(p) >= 2:
                lexicon[p[0]] = p[1:]

    rows, t0 = [], time.time()
    for lang in args.langs:
        for split in args.splits:
            rows += evaluate_lang_split(model, lang, split, args.views, args, phones, lexicon, device, cache_dir)

    df = pd.DataFrame(rows)
    out_csv = os.path.join(args.out_dir, f"word_discrimination_{tag}.csv")
    df.to_csv(out_csv, index=False)
    table = df.pivot_table(index="lang", columns=["view", "split"], values="AP").round(1)
    if len(table) > 1:
        table.loc["Mean"] = table.mean().round(1)
    print("\nAP (%)\n" + table.to_string())
    eer = df.pivot_table(index="lang", columns=["view", "split"], values="EER").round(1)
    if len(eer) > 1:
        eer.loc["Mean"] = eer.mean().round(1)
    print("\nEER (%)\n" + eer.to_string())
    print(f"\nsaved {out_csv}  ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
