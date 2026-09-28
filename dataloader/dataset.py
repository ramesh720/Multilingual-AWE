"""
=============================================================================
PHONES / LEXICON: single shared "overall" files covering all languages,
located at:
  /home/siplabiith/clap_dwd_overall/data/lang/overall/phones.txt
  /home/siplabiith/clap_dwd_overall/data/lang/overall/lexicon.txt

No per-language union or per-language lookup needed -- just load these
two files once. Word lookup in the lexicon uses the RAW word (language
tag stripped off the "lang__word" key), since the overall lexicon is
presumably already unique across languages by virtue of covering
distinct scripts.
=============================================================================
"""

import random
import numpy as np
import torch
import pickle
import os
import torchaudio
from torch.utils.data import Dataset, Sampler
from torch.nn.utils.rnn import pad_sequence
from indic_unified_parser.uparser import wordparse


# Common root that, together with the full relative path stored in the
# merged pkl's file_id, produces the actual audio file path on disk.
AUDIO_ROOT = '/home/siplabiith/'

# Shared phone inventory + lexicon covering ALL languages (single files,
# not per-language).
PHONES_PATH   = '/home/siplabiith/clap_dwd_overall/data/lang/overall/phones.txt'
LEXICON_PATH  = '/home/siplabiith/clap_dwd_overall/data/lang/overall/lexicon.txt'


class TripletSpeechDataset(Dataset):
    def __init__(self, config, transform=None, type="train"):
        self.config = config
        self.type   = type

        raw_data = pickle.load(
            open(self.config.data_info_dict[type]['alignments_path'], "rb")
        )
        top_keys  = list(raw_data.keys())
        first_val = raw_data[top_keys[0]]

        # Combined multilingual pkl (from merge_lang_pkls.py /
        # merge_iv_oov_pkls.py) is ALREADY flat: {"lang__word": {...}, ...}
        # Legacy single-language train pkl has one wrapper key ('telugu')
        # whose value is the actual word dict. IV/OOV single-language pkls
        # are already flat {word: {...}}.
        if (len(top_keys) == 1
                and isinstance(first_val, dict)
                and all(isinstance(v, (dict, list))
                        for v in list(first_val.values())[:10])):
            self.word_dict = first_val          # unwrap language key
            print(f"  [dataset:{type}] nested layout — unwrapped '{top_keys[0]}'")
        else:
            self.word_dict = raw_data           # already flat (combined or legacy-flat)
            print(f"  [dataset:{type}] flat layout, {len(top_keys)} keys")

        self.words     = list(self.word_dict.keys())
        self.transform = transform

        # Integer label per word — used for the unique-word batch sampler
        # AND for the DWD loss's word_labels tensor. With the combined
        # pkl, each key already includes the language tag, so
        # "telugu__పదం" and "tamil__வார்த்தை" get distinct labels even
        # if the untagged words happened to collide.
        self.word2idx  = {w: i for i, w in enumerate(self.words)}

        self.M = getattr(config, 'M_views', config.number_examples_wer_word)

        # ------------------------------------------------------------
        # Shared phone inventory covering all languages (single file).
        # ------------------------------------------------------------
        self.phones_dict = {}
        if not os.path.exists(PHONES_PATH):
            raise FileNotFoundError(f"phones.txt not found at {PHONES_PATH}")
        with open(PHONES_PATH) as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    self.phones_dict[parts[0]] = int(parts[1])
        print(f"  [dataset:{type}] loaded phones.txt from {PHONES_PATH} "
              f"({len(self.phones_dict)} phone symbols)")

        # ------------------------------------------------------------
        # Shared pronunciation lexicon covering all languages (single
        # file): word -> [phone, phone, ...]. Looked up by the raw word
        # after stripping the "lang__" tag from the pkl key.
        # ------------------------------------------------------------
        self.lexicon = {}
        if not os.path.exists(LEXICON_PATH):
            raise FileNotFoundError(f"lexicon.txt not found at {LEXICON_PATH}")
        with open(LEXICON_PATH) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    self.lexicon[parts[0]] = parts[1:]
        print(f"  [dataset:{type}] loaded lexicon.txt from {LEXICON_PATH} "
              f"({len(self.lexicon)} words)")

        # Build flat instance list: (word, file_id, (start, end))
        # `word` here is the TAGGED key, e.g. "telugu__పదం"
        self.instances = []
        is_test = self.type in ['test', 'val', 'validation']

        for word in self.words:
            entry = self.word_dict[word]

            if isinstance(entry, dict):
                if is_test or len(entry) >= self.config.number_examples_wer_word:
                    for file_id, times in entry.items():
                        if (isinstance(times, (list, tuple))
                                and len(times) > 0
                                and isinstance(times[0], (list, tuple))):
                            for t in times:
                                self.instances.append((word, file_id, t))
                        else:
                            self.instances.append((word, file_id, times))

            elif isinstance(entry, list):
                if is_test or len(entry) >= self.config.number_examples_wer_word:
                    for item in entry:
                        if isinstance(item, (tuple, list)) and len(item) >= 2:
                            self.instances.append((word, item[0], item[1]))

        # word -> list of (file_id, (start, end)) — the pool we sample M
        # views from in __getitem__.
        self.word_to_pool = {}
        for word, file_id, times in self.instances:
            self.word_to_pool.setdefault(word, []).append((file_id, times))

        print(f"  [dataset:{type}] {len(self.instances)} instances, "
              f"{len(self.words)} words, M={self.M} views/word")
        self._resampler_cache = {}

    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.instances)

    # ------------------------------------------------------------------

    @staticmethod
    def _split_lang_word(tagged_word):
        """'telugu__పదం' -> ('telugu', 'పదం'). Falls back to (None, word)
        if there's no '__' separator (legacy untagged single-language pkl)."""
        if '__' in tagged_word:
            lang, raw_word = tagged_word.split('__', 1)
            return lang, raw_word
        return None, tagged_word

    # ------------------------------------------------------------------

    def _load_segment(self, file_id, start, end, lang=None):
        """
        Load one word segment from disk.

        The relative path stored in the pkl (e.g.
        'Kathbath/kb_data_clean_m4a/telugu/train/audio/514-m/....wav')
        does NOT reliably match the real on-disk layout across
        languages: some languages nest audio under a per-speaker
        subfolder (confirmed for Telugu), others store it flat directly
        under train/audio/ (confirmed for Bengali, likely also Tamil).

        Strategy: try the path exactly as stored first (extension
        swapped .wav -> .m4a). If that doesn't exist, fall back to a
        flattened version of the same path (same folder, basename only,
        no subfolder) before giving up. This covers both layouts without
        needing per-language special-casing.
        """
        fid = str(file_id).strip()

        candidates = []

        if fid.startswith('Kathbath/'):
            if fid.endswith('.wav'):
                fid_m4a = fid[:-4] + '.m4a'
            elif not fid.endswith('.m4a'):
                fid_m4a = fid + '.m4a'
            else:
                fid_m4a = fid

            # (a) exactly as stored
            candidates.append(os.path.join(AUDIO_ROOT, fid_m4a))

            # (b) flattened: drop any subfolder immediately before the
            # filename, e.g. ".../train/audio/1038-m/x.m4a" ->
            # ".../train/audio/x.m4a"
            parts = fid_m4a.split('/')
            if len(parts) >= 2:
                flattened = '/'.join(parts[:-2] + parts[-1:])
                candidates.append(os.path.join(AUDIO_ROOT, flattened))
        else:
            base = (os.path.basename(fid)
                      .replace('.wav', '')
                      .replace('.m4a', '')
                      .strip())
            if lang is not None:
                candidates.append(
                    f'{AUDIO_ROOT}Kathbath/kb_data_clean_m4a/'
                    f'{lang}/train/audio/{base}.m4a'
                )

        path = None
        for c in candidates:
            if os.path.exists(c):
                path = c
                break

        if path is None:
            return None
        try:
            waveform, sr = torchaudio.load(path)
            s = int(float(start) * sr)
            e = int(float(end) * sr)
            seg = waveform[:, s:e]

            if seg.shape[1] < 10:       # discard near-empty segments
                return None

            target_sr = getattr(self.config, 'sample_rate', 16000)
            if sr != target_sr:
                if sr not in self._resampler_cache:
                    self._resampler_cache[sr] = torchaudio.transforms.Resample(
                        orig_freq=sr,
                        new_freq=target_sr
                    )
                seg = self._resampler_cache[sr](seg)

            if self.transform:
                seg = self.transform(seg)

            return seg.squeeze(0).transpose(0, 1)   # (T, n_mels)
        except Exception as e:
            print(f"❌ _load_segment failed for {path}")
            print(f"   start={start}, end={end}")
            print(f"   Error: {type(e).__name__}: {e}")
            return None

    # ------------------------------------------------------------------

    def _sample_word_group(self, tagged_word, M):
        """
        Sample M valid audio segments for `tagged_word` (e.g.
        "telugu__పదం"), without replacement when the pool is large
        enough, falling back to sampling with replacement if segments
        fail to load or the pool is smaller than M.
        """
        pool = self.word_to_pool.get(tagged_word, [])
        if not pool:
            return None

        lang, _raw_word = self._split_lang_word(tagged_word)

        pool_order = list(range(len(pool)))
        random.shuffle(pool_order)

        mels = []
        pi = 0
        max_attempts = M * 6 + 10
        attempts = 0

        while len(mels) < M and attempts < max_attempts:
            attempts += 1
            if pi < len(pool_order):
                file_id, times = pool[pool_order[pi]]
                pi += 1
            else:
                file_id, times = random.choice(pool)  # replacement fallback

            seg = self._load_segment(file_id, *times, lang=lang)
            if seg is not None:
                mels.append(seg)

        if len(mels) < M:
            return None
        return mels

    # ------------------------------------------------------------------

    def __getitem__(self, idx):
        """
        Returns M audio views of ONE word (for DWD) plus ONE shared
        phoneme sequence for that word (for CLAP) — the word itself is
        chosen by `UniqueWordBatchSampler`, which guarantees no two
        items in the same batch share a word. `collate_fn` stacks these
        into (N, M, T, n_mels) audio and (N, P) text tensors.
        """
        anchor_word, _, _ = self.instances[idx]   # tagged, e.g. "telugu__పదం"
        lang, raw_word = self._split_lang_word(anchor_word)

        mels = self._sample_word_group(anchor_word, self.M)
        if mels is None:
            return None

        if raw_word in self.lexicon:
            phonemes = self.lexicon[raw_word]
        else:
            try:
                phonemes = wordparse(raw_word, 0, 0, 1).split()
            except Exception:
                phonemes = []

        if not phonemes:
            phonemes = ['SIL']

        anc_seq = torch.IntTensor(
            [self.phones_dict.get(p, 1) for p in phonemes]
        )                           # shape: (P,)

        return {
            'mels'          : mels,                        # list of M tensors, each (T_i, n_mels)
            'anchor_word'   : anchor_word,
            'word_label'    : self.word2idx[anchor_word],
            'anc_seq'       : anc_seq,
        }


# ---------------------------------------------------------------------------
# Batch sampler — guarantees every WORD in a batch is DIFFERENT.
# Unchanged from before: operates on tagged word keys, so uniqueness is
# already enforced across languages too (no code change needed here).
# ---------------------------------------------------------------------------
class UniqueWordBatchSampler(Sampler):
    """
    Yields lists of dataset indices such that no two indices in the same
    batch share a word. Cycles through every word's instances across an
    epoch, so a word with many recordings still gets seen multiple times
    over the epoch — just never twice in the *same* batch.
    """

    def __init__(self, dataset, batch_size, shuffle=True, drop_last=True):
        self.dataset    = dataset
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.drop_last  = drop_last

        self.word_to_indices = {}
        for idx, (word, _, _) in enumerate(dataset.instances):
            self.word_to_indices.setdefault(word, []).append(idx)

    def __iter__(self):
        word_pools = {w: idxs.copy() for w, idxs in self.word_to_indices.items()}
        if self.shuffle:
            for idxs in word_pools.values():
                random.shuffle(idxs)

        pointers = {w: 0 for w in word_pools}
        remaining_words = [w for w, idxs in word_pools.items() if idxs]

        while remaining_words:
            if self.shuffle:
                random.shuffle(remaining_words)

            if len(remaining_words) < self.batch_size and self.drop_last:
                break

            batch_words = remaining_words[:self.batch_size]
            batch = []
            still_remaining = []

            for w in remaining_words:
                if w in batch_words:
                    p = pointers[w]
                    batch.append(word_pools[w][p])
                    pointers[w] += 1
                    if pointers[w] < len(word_pools[w]):
                        still_remaining.append(w)
                else:
                    still_remaining.append(w)

            remaining_words = still_remaining
            yield batch

    def __len__(self):
        total_instances = len(self.dataset.instances)
        if self.drop_last:
            return total_instances // self.batch_size
        return -(-total_instances // self.batch_size)  # ceil division


# ---------------------------------------------------------------------------
# Collate function — builds grouped (N, M, ...) tensors. Unchanged.
# ---------------------------------------------------------------------------
def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    N = len(batch)
    M = len(batch[0]['mels'])

    all_mels = []
    for b in batch:
        all_mels.extend(b['mels'])

    mel_lengths_flat = torch.LongTensor([m.shape[0] for m in all_mels])
    mels_padded_flat = pad_sequence(all_mels, batch_first=True)  # (N*M, T_max, n_mels)

    T_max  = mels_padded_flat.shape[1]
    n_mels = mels_padded_flat.shape[2]
    mels_padded = mels_padded_flat.view(N, M, T_max, n_mels)
    mel_lengths = mel_lengths_flat.view(N, M)

    anc_seqs    = [b['anc_seq'] for b in batch]                    # N items
    anc_lengths = torch.LongTensor([len(s) for s in anc_seqs])
    anc_padded  = pad_sequence(anc_seqs, batch_first=True)         # (N, P_max)

    word_labels = torch.LongTensor([b['word_label'] for b in batch])  # (N,)

    return mels_padded, anc_padded, mel_lengths, word_labels, anc_lengths