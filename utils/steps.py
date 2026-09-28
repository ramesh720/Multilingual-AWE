import torch
import os
import copy
import matplotlib.pyplot as plt
import numpy as np
from conf.conf import Config
from utils.loss_functions import CLAP_loss, DeepWordDiscriminationLoss
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Shared loss computation — CLAP + DWD, per Eq. 14 of the JMCL paper:
#
#   L_total = alpha1 * (1/M) * sum_m L_at^(m)   +   alpha2 * L_aa
#
# acoustic_embeddings : (N, M, D) — N unique words (batch sampler
#                        guarantees this), M same-word audio views each.
# text_embeddings     : (N, D)    — one text embedding per word (the
#                        keyword's phoneme sequence doesn't change across
#                        its M audio views, so we only encode it once).
# word_labels         : (N,)      — integer word id per word in the batch.
# ---------------------------------------------------------------------------
def _compute_losses(model, acoustic_embeddings, text_embeddings, word_labels,
                     config, loss_fn_clap, loss_fn_dwd):
    N, M, D = acoustic_embeddings.shape

    if N == 0:
        zero = acoustic_embeddings.sum() * 0.0
        return zero, zero, zero

    # --- CLAP: one N x N logits matrix per view, averaged over M (Eq. 14) ---
    clap_losses = []
    for m in range(M):
        audio_view = acoustic_embeddings[:, m, :]                    # (N, D)
        logits = model.logit_scale.exp() * (text_embeddings @ audio_view.T)
        clap_losses.append(loss_fn_clap(audio_view, logits))
    clap_loss = torch.stack(clap_losses).mean()

    # --- DWD: across all N*M audio embeddings in the batch (Eq. 10-13) ---
    acoustic_flat = acoustic_embeddings.reshape(N * M, D)
    labels_flat   = word_labels.unsqueeze(1).expand(N, M).reshape(N * M)
    dwd_loss = loss_fn_dwd(acoustic_flat, labels_flat)

    total_loss = config.wt_clap_loss * clap_loss + config.wt_dwd_loss * dwd_loss
    return total_loss, clap_loss, dwd_loss


def _forward_model(model, melspecgram, anc_sequences,
                    melspecgram_lengths, anc_sequences_lengths):
    """
    melspecgram         : (N, M, T, n_mels)
    melspecgram_lengths : (N, M)
    anc_sequences        : (N, P)
    anc_sequences_lengths: (N,)

    Flattens the audio to (N*M, T, n_mels) for the audio encoder (the
    encoder doesn't care about the word grouping — that's only needed
    afterward for the loss computation), while text stays at (N, P)
    since there's only one phoneme sequence per word. Returns
    acoustic_embeddings reshaped back to (N, M, D) and text_embeddings
    as (N, D).

    NOTE: this assumes model.forward(mels, text, mel_lengths, text_lengths)
    encodes the audio and text branches independently and does not require
    their batch dimensions to match (true of the original code, which
    computes CLAP logits externally via text @ audio.T rather than inside
    the model). If your model asserts equal batch sizes internally, this
    will need a small change there instead.
    """
    N, M, T, n_mels = melspecgram.shape
    mels_flat        = melspecgram.reshape(N * M, T, n_mels)
    mel_lengths_flat = melspecgram_lengths.reshape(N * M)

    acoustic_emb_flat, text_emb, _ = model(
        mels_flat, anc_sequences, mel_lengths_flat, anc_sequences_lengths
    )
    D = acoustic_emb_flat.shape[-1]
    acoustic_emb = acoustic_emb_flat.view(N, M, D)

    return acoustic_emb, text_emb


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------
def train_step(model, dataloader, loss_fn_clap, loss_fn_dwd,
               optimizer, scheduler, writer, device, epoch, config):
    model.train()
    total_loss_acc = 0.0
    total_clap_acc = 0.0
    total_dwd_acc  = 0.0
    valid_batches = 0

    progress_bar = tqdm(dataloader, desc=f"Train Epoch {epoch+1}")
    for data in progress_bar:
        if data is None or data[0].numel() == 0:
            continue

        melspecgram = data[0].to(device)
        anc_sequences = data[1].to(device)
        melspecgram_lengths = data[2].to('cpu')
        word_labels = data[3].to(device)
        anc_sequences_lengths = data[4].to('cpu')

        acoustic_emb, text_emb = _forward_model(
            model, melspecgram, anc_sequences,
            melspecgram_lengths, anc_sequences_lengths
        )

        total_loss, clap_loss, dwd_loss = _compute_losses(
            model, acoustic_emb, text_emb, word_labels,
            config, loss_fn_clap, loss_fn_dwd
        )

        if total_loss.item() == 0.0:
            continue

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss_acc += total_loss.item()
        total_clap_acc += clap_loss.item()
        total_dwd_acc  += dwd_loss.item()
        valid_batches += 1

        progress_bar.set_postfix({
            'loss': f"{total_loss.item():.4f}",
            'clap': f"{(clap_loss * config.wt_clap_loss).item():.4f}",
            'dwd' : f"{(dwd_loss * config.wt_dwd_loss).item():.4f}",
        })

    if valid_batches == 0:
        return 0.0, 0.0, 0.0
    return (total_loss_acc / valid_batches,
            total_clap_acc / valid_batches,
            total_dwd_acc / valid_batches)


# ---------------------------------------------------------------------------
# Validation step
# ---------------------------------------------------------------------------
def val_step(model, dataloader, loss_fn_clap, loss_fn_dwd, device, config):
    model.eval()
    total_loss_acc = 0.0
    total_clap_acc = 0.0
    total_dwd_acc  = 0.0
    valid_batches = 0

    progress_bar = tqdm(dataloader, desc="Validation")
    with torch.no_grad():
        for data in progress_bar:
            if data is None or data[0].numel() == 0:
                continue

            melspecgram = data[0].to(device)
            anc_sequences = data[1].to(device)
            melspecgram_lengths = data[2].to('cpu')
            word_labels = data[3].to(device)
            anc_sequences_lengths = data[4].to('cpu')

            acoustic_emb, text_emb = _forward_model(
                model, melspecgram, anc_sequences,
                melspecgram_lengths, anc_sequences_lengths
            )

            total_loss, clap_loss, dwd_loss = _compute_losses(
                model, acoustic_emb, text_emb, word_labels,
                config, loss_fn_clap, loss_fn_dwd
            )

            if total_loss.item() == 0.0:
                continue

            total_loss_acc += total_loss.item()
            total_clap_acc += clap_loss.item()
            total_dwd_acc  += dwd_loss.item()
            valid_batches += 1

            progress_bar.set_postfix({
                'val_loss': f"{total_loss.item():.4f}",
                'clap': f"{(clap_loss * config.wt_clap_loss).item():.4f}",
                'dwd' : f"{(dwd_loss * config.wt_dwd_loss).item():.4f}",
            })

    if valid_batches == 0:
        return 0.0, 0.0, 0.0
    return (total_loss_acc / valid_batches,
            total_clap_acc / valid_batches,
            total_dwd_acc / valid_batches)


# ---------------------------------------------------------------------------
# Embedding diagnostics (returns similarity stats)
# ---------------------------------------------------------------------------
def run_embedding_diagnostics(model, val_loader, device, epoch, writer, max_samples=300):
    """
    Computes positive/negative similarity stats using ONE audio view per
    word (view 0) against its text embedding, and returns them. Using a
    single view keeps this comparable to the pre-DWD diagnostics (one
    embedding per word) rather than mixing in M redundant copies.
    """
    model.eval()
    all_acoustic = []
    all_text = []
    all_labels = []
    seen = 0

    with torch.no_grad():
        for data in val_loader:
            if data is None or data[0].numel() == 0:
                continue
            if seen >= max_samples:
                break

            mels = data[0].to(device)
            anc = data[1].to(device)
            mel_lens = data[2].to('cpu')
            labels = data[3]
            anc_lens = data[4].to('cpu')

            acoustic_emb, text_emb = _forward_model(
                model, mels, anc, mel_lens, anc_lens
            )
            a_emb_view0 = acoustic_emb[:, 0, :]   # (N, D) — one view per word

            all_acoustic.append(a_emb_view0.cpu())
            all_text.append(text_emb.cpu())
            all_labels.append(labels)
            seen += a_emb_view0.size(0)

    if not all_acoustic:
        return None, None, None, None, None

    a_emb = torch.cat(all_acoustic, dim=0)
    t_emb = torch.cat(all_text, dim=0)
    labels = torch.cat(all_labels, dim=0)

    a_emb = torch.nn.functional.normalize(a_emb, dim=1)
    t_emb = torch.nn.functional.normalize(t_emb, dim=1)

    sim_matrix = a_emb @ t_emb.T  # (N, N)

    same_word_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
    eye = torch.eye(len(labels), dtype=torch.bool)
    same_word_mask = same_word_mask & ~eye

    sim_pos = sim_matrix[same_word_mask]

    neg_mask = ~same_word_mask & ~eye
    neg_indices = torch.nonzero(neg_mask)
    if neg_indices.size(0) > 10000:
        perm = torch.randperm(neg_indices.size(0))[:10000]
        neg_indices = neg_indices[perm]
    sim_neg = sim_matrix[neg_indices[:, 0], neg_indices[:, 1]]

    # Retrieval@1
    nearest = sim_matrix.argmax(dim=1)
    retrieval_acc = (labels == labels[nearest]).float().mean()

    pos_mean = sim_pos.mean().item()
    neg_mean = sim_neg.mean().item() if sim_neg.numel() > 0 else 0.0
    margin = pos_mean - neg_mean

    if writer is not None:
        writer.add_histogram('similarity/positive', sim_pos, epoch)
        if sim_neg.numel() > 0:
            writer.add_histogram('similarity/negative', sim_neg, epoch)
        writer.add_scalar('similarity/avg_pos', pos_mean, epoch)
        writer.add_scalar('similarity/avg_neg', neg_mean, epoch)
        writer.add_scalar('similarity/margin', margin, epoch)
        writer.add_scalar('retrieval/acc@1', retrieval_acc.item(), epoch)

    return pos_mean, neg_mean, margin, retrieval_acc.item(), (sim_pos, sim_neg)


# ---------------------------------------------------------------------------
# Plotting function (saves figure)
# --------------------------------------------------------------------------
def save_training_plot(metrics, save_path):
    """
    metrics: dict with lists: 'train_loss', 'val_loss', 'logit_scale',
             'pos_sim', 'neg_sim', 'margin', 'retrieval',
             'train_clap', 'train_dwd'
    """
    epochs = range(1, len(metrics['train_loss']) + 1)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Training Progress (CLAP + DWD)', fontsize=16)

    # Loss
    axes[0, 0].plot(epochs, metrics['train_loss'], 'b-', label='Train')
    axes[0, 0].plot(epochs, metrics['val_loss'], 'r-', label='Val')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # logit_scale
    axes[0, 1].plot(epochs, metrics['logit_scale'], 'g-')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('logit_scale')
    axes[0, 1].set_title('Temperature Parameter')
    axes[0, 1].grid(True)

    # Retrieval@1
    axes[0, 2].plot(epochs, metrics['retrieval'], 'm-')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Accuracy')
    axes[0, 2].set_title('Retrieval@1')
    axes[0, 2].grid(True)

    # Similarities
    axes[1, 0].plot(epochs, metrics['pos_sim'], 'b-', label='Positive')
    axes[1, 0].plot(epochs, metrics['neg_sim'], 'r-', label='Negative')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Cosine Similarity')
    axes[1, 0].set_title('Avg Similarities')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # CLAP vs DWD component losses
    axes[1, 1].plot(epochs, metrics['train_clap'], 'c-', label='CLAP')
    axes[1, 1].plot(epochs, metrics['train_dwd'], 'y-', label='DWD')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss (unweighted)')
    axes[1, 1].set_title('CLAP vs DWD components')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    # Histogram of similarities at last epoch (if available)
    if 'sim_pos_hist' in metrics and metrics['sim_pos_hist'] is not None:
        axes[1, 2].hist(metrics['sim_pos_hist'].numpy(), bins=50, alpha=0.5, label='Positive', color='blue')
        axes[1, 2].hist(metrics['sim_neg_hist'].numpy(), bins=50, alpha=0.5, label='Negative', color='red')
        axes[1, 2].set_xlabel('Cosine Similarity')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].set_title('Similarity Distribution (latest epoch)')
        axes[1, 2].legend()
        axes[1, 2].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main training loop (with plotting)
# ---------------------------------------------------------------------------
def train(model, train_dataloader, validation_dataloader,
          optimizer, scheduler, checkpoint_dir="saved_models",
          epochs=5, log_dir="logs", device=torch.device('cpu'),
          config=None):
    if config is None:
        config = Config()

    loss_fn_clap = CLAP_loss()
    loss_fn_dwd  = DeepWordDiscriminationLoss()

    writer = SummaryWriter(log_dir=log_dir)
    min_loss = float('inf')

    metrics = {
        'train_loss': [],
        'val_loss': [],
        'train_clap': [],
        'train_dwd': [],
        'logit_scale': [],
        'pos_sim': [],
        'neg_sim': [],
        'margin': [],
        'retrieval': [],
        'sim_pos_hist': None,
        'sim_neg_hist': None,
    }

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        train_loss, train_clap, train_dwd = train_step(
            model, train_dataloader, loss_fn_clap, loss_fn_dwd,
            optimizer, scheduler, writer, device, epoch, config
        )

        val_loss, val_clap, val_dwd = val_step(
            model, validation_dataloader, loss_fn_clap, loss_fn_dwd,
            device, config
        )

        logit_val = model.logit_scale.item()
        logit_exp = model.logit_scale.exp().item()
        print(
            f"Epoch {epoch+1:>3} | "
            f"train_loss={train_loss:.4f}  clap={train_clap:.4f}  dwd={train_dwd:.4f} | "
            f"val_loss={val_loss:.4f}  clap={val_clap:.4f}  dwd={val_dwd:.4f} | "
            f"logit_scale={logit_val:.4f} (exp={logit_exp:.2f})"
        )

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/train_clap', train_clap, epoch)
        writer.add_scalar('Loss/train_dwd', train_dwd, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Loss/val_clap', val_clap, epoch)
        writer.add_scalar('Loss/val_dwd', val_dwd, epoch)
        writer.add_scalar('logit_scale', logit_val, epoch)

        pos_mean, neg_mean, margin, ret_acc, hist_tensors = run_embedding_diagnostics(
            model, validation_dataloader, device, epoch, writer
        )

        metrics['train_loss'].append(train_loss)
        metrics['val_loss'].append(val_loss)
        metrics['train_clap'].append(train_clap)
        metrics['train_dwd'].append(train_dwd)
        metrics['logit_scale'].append(logit_val)
        if pos_mean is not None:
            metrics['pos_sim'].append(pos_mean)
            metrics['neg_sim'].append(neg_mean)
            metrics['margin'].append(margin)
            metrics['retrieval'].append(ret_acc)
            if hist_tensors is not None:
                metrics['sim_pos_hist'] = hist_tensors[0]
                metrics['sim_neg_hist'] = hist_tensors[1]

        if len(metrics['train_loss']) > 0:
            plot_path = os.path.join(log_dir, f'training_progress_epoch_{epoch+1}.png')
            save_training_plot(metrics, plot_path)
            print(f"  📊 Saved plot → {plot_path}")

        if val_loss < min_loss:
            min_loss = val_loss
            os.makedirs(checkpoint_dir, exist_ok=True)
            save_path = os.path.join(checkpoint_dir, f'NAW_best_epoch{epoch}.pt')
            torch.save(copy.deepcopy(model).state_dict(), save_path)
            print(f"  ✅ Saved best checkpoint → {save_path}")

    writer.close()
    final_plot = os.path.join(log_dir, 'training_progress_final.png')
    save_training_plot(metrics, final_plot)
    print(f"  📊 Final plot saved → {final_plot}")