import torch
import torch.nn as nn
import torch.nn.functional as F

def cosine_dist(a, b):
    return 1 - (a * b).sum(dim=1)

class MultiObjectiveContrastiveLoss(nn.Module):
    def __init__(self, margin=0.3, reduction='mean'):
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, fx_pos, fx_neg, gc_pos, gc_neg):
        d_pos = cosine_dist(fx_pos, gc_pos)
        d_neg_text = cosine_dist(gc_pos, gc_neg)
        d_neg_audio = cosine_dist(fx_neg, gc_pos)
        d_audio_audio = cosine_dist(fx_pos, fx_neg)

        obj0 = F.relu(self.margin + d_pos - cosine_dist(fx_pos, gc_neg))
        obj1 = F.relu(self.margin + d_pos - d_neg_text)
        obj2 = F.relu(self.margin + d_pos - d_neg_audio)
        obj3 = F.relu(self.margin + d_pos - d_audio_audio)

        if self.reduction == 'mean':
            return {'obj0': obj0.mean(), 'obj1': obj1.mean(), 'obj2': obj2.mean(), 'obj3': obj3.mean()}
        return {'obj0': obj0, 'obj1': obj1, 'obj2': obj2, 'obj3': obj3}

class CLAP_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, embeddings, logits):
        labels = torch.arange(embeddings.size(0), device=embeddings.device)
        loss_t2a = F.cross_entropy(logits, labels)
        loss_a2t = F.cross_entropy(logits.T, labels)
        return 0.5 * (loss_t2a + loss_a2t)

class DeepWordDiscriminationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, embeddings: torch.Tensor, word_labels: torch.Tensor) -> torch.Tensor:
        device = embeddings.device
        B, D = embeddings.size()

        # 1. Normalize input embeddings
        embeddings = F.normalize(embeddings, dim=1)

        # 2. Build Intra-class mask (where do the word labels match?)
        same_label = word_labels.unsqueeze(0) == word_labels.unsqueeze(1)  # (B, B)
        eye_mask = ~torch.eye(B, dtype=torch.bool, device=device)          # (B, B)
        mask = same_label & eye_mask                                       # (B, B)

        # Calculate localized centroids for matching words
        class_sizes = mask.sum(dim=1, keepdim=True).clamp(min=1)
        centroids = (mask.float() @ embeddings) / class_sizes              # (B, D)
        centroids = F.normalize(centroids, dim=1)

        # 3. Softmax-based Loss (L_sm) across all unique words in this batch
        unique_labels, label_to_class = torch.unique(word_labels, return_inverse=True)
        N_word = unique_labels.size(0)

        class_mask = word_labels.unsqueeze(1) == unique_labels.unsqueeze(0)  # (B, N_word)
        class_counts = class_mask.sum(dim=0, keepdim=True).clamp(min=1)
        class_centroids = (class_mask.float().T @ embeddings) / class_counts.T  # (N_word, D)
        class_centroids = F.normalize(class_centroids, dim=1)

        sim_matrix = embeddings @ class_centroids.T
        log_probs = F.log_softmax(sim_matrix, dim=1)
        L_sm = -log_probs[torch.arange(B, device=device), label_to_class].mean()

        # 4. Multi-Cluster Contrastive Centroid Loss (L_cc)
        sim_pos = F.cosine_similarity(embeddings, centroids, dim=1)  # (B,)
        sim_matrix_neg = sim_matrix.clone()
        
        # Safe masking: if only 1 unique word exists, set negative similarity safely to 0
        if N_word <= 1:
            sim_neg = torch.zeros_like(sim_pos)
        else:
            sim_matrix_neg[torch.arange(B, device=device), label_to_class] = -1e9
            sim_neg = sim_matrix_neg.max(dim=1).values  # Highest similarity to a DIFFERENT word class

        L_cc = F.relu((1 - sim_pos) + sim_neg).mean()

        return L_sm + L_cc
