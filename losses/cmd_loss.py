import torch
import torch.nn.functional as F
from torch import nn


class CMDLoss(nn.Module):
    def __init__(
        self,
        # clip_loss_cache_labels=True,
        # alignment_loss_weight=0.5,
        he_retention_loss_weight=0.1,
        ihc_retention_loss_weight=0.1,
        he_ce_loss_weight=1.0,
        ihc_ce_loss_weight=1.0,
        # style_loss_weight=0.1,
        # cluster_loss_weight=0.2,
    ):
        super().__init__()

        self.he_retention_loss_weight = he_retention_loss_weight
        self.ihc_retention_loss_weight = ihc_retention_loss_weight
        self.he_ce_loss_weight = he_ce_loss_weight
        self.ihc_ce_loss_weight = ihc_ce_loss_weight

    def forward(
        self,
        he_retention_emb,
        he_retention_target,
        he_mask,
        he_results_dict,
        ihc_retention_emb,
        ihc_retention_target,
        ihc_mask,
        ihc_results_dict,
        label
    ):

        he_retention_loss = (he_retention_emb - he_retention_target) ** 2
        he_retention_loss = he_retention_loss.mean(dim=-1)
        he_retention_loss = (he_retention_loss * he_mask).sum() / he_mask.sum()

        ihc_retention_loss = (ihc_retention_emb - ihc_retention_target) ** 2
        ihc_retention_loss = ihc_retention_loss.mean(dim=-1)
        ihc_retention_loss = (ihc_retention_loss * ihc_mask).sum() / ihc_mask.sum()

        he_ce_loss = F.cross_entropy(he_results_dict["logits"], label)
        ihc_ce_loss = F.cross_entropy(ihc_results_dict["logits"], label)

        # style_loss = 0.5 * (
        #     torch.sum(
        #         torch.exp(wsi_logstd) + wsi_mu**2 - 1.0 - wsi_logstd, dim=1
        #     ).mean()
        #     + torch.sum(
        #         torch.exp(rna_logstd) + rna_mu**2 - 1.0 - rna_logstd, dim=1
        #     ).mean()
        # )

        # wsi_prob = F.softmax(wsi_score, dim=-1)
        # rna_prob = F.softmax(rna_score, dim=-1)
        # cluster_loss = 0.5 * (
        #     F.kl_div(wsi_prob.log(), rna_prob, reduction="batchmean")
        #     + F.kl_div(rna_prob.log(), wsi_prob, reduction="batchmean")
        # )

        total_loss = (
            self.he_retention_loss_weight * he_retention_loss
            + self.ihc_retention_loss_weight * ihc_retention_loss
            + self.he_ce_loss_weight * he_ce_loss
            + self.ihc_ce_loss_weight * ihc_ce_loss
        )
        return (
            total_loss,
            he_retention_loss,
            ihc_retention_loss,
            he_ce_loss,
            ihc_ce_loss,
        )
