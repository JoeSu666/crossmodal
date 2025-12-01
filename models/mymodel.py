import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention

from torch.distributions import Normal
from typing import Tuple, Type

LayerType = Type[nn.Module]

# ===========================================
#  TransMIL
# ===========================================
class TransLayer(nn.Module):
    def __init__(self, norm_layer: LayerType = nn.LayerNorm, dim: int = 512) -> None:
        super().__init__()
        self.norm = norm_layer(dim)  # type: ignore[operator]
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,  # number of landmarks
            pinv_iterations=6,
            # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual=True,
            # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1,
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x


class PPEG(nn.Module):
    def __init__(self, dim: int = 512) -> None:
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, H, W):  # noqa: N803
        B, _, C = x.shape  # noqa: N806
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class FeatureTransMIL(nn.Module):
    def __init__(self, input_dim: int = 1024, embed_dim: int = 512) -> None:
        """TransMIL model for histopathology data.
        Args:
            input_dim: Number of input features.
            embed_dim: Number of features embedding dimension.
        """
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        self.pos_layer = PPEG(dim=self.embed_dim)
        self._fc1 = nn.Sequential(nn.Linear(self.input_dim, self.embed_dim), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.layer1 = TransLayer(dim=self.embed_dim)
        self.layer2 = TransLayer(dim=self.embed_dim)
        self.norm = nn.LayerNorm(self.embed_dim)

    def forward(self, h):
        h = h.float()  # [B, n, 1024]

        h = self._fc1(h)  # [B, n, 512]

        # ---->pad
        H = h.shape[1]  # noqa: N806
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))  # noqa: N806
        add_length = _H * _W - H
        h = torch.cat([h, h[:, :add_length, :]], dim=1)  # [B, N, 512]

        # ---->cls_token
        B = h.shape[0]  # noqa: N806
        cls_tokens = self.cls_token.expand(B, -1, -1)
        h = torch.cat((cls_tokens, h), dim=1)

        # ---->Translayer x1
        h = self.layer1(h)  # [B, N, 512]

        # ---->PPEG
        h = self.pos_layer(h, _H, _W)  # [B, N, 512]

        # ---->Translayer x2
        h = self.layer2(h)  # [B, N, 512]

        # ---->cls_token
        h = self.norm(h)[:, 0]

        return h

class FeatureTransMILHybrid(FeatureTransMIL):
    def __init__(
        self,
        input_dim: int = 1024,
        embed_dim: int = 512,
        num_tokens: int = 2048,
        retention_decoder_depth: int = 1,
    ) -> None:
        """Pre-training TransMIL model for histopathology data.
        Args:
            input_dim: Number of input features.
            embed_dim: Number of features embedding dimension.
            num_tokens: Number of tokens.
            retention_decoder_depth: Depth of retention decoder.
        """
        super().__init__(input_dim, embed_dim)

        self.num_tokens = num_tokens

        # self.alignment_head = nn.Linear(embed_dim, embed_dim)

        self.retention_embed = nn.Linear(embed_dim, embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.retention_pos_embed = nn.Embedding(num_tokens + 1, embed_dim)

        self.retention_blocks = nn.ModuleList(
            [TransLayer(dim=embed_dim) for _ in range(retention_decoder_depth)]
        )
        self.retention_norm = nn.LayerNorm(embed_dim)
        self.retention_head = nn.Linear(embed_dim, embed_dim)

        self.init_weights()

    def init_weights(self) -> None:
        torch.nn.init.normal_(self.mask_token, std=0.02)
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.retention_gene_embed.weight, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(
        self, h: torch.Tensor, mask_ratio: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, C = h.shape  # noqa: N806
        len_keep = int(N * (1 - mask_ratio))

        noise = torch.rand(B, N, device=h.device)

        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        h_masked = torch.gather(h, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, C))
        mask_tokens = self.mask_token.repeat(
            B, ids_restore.shape[1] - h_masked.shape[1], 1
        )
        h_masked = torch.cat([h_masked, mask_tokens], dim=1)
        h_masked = torch.gather(
            h_masked, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, C)
        )

        mask = torch.ones([B, N], device=h.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return h_masked, mask

    def forward_encoder(self, h: torch.Tensor) -> torch.Tensor:
        h = h.float()  # [B, n, 1024]

        h = self._fc1(h)  # [B, n, 512]

        # ---->pad
        H = h.shape[1]  # noqa: N806
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))  # noqa: N806
        add_length = _H * _W - H
        h = torch.cat([h, h[:, :add_length, :]], dim=1)  # [B, N, 512]

        # ---->cls_token
        B = h.shape[0]  # noqa: N806
        cls_tokens = self.cls_token.expand(B, -1, -1)
        h = torch.cat((cls_tokens, h), dim=1)

        # ---->Translayer x1
        h = self.layer1(h)  # [B, N, 512]

        # ---->PPEG
        h = self.pos_layer(h, _H, _W)  # [B, N, 512]

        # ---->Translayer x2
        h = self.layer2(h)  # [B, N, 512]

        # ---->cls_token
        h = self.norm(h)

        return h[:, : h.shape[1] - add_length, :]

    # def forward_alignment_head(self, h: torch.Tensor) -> torch.Tensor:
    #     eps = 1e-6 if h.dtype == torch.float16 else 1e-12
    #     h = nn.functional.normalize(h, dim=-1, p=2, eps=eps)
    #     alignment_h = self.alignment_head(h[:, 0, :])
    #     return alignment_h  # type: ignore[no-any-return]

    def forward_retention_head(
        self, h: torch.Tensor, mask_ratio: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        retention_h = self.retention_embed(h)
        retention_h_, mask = self.random_masking(retention_h[:, 1:, :], mask_ratio)
        retention_h = torch.cat([retention_h[:, :1, :], retention_h_], dim=1)

        # Add positional embedding
        B, L, _ = retention_h.shape
        device = retention_h.device

        pos_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)  # [B, L]
        pos_embed = self.retention_pos_embed(pos_ids)  # [B, L, D]

        retention_h = retention_h + pos_embed


        for blk in self.retention_blocks:
            retention_h = blk(retention_h)
        retention_h = self.retention_norm(retention_h)
        retention_h = self.retention_head(retention_h)
        retention_h = retention_h[:, 1:, :]
        return retention_h, mask

    def forward_decoders(
        self, h: torch.Tensor, mask_ratio: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # alignment_h = self.forward_alignment_head(h)
        retention_h, mask = self.forward_retention_head(h, mask_ratio)
        return retention_h, mask

    def forward(
        self, h: torch.Tensor, mask_ratio: float = 0.75
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.forward_encoder(h)
        retention_h, mask = self.forward_decoders(h, mask_ratio)
        retention_target_h = h[:, 1:, :]
        return retention_h, retention_target_h, mask


# ===========================================
#  CMD for Pre-training
# ===========================================
class CMD(nn.Module):
    def __init__(
        self,
        wsi_embed_dim: int,
        embed_dim: int,
        wsi_num_tokens: int = 30000,
        wsi_retention_decoder_depth: int = 1,
        # init_logit_scale: float = np.log(1 / 0.07),
        num_classes: int = 4,
        # style_mlp_hidden_dim: int = 512,
        # style_mlp_out_dim: int = 256,
        # style_norm_layer: Optional[LayerType] = None,
        # style_act_layer: Optional[LayerType] = None,
        # style_latent_dim: int = 128,
        # num_prototypes: int = 3000,
    ) -> None:
        """CMD model for pre-training.
        Args:
            wsi_embed_dim: Number of WSI embedding features.
            embed_dim: Number of features embedding dimension.
            wsi_num_tokens: Number of WSI tokens.
            wsi_retention_decoder_depth: Depth of WSI retention decoder.
            num_classes: Number of classes for classification.
        """
        super().__init__()

        self.wsi_embed_dim = wsi_embed_dim
        self.embed_dim = embed_dim
        self.wsi_num_tokens = wsi_num_tokens
        self.wsi_retention_decoder_depth = wsi_retention_decoder_depth
        self.num_classes = num_classes

        # self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)

        self.he_encoder = FeatureTransMILHybrid(
            input_dim=self.wsi_embed_dim,
            embed_dim=self.embed_dim,
            num_tokens=self.wsi_num_tokens,
            retention_decoder_depth=self.wsi_retention_decoder_depth,
        )

        self.ihc_encoder = FeatureTransMILHybrid(
            input_dim=self.wsi_embed_dim,
            embed_dim=self.embed_dim,
            num_tokens=self.wsi_num_tokens,
            retention_decoder_depth=self.wsi_retention_decoder_depth,
        )

        self.he_classifier = nn.Linear(self.embed_dim, num_classes)
        self.ihc_classifier = nn.Linear(self.embed_dim, num_classes)


        # style_act_layer = get_act_layer(style_act_layer) or nn.GELU  # type: ignore[arg-type]

        # self.style_encoder_mlp = Mlp(
        #     in_features=embed_dim,
        #     hidden_features=style_mlp_hidden_dim,
        #     out_features=style_mlp_out_dim,
        #     act_layer=style_act_layer,
        #     norm_layer=style_norm_layer,
        #     drop=0.0,
        # )
        # self.style_mu = nn.Linear(style_mlp_out_dim, style_latent_dim)
        # self.style_logstd = nn.Linear(style_mlp_out_dim, style_latent_dim)
        # self.style_decoder = nn.Linear(style_latent_dim, embed_dim)

        # self.prototypes = nn.Linear(embed_dim, num_prototypes, bias=False)
        # nn.init.orthogonal_(self.prototypes.weight)

    # def reparameterize(self, mu: torch.Tensor, logstd: torch.Tensor) -> torch.Tensor:
    #     std = torch.exp(0.5 * logstd)
    #     dist = Normal(mu, std)
    #     return dist.rsample()

    # def forward_style_clustering(
    #     self, wsi_emb: torch.Tensor, rna_emb: torch.Tensor
    # ) -> Tuple[
    #     torch.Tensor,
    #     torch.Tensor,
    #     torch.Tensor,
    #     torch.Tensor,
    #     torch.Tensor,
    #     torch.Tensor,
    # ]:
    #     wsi_emb = self.style_encoder_mlp(wsi_emb)
    #     wsi_mu = self.style_mu(wsi_emb)
    #     wsi_logstd = self.style_logstd(wsi_emb)
    #     wsi_z = self.reparameterize(wsi_mu, wsi_logstd)
    #     wsi_z = self.style_decoder(wsi_z)
    #     wsi_score = self.prototypes(wsi_z)

    #     rna_emb = self.style_encoder_mlp(rna_emb)
    #     rna_mu = self.style_mu(rna_emb)
    #     rna_logstd = self.style_logstd(rna_emb)
    #     rna_z = self.reparameterize(rna_mu, rna_logstd)
    #     rna_z = self.style_decoder(rna_z)
    #     rna_score = self.prototypes(rna_z)
    #     return wsi_score, wsi_mu, wsi_logstd, rna_score, rna_mu, rna_logstd

    def forward(
        self,
        he_emb: torch.Tensor,
        ihc_emb: torch.Tensor,
        # rna_emb: torch.Tensor,
        wsi_mask_ratio: float = 0.75,
        # rna_mask_ratio: float = 0.75,
    ):
        he_emb = self.he_encoder.forward_encoder(he_emb)
        he_cls = he_emb[:, 0, :]
        he_retention_emb, he_mask = (
            self.he_encoder.forward_decoders(he_emb, mask_ratio=wsi_mask_ratio)
        )
        he_retention_target = he_emb[:, 1:, :]

        he_logits = self.he_classifier(he_cls) #[B, n_classes]
        he_results_dict = {'logits': he_logits, 'probs': F.softmax(he_logits, dim = 1), 'preds': torch.argmax(he_logits, dim=1)}

        ihc_emb = self.ihc_encoder.forward_encoder(ihc_emb)
        ihc_cls = ihc_emb[:, 0, :]
        ihc_retention_emb, ihc_mask = (
            self.ihc_encoder.forward_decoders(ihc_emb, mask_ratio=wsi_mask_ratio)
        )
        ihc_retention_target = ihc_emb[:, 1:, :]

        ihc_logits = self.ihc_classifier(ihc_cls) #[B, n_classes]
        ihc_results_dict = {'logits': ihc_logits, 'probs': F.softmax(ihc_logits, dim = 1), 'preds': torch.argmax(ihc_logits, dim=1)}

        # rna_emb = self.rna_encoder.forward_encoder(rna_emb)
        # rna_alignment_emb, rna_retention_emb, rna_mask = (
        #     self.rna_encoder.forward_decoders(rna_emb, mask_ratio=rna_mask_ratio)
        # )
        # rna_retention_target = rna_emb

        # wsi_score, wsi_mu, wsi_logstd, rna_score, rna_mu, rna_logstd = (
        #     self.forward_style_clustering(wsi_emb[:, 0, :], rna_emb)
        # )

        return (
            he_retention_emb,
            he_retention_target,
            he_mask,
            he_results_dict,
            ihc_retention_emb,
            ihc_retention_target,
            ihc_mask,
            ihc_results_dict,
            # wsi_score,
            # wsi_mu,
            # wsi_logstd,
            # rna_alignment_emb,
            # rna_retention_emb,
            # rna_retention_target,
            # rna_mask,
            # rna_score,
            # rna_mu,
            # rna_logstd,
            # self.logit_scale.exp(),
        )

    def foward_validate(self,
        he_emb: torch.Tensor):
        
        he_emb = self.he_encoder.forward_encoder(he_emb)
        he_cls = he_emb[:, 0, :]
        he_logits = self.he_classifier(he_cls) #[B, n_classes]
        he_results_dict = {'logits': he_logits, 'probs': F.softmax(he_logits, dim = 1), 'preds': torch.argmax(he_logits, dim=1)}
        
        return he_results_dict