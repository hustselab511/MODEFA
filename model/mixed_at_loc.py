import torch
import torch.nn as nn


class MultiScaleSpacialGating(nn.Module):
    def __init__(self, channel_dim, ob_size, hidden_dim):
        super(MultiScaleSpacialGating, self).__init__()

        self.query_proj = nn.Linear(channel_dim * ob_size, hidden_dim)
        self.key_proj = nn.Linear(channel_dim * ob_size, hidden_dim)
        self.value_proj = nn.Linear(channel_dim * ob_size, hidden_dim)

        self.softmax_at_score = nn.Softmax(dim=-1)

        self.at_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.gating_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
        )

        self.softmax_gating = nn.Softmax(dim=-1)

        self.feat_proj = nn.Linear(hidden_dim, channel_dim * ob_size)


    def forward(self, feat_input, patch_feats_list):
        at_list = []

        Q = self.query_proj(feat_input.view(feat_input.size(0), -1))

        for patch_feats in patch_feats_list:
            K = self.key_proj(patch_feats.view(patch_feats.size(0), -1))
            V = self.value_proj(patch_feats.view(patch_feats.size(0), -1))

            at_score = torch.matmul(Q, K.t())
            at_score = at_score / (K.size(-1) ** 0.5)
            at_score = self.softmax_at_score(at_score)

            at = torch.matmul(at_score, V)

            at = self.at_mlp(at) + at

            at_list.append(at)

        combined_at = torch.stack(at_list, dim=1)

        weights = self.gating_mlp(combined_at)
        weights = weights.squeeze(-1)
        weights = self.softmax_gating(weights)

        gated_at = torch.sum(weights.unsqueeze(-1) * combined_at, dim=1)

        gated_feat = self.feat_proj(gated_at)
        gated_feat = gated_feat.view(feat_input.size(0), feat_input.size(1), -1)

        return gated_feat


class MixedATLoc(nn.Module):
    def __init__(self, ob_size, channel_dim, hidden_dim, gating_hidden_dim, output_dim):
        super(MixedATLoc, self).__init__()

        self.multi_scale_spacial_gating = MultiScaleSpacialGating(channel_dim, ob_size, gating_hidden_dim)

        self.mlp_final = nn.Sequential(
            nn.Linear(ob_size * channel_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, feat_input, patch_feats_list):
        gated_feat = self.multi_scale_spacial_gating(feat_input, patch_feats_list)

        processed_feat = 0.01 * gated_feat + feat_input

        feat_flatten = processed_feat.view(processed_feat.size(0), -1)
        pos_output = self.mlp_final(feat_flatten)

        return pos_output