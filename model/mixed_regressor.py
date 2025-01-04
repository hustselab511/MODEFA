import torch
import torch.nn as nn


class MixedRegressor(nn.Module):
    def __init__(self, dist_input_dim, pos_input_dim, con_input_dim,
                 dist_hidden_dim, pos_hidden_dim, con_hidden_dim,
                 combined_hidden_dim, output_dim):
        super(MixedRegressor, self).__init__()

        self.dist_mlp = nn.Sequential(
            nn.Linear(dist_input_dim, dist_hidden_dim),
            nn.ReLU(),
            nn.Linear(dist_hidden_dim, dist_hidden_dim),
            nn.ReLU(),
            nn.Linear(dist_hidden_dim, dist_hidden_dim),
            nn.Dropout(0.1)
        )

        self.con_mlp = nn.Sequential(
            nn.Linear(con_input_dim, con_hidden_dim),
            nn.ReLU(),
            nn.Linear(con_hidden_dim, con_hidden_dim),
            nn.ReLU(),
            nn.Linear(con_hidden_dim, con_hidden_dim),
            nn.Dropout(0.1)
        )

        self.pos_mlp = nn.Sequential(
            nn.Linear(pos_input_dim, pos_hidden_dim),
            nn.ReLU(),
            nn.Linear(pos_hidden_dim, pos_hidden_dim),
            nn.ReLU(),
            nn.Linear(pos_hidden_dim, pos_hidden_dim),
            nn.Dropout(0.1)
        )

        self.combined_mlp = nn.Sequential(
            nn.Linear(dist_hidden_dim + pos_hidden_dim, combined_hidden_dim),
            nn.ReLU(),
            nn.Linear(combined_hidden_dim, combined_hidden_dim),
            nn.ReLU(),
            nn.Linear(combined_hidden_dim, combined_hidden_dim),
            nn.ReLU(),
            nn.Linear(combined_hidden_dim, combined_hidden_dim),
            nn.ReLU(),
            nn.Linear(combined_hidden_dim, output_dim)
        )

    def forward(self, dist_vector, pos_vector, con_vector):
        dist_feature = self.dist_mlp(dist_vector)
        con_feature = self.con_mlp(con_vector)

        combined_dist_con_feature = dist_feature + 0.001 * con_feature

        pos_feature = self.pos_mlp(pos_vector)

        combined_feature = torch.cat((combined_dist_con_feature, pos_feature), dim=-1)

        output = self.combined_mlp(combined_feature)
        return output


