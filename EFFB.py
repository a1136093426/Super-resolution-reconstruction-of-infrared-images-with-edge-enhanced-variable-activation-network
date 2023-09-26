import torch
from torch import nn as nn

"""Edge feature fusion block (EFFB)"""
class EFFB(nn.Module):
    def __init__(self, num_feat=64):
        super(EFFB, self).__init__()

        self.conv1 = nn.Conv2d(1, num_feat, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat*2, num_feat, 1, 1)

    def forward(self, edge_map, feature):
        # edge_map: edge maps
        # feature: high-frequency features
        edge_fea = self.conv1(edge_map)
        fuse_edge_fea = edge_fea + feature
        cat_fea = torch.cat([fuse_edge_fea, edge_fea], dim=1)
        output_fea = self.conv2(cat_fea)

        return output_fea


