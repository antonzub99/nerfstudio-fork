"""
Depth-Supervsied Dataset.
"""
from typing import Dict

import torch

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs, Semantics
from nerfstudio.data.datasets.base_dataset import InputDataset


class DSInputDataset(InputDataset):
    """Dataset that returns images and sparse keypoints depths.

    Args:
        dataparser_outputs: description of where and how to read input images.
    """

    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0):
        super().__init__(dataparser_outputs, scale_factor)
        assert "depth_data" in dataparser_outputs.metadata.keys() and isinstance(self.metadata["depth_data"], list)
        #assert "semantics" in dataparser_outputs.metadata.keys() and isinstance(self.metadata["semantics"], Semantics)
        self.depth_data = self.metadata["depth_data"]
        
    def get_metadata(self, data: Dict) -> Dict:
        # handle mask
        img_depth_data = self.depth_data[data["image_idx"]]
        return {key: torch.tensor(val) for key, val in img_depth_data.items()}
