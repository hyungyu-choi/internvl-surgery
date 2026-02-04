from .base_dataset import SurgicalPhaseDataset
from .cholec_dataset import Cholec80Dataset
from .autolaparo_dataset import AutoLaparoDataset
from . cholec_ranking_dataset import CholecRankingDataset

__all__ = [
    'SurgicalPhaseDataset',
    'Cholec80Dataset',
    'AutoLaparoDataset'
]


def get_dataset(dataset_name, **kwargs):
    """
    Factory function to get dataset by name
    
    Args:
        dataset_name: 'cholec' or 'autolaparo'
        **kwargs: Arguments passed to dataset constructor
    
    Returns:
        Dataset instance
    """
    dataset_map = {
        'cholec': Cholec80Dataset,
        'autolaparo': AutoLaparoDataset
    }
    
    if dataset_name not in dataset_map:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(dataset_map.keys())}")
    
    return dataset_map[dataset_name](**kwargs)
