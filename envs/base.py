from typing import TypeVar, Generic, Dict, Any, Tuple, Union, List, TypedDict
import numpy as np

ObsType = TypeVar('ObsType')
ActionType = TypeVar('ActionType')

class EnvOutput(TypedDict):
    pixel_values: List[np.ndarray]
    prompts: List[str]
