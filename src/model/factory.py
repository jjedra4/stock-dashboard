from .interface import BaseModel
from .algorithms import SklearnModel

class ModelFactory:
    """
    Factory class to create model instances.
    """
    @staticmethod
    def get_model(model_type: str, **hyperparameters) -> BaseModel:
        """
        Returns a model instance based on the model_type string.
        """
        # Mapping simple names to specific implementations
        sklearn_types = ["random_forest", "gradient_boosting", "linear"]
        
        if model_type in sklearn_types:
            return SklearnModel(model_type=model_type, **hyperparameters)
        
        # Future:
        # if model_type == "lstm":
        #     return LSTMModel(**hyperparameters)
            
        raise ValueError(f"Unknown model type: {model_type}")

