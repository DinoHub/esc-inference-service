"""
This file contains all of the configurations to run the gradio app.
"""
from pydantic import BaseSettings, Field

class BaseConfig(BaseSettings):
    """Define any config here.

    See here for documentation:
    https://pydantic-docs.helpmanual.io/usage/settings/
    """
    # KNative assigns a $PORT environment variable to the container
    port: int = Field(default=8080, env="PORT",description="Gradio App Server Port")
    esc_model_path: str = 'models/AS2M_beats.pt'
    labels_path: str = 'misc/class_labels_indices.csv'
    topk: int = 10

    example_dir: str = 'examples'


config = BaseConfig()
