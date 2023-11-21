"""
This file contains code to launch the gradio app.
"""

import logging

import gradio as gr
from gr_config import BaseConfig
from gr_predict import inputs, outputs, predict

if __name__ == "__main__":
    logging.basicConfig(format="[%(asctime)s] %(levelname)s: %(message)s")
    config = BaseConfig()

    app = gr.Interface(
        predict,
        inputs=inputs,
        outputs=outputs,
        title="ESC Inference Service",
        description="ESC Inference Service for AI App Store",
        examples=config.example_dir,
    )

    app.launch(server_name="0.0.0.0", server_port=config.port, enable_queue=True)
