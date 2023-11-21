# ESC Inference Service

Environment Sound Classification Inference Service for AI App Store

This uses the BEATS pretrained model developed by Microsoft. 

# Download weights
Download any of the finetuned model weights from Microsoft's official Github [here](https://github.com/microsoft/unilm/tree/master/beats). (You do not need the Tokenizer weights)

Create and place the weights in the `models/` folder, and rename the weights file to `AS2M_beats.pt`.

## Build
To build the docker container, run
```sh
make build
```

## Run Gradio App Locally

To run the Gradio application locally, run the following
```sh
make gradio
```

## Run Batch Inference Code Locally

If you are running over batches of audio data, it's best to run in the regular way outside of Gradio. 

1 - First edit the `docker-compose.yaml` file to correctly mount your dataset location. Then run:
```sh
docker-compose run --rm local bash
```

2 - For the audio files you want to run on, prepare a manifest file so that it contains the relative paths to those files. i.e. it should be a json file (`example.json`) in the following format:
```json
{"audio_filepath": "rel/path/to/audio0.wav"}
{"audio_filepath": "rel/path/to/audio1.wav"}
{"audio_filepath": "rel/path/to/audio2.wav"}
```
3 - Then edit the `local_config.py` file to your required configurations.

4 - Lastly run the inference script inside the container.
```sh
python3 src/local_infer_batch.py
```


## Deployment on AI App Store

### 1. Push to Registry
To push the image to a registry, first build the image, then run
```sh
docker tag esc-inference-service:1.0.0 <REGISTRY>/<REPO>/esc-inference-service:1.0.0
```

If not logged in to the registry, run
```sh
docker login -u <USERNAME> -p <PASSWORD> <REGISTRY>
```

Then, push the tagged image to a registry
```sh
docker push <REGISTRY>/<REPO>/esc-inference-service:1.0.0
```

### 2. Deployment on AI App Store
Check out the AI App Store documentation for full details, but in general:
1. Create/edit a model card
2. Pass the docker image URI (e.g `<REGISTRY>/<REPO>/esc-inference-service:1.0.0`) when creating/editing the inference service
