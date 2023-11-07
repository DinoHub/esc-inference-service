build:
	DOCKER_BUILDKIT=1 docker build -t dleongsh/esc-service:v1.0.0 .
dev:
	docker run -p 8080:8080 --rm -it --gpus all -v ${PWD}:/workspace dleongsh/esc-service:v1.0.0
