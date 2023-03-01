build:
	docker build . -t gradio-esc-inference-service:1.0.0
dev:
	docker run -p 8080:8080 --rm -it -v ${PWD}:/workspace $ gradio-esc-inference-service:1.0.0
