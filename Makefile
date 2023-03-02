build:
	docker build . -t gradio-esc-inference-service:1.0.0
dev:
	docker run -p 8081:8081 --rm -it -v ${PWD}:/workspace $ gradio-esc-inference-service:1.0.0
