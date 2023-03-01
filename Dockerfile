# TODO: replace FROM
FROM dleongsh/torchaudio:0.11.0
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

CMD ["python", "src/app.py"]
