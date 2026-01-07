.PHONY: help install train run clean

help:
	@echo "Comandos disponíveis:"
	@echo "  make install  - Instala as dependências"
	@echo "  make train    - Treina o modelo e gera os artefatos"
	@echo "  make run      - Executa a aplicação Streamlit"
	@echo "  make clean    - Remove arquivos gerados (modelo e métricas)"

install:
	pip install -r requirements.txt

train:
	python train_model.py

run:
	streamlit run app.py

clean:
	rm -f model.joblib metrics.json
