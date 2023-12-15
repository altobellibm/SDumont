# Exemplo para teste na máquina do santo dumont 

Este repositório contém exemplo usado durante a explicação no ppt executando jobs no Santos Dumont.

## Como Rodar

Para executar a interface, siga os passos abaixo, :

1. Clone o repositório na sua máquina local:

```bash
git clone https://github.com/altobellibm/SDumont.git
```

2. Criar environment:
```bash
conda create --name SD python=3.10
conda activate SD
pip install -r requirements.txt
```

3. Baixar e criar pasta com as imagens da base de dados MNIST:

```bash
python load_dataset.py
```

4. Treinamento do modelo:

```bash
python train.py
```

5. Testar modelo:

```bash
python test.py
```

## Licença

Este projeto está licenciado sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

---
