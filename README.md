# APS - Implementação de Redes Convolucionais

Atividade Prática Supervisionada da disciplina de Introdução à Aprendizagem Profunda.

## Objetivo

Implementar uma CNN customizada (LightCNN) em PyTorch e comparar seu desempenho com uma rede pré-treinada (ResNet18) nos datasets MNIST e CIFAR-10.

## Estrutura do Projeto

| Arquivo | Descrição |
|---|---|
| `models.py` | Definição da arquitetura LightCNN |
| `utils.py` | Funções auxiliares de treino, avaliação e visualização |
| `cnn_mnist.ipynb` | Experimentos no dataset MNIST |
| `cnn_cifar10.ipynb` | Experimentos no dataset CIFAR-10 |

## Instalação

```bash
pip install -r requirements.txt
```

## Execução

Abra os notebooks Jupyter e execute as células sequencialmente:

```bash
jupyter notebook cnn_mnist.ipynb
jupyter notebook cnn_cifar10.ipynb
```

## Arquitetura da LightCNN

```
Conv2d -> BatchNorm -> ReLU -> MaxPool  (32 filtros)
Conv2d -> BatchNorm -> ReLU -> MaxPool  (64 filtros)
Conv2d -> BatchNorm -> ReLU -> MaxPool  (128 filtros)
Flatten -> Linear(256) -> ReLU -> Dropout(0.5) -> Linear(10)
```

## Rede Pré-treinada

ResNet18 com fine-tuning (camadas iniciais congeladas, classificador adaptado para 10 classes).
