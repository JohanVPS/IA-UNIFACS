# Projeto de CNN para Classificação do CIFAR-10
*(Competição Acadêmica - UNIFACS)*  

![Python](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue) 
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white) 
![Google Colab](https://img.shields.io/badge/Google_Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white) 
![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)

## 📌 Objetivo
Desenvolver uma Rede Neural Convolucional (CNN) com o melhor desempenho no dataset CIFAR-10, maximizando a métrica:
 $$ Média de Acurácia =\frac{(Accuracy + LastAccuracy)}{2}$$

## ⚙️ Configuração do Ambiente
- **Plataforma obrigatória**: [Google Colab](https://colab.research.google.com/) (com GPU)
- **Tempo estimado**: 
  - CPU: ~2 horas
  - GPU: ~15 minutos

## 📋 Regras da Competição
✅ **Permitido**:
- Alterar épocas, camadas, topologia, dropout, data augmentation, funções de ativação
- Usar técnicas de otimização (BatchNorm, Early Stopping, etc.)

❌ **Proibido**:
- Reduzir o tamanho do dataset
- Usar editores fora do Google Colab

## 📤 Entrega
1. **Parcias**: Print da última época no grupo da UC
2. **Final**: Link do repositório GitHub para [Prof. Noberto](https://github.com/nobertomaciel)  
⚠️ A acurácia final deve ser ≥ à última parcial!

## 📂 Código Base
- [cnn.ipynb](https://github.com/nobertomaciel/AI-UNIFACS/blob/main/CNN/cnn.ipynb)

## 🛠️ Como Contribuir?
1. Faça um **fork** deste repositório
2. Crie uma branch: `git checkout -b minha-melhoria`
3. Envie um **Pull Request** com suas alterações

**Dica**: Use **Data Augmentation** e **Dropout** para evitar overfitting!  
**Boa sorte a todas as equipes!** 🚀