# Spam Detection

Esse repositório contém os artefatos gerados para o desafio de classificação de emails em spam ou não spam.

## Descrição da abordagem aplicada

* Foi utilizado a linguagem Python para criação dos algoritmos. 
* Foram utilizadas diversas bibliotecas para tratamento do texto (`nltk`), treinamento de algoritmos de classificação (`Scikit-learn`) e plotagem dos dados (`Matplotlib` e `seaborn`) entre outras.

Para solucionar o desafio foram utilizados dois modelos de classificação `Logistic Regression` e `RandomForest`.

## Passo a Passo

1 - Primeiro passo foi o tratamento dos textos:

* Removendo as stopwords presentes e palavras com tamanho menor ou igual a 2;
* Deixando todas as palavras em minúsculo;
* Removendo termos que possuem dígitos;
* Limpeza de caracteres especiais e dígitos;
* Stemização dos dados para remoção de inflexões;
  
2 - O segundo passo foi converter os dados em texto para valores numéricos, para que os modelos de classificação possam ser aplicados. Foi utilizado a técnicas TF-IDF que converte cada texto em um vetor que indica se determinada palavra está presente no texto(1) ou não(0). 

3 - O terceiro passo foi transformar o label (ham e spam) em categorias numéricas (0 e 1). Essa etapa foi necessária pois os modelos não conseguem lidar com variáveis não numéricas.

4 - O quarto passo foi dividir os dados em treino e teste para treinamento e avaliação do modelo. Essa etapa foi necessária para que possamos avaliar a qualidade do modelo em dados ainda não vistos pelo modelo.

5 - Como foi identificado que os dados possuem desbalancementamento, ou seja, existe muito mais exemplo da classe ham do que da classe spam, o quinto passo foi aplicar uma técnica de `OverSampling` para criação de novos exemplos da classe minoritária. Para tal, usamos a técnica de SMOTE para produção de novos dados da classe minoritária.

6 - O sexto passo foi treinar os modelos de classificação nos dados gerados nas etapas anteriores. Para tal, usamos técnicas de validação cruzada para evitar overfitting e técnicas de GridSearch para busca dos melhores parâmetros para cada modelo treinado.

7 - Sétimo e último passo foi a avaliação dos modelos. Ambos os modelos tiveram ótimos resultados: Logistic Regression obteve uma taxa de acerto de **0.967**, enquanto o modelo de RandomForest obteve uma leve melhora apresentando resultados de **0.982**. Por fim aplicamos o melhor modelo (RandomForest) e classificamos os dados presente no diretório de TestSet, disponibilizado pela nuveo. A partir daí, criamos um arquivo csv com a classificação do modelo (arquivo `results.csv` presente no diretório `Results`).


## Execução

OBS: Não é necessário rodar o projeto, visto que todos os scrips já foram executados e os resultados salvos na diretório `Results`.

1 - Para executar o projeto basta apenas dar build na imagem e executar a imagem.

```
docker build -t python-opencv-img .
```

depois rode:

```
sudo docker run python-opencv-img
```

## OBSERVAÇÕES:

* Link para o colab com os scripts executados: https://colab.research.google.com/drive/1FVli5JD0b7_sPImnXzR_3dPcHcnJ7l91?usp=sharing