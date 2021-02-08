#  SMS Spam Detection

A solução deste desafio consiste no treinamento de um modelo preditivo para detectar mensagens que são spam ou não.  

Para chegar neste resultado foi primeiramente realizado a separação dos dados em conjuntos de treino e teste, respectivamente 80% e 20% dos dados, e foi aplicado um vetorizador TF-IDF para selecionar apenas as palavras que de fato trazem relevância para o modelo.  

Foi utilizado um SVC para realizar a classificação das mensanges utilizadno os dados extraídos do vetorizador. O modelo treinado atinge em media um valor de acurácia de 98%.  

Ao final é realizado a predição em novas mensagens nunca vistas pelo modelo anteriormente. Essa predição exportada para um arquivo chamado output.csv que se encontra na pasta denominada output.