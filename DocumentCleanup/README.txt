	Neste desafio optei por usar uma abordagem de classificação pixel a pixel, visto que o objetivo é classificar background/ruído de texto. Não optei optei pela utilização de filtros pois quis minimizar ao máximo a intervenção humana, pois teria que estipular os parametros e thresholds manualmente.

	A abordagem consiste na extração de atributos das vizinhas de cada pixel, visto que vizinhanças de background/ruído possuem atributos com padrões diferentes aos de vizinhanças de texto. Um dos pontos cruciais desta abordagem é a classificação das imagens que serão usadas como treinamento, visto que as mesmas já devem estar classificadas para servirem como base do método. Optei por binarizar 4 imagens (uma de cada padrão de ruído) manualmente. Em algumas um simples threshold já foi o suficiente, em outras tive que utilizar de ferramentas como pincel e borracha para contornar os labels de forma eficiente. Nesta etapa utilizei a ferramenta 3d slicer, na qual utilizo esporadicamente em meu tcc e doutorado, foi um processo relativamente rápido.

	Etapas do pipeline:

	1) (Arquivo 01_training_feature_extractor.ipynb) Após a classificação manual, tenho 4 imagens que poderão ser usadas como dataset de treino. O pipeline se inicia com a extração dos atributos das vizinhanças de cada pixel (mais detalhes no código), além disso consigo definir a classe de cada um deles com base nas imagens que foram binarizadas anteriormente. Este resultado será usado para treinar uma rede neural (etapa 3). Esta etapa tem como resultado um csv salvo em disco contendo o dataset de treino, onde as linhas representam cada pixel e as colunas os atributos e a classe. (Descompactar arquivo)
    
    *** Decidi manter esta etapa em um código separado pois esta estapa só ocorre uma vez, dessa forma não é preciso executala toda vez que o pipeline é requisitado ***

	2) (Arquivo 02_pipeline.ipynb) Neste arquivo o restante do pipeline é executado. Para cada imagem a se binarizada.
    
    - Primeiramente são extraídos os mesmos atributos da etapa anterior (menos a classe de cada pixel). 
    - Em seguida é feito o treinamento da rede (MultiLayerPerceptron - MLP) com base no dataset de treino (eu deixei o código de forma que carregue o modelo e não precise treinar novamente)
    - Na sequência todos os pixels da imagem são classificados
    - O próximo passo é montar a imagem resultande
    - E por fim é feita a rotação da imagem
    
    *** Logo a após a etapa de rotação, percebi que algumas imagens continham alguns ruidos de granulação, inclusive alguns deles foram acentuados pela rotação, infelizmente não tive tempo de tratar esse problema ***
    
    

Considerações:

    Decidi usar rede neural pois como a classificação é pixel a pixel, isso gera uma grande quantidade de instâncias a serem treinadas e classificadas, o que é ideal para redes neurais. 
    
    Por se tratar de uma abordagem pixel a pixel, o tempo de processamento é relativamente alto. Após o treino da rede, cada imagem demora cerca de um minuto para ser binarizada
