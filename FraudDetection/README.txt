Em relação a análise:

    Como não dispunha de um dataset para validação, decidi utilizar todo o dataset oferecido como treino/teste e fazer uma validação cruzada afim de mensurar seu desempenho.
    
Em relação ao método:

    Gostaria de mencionar que este foi um desafio completamente novo para mim e que nunca havia trabalhado com este tipo de problema. Achei bastante interessante e desafiador.

    A abordagem que eu tracei envolve a detecção de estruturas da assinatura através do método de detecção de borda e convex hull, extração de atributos destas estruturas e a classificação utilizando tais atributos como base. (maiores detalhe no código)
    
    O classificador utilizado foi o SVM visto que o numero de imagens era relativamente pequeno para utilizar redes neurais. Com o SVM obtive uma acurácia de ~~ 90% na validação cruzada.
    
    Toda parte de extração foi feita no código contido no arquvio PipiLine.ipynb
    
    Toda parte de classificação foi feita no código contido no arquvio SVM.ipynb
    
