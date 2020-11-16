O algoritmo foi desenvolvido com Python 3.7.9, tensorflow 2.3.0 e OpenCV 3.4.2
A pasta TrainingSet possui os dados usados para treino
A pasta TestSet possui os dados usados para teste
A pasta pretreinedmodels possui o arquivo gerado pelo treinamento da rede a ser usado nos testes
A pasta results possui um arquivo csv que ilustra os resultados obtidos, mais especificamente nome da imagens e cordenadas das centroides.
Além disso a pasta results possui uma sub pasta chamada segmentedmasks a qual ilustra em imagens as areas segmentadas pela rede para a geração das centroides presentes no arquivo csv.

O arquivo com codigos chama-se main.ipynb, nele se encontra todo o processo usado para resolução do teste.
Caso preferir testar, Aconselha-se o uso apenas da ultima celula deste notebook por estar configurada para testes com o modelo pre-treinado.