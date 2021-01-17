# Spam Detection
 
Esse repositório contém os artefatos gerados para o desafio de classificação de emails em spam ou não spam.
 
## Descrição da abordagem aplicada
 
* Foi utilizado a linguagem Python para criação dos algoritmos.
* Foi utilizada a biblioteca Opencv para desenvolvimento da solução proposta.
 
Para solucionar o desafio foi utilizado o modelo Haar Cascades para identificação de objetos. Os detectores baseados em cascade (cascata) são chamados assim pois treinam uma árvore de decisão em que cada nível analisa um conjunto de atributos diferentes e avalia se esses atributos representam ou não o objeto de interesse. Esse modelo pode funcionar muito bem, especialmente se você estiver apenas procurando por um objeto específico.
 
Quando você quer construir um modelo de  Haar cascades, você precisa de imagens "positivas" e imagens "negativas". As imagens "positivas" são imagens que contêm o objeto que você deseja encontrar. Podem ser imagens que contêm principalmente o objeto ou podem ser imagens que contêm o objeto, e você especifica a ROI (região de interesse) onde o objeto está. Uma coisa boa sobre o modelo é que só é necessário uma imagem positiva do objeto que deseja detectar e, alguns milhares de imagens negativas. As imagens negativas podem ser qualquer coisa, exceto que não podem conter seu objeto.
 
 
A partir daqui, com sua única imagem positiva, você pode usar o comando **opencv_createsamples** para realmente criar um monte de exemplos positivos, usando suas imagens negativas. Sua imagem positiva será sobreposta a esses negativos e será inclinada, entre outras coisas.
 
## Passo a Passo da resolução do problema
 
1 - O primeiro passo foi produzir as imagens negativas. Para isso o procedimento realizado foi percorrer as imagens de treino e remover o objeto wally de cada uma atráves das suas coordenadas do polígono (disponibilizados no arquivo .json de cada imagem). O resultado desse processo foi salvo na pasta `neg`.
 
2 - O segundo passo a partir daqui seria com a imagem do wally, executar o comando opencv_createsamples para realmente criar um monte de exemplos positivos, usando suas imagens negativas. Porém, como já foram disponibilizados imagens positivas com o personagem wally e o polígono que representa seu ROI, iremos utilizar essas imagens para treinar o modelo. Para tal, precisamos converter as posições do polígono em um retângulo que englobe esse polígono, pois o modelo faz uso de retângulos para o ROI dos objetos na imagem. Para converter as coordenadas do polígono nas coordenadas do retângulo usamos um cálculo simples, pegamos o ponto com menor valor e x e o com menor valor de y pra serem o x e y do retângulo e pegamos a diferença entre o menor valor de x e o maior para ser o valor de width da imagem e a diferença do menor valor de y e o maior para ser o valor de height do retângulo.
 
Ex:
```
Coordenadas do polígono: [[682, 467], [517, 466], [432, 683], [591, 683]] onde os o primeiro valor de cada array representa o `x` e o segundo o `y`.
Cálculo:
* x_values = [i[0] for i in pts]
* y_values = [i[1] for i in pts]
* x = min(x_values)
* y = min(y_values)
* w = max(x_values) - min(x_values)
* h = max(y_values) - min(y_values)
```
Com isso os valores que eram [[682, 467], [517, 466], [432, 683], [591, 683]] se tornam 432 466 250 217.
 
A partir daí criamos o arquivo que contém as imagens e as coordenadas do retângulo.
 
Caminho da imagem          n x1  y1   w   h
NewtrainingSet\image_1.jpg 1 432 466 250 217
 
3 - O terceiro passo criando o arquivo das imagens negativas, esse passo é simples basta percorrer o nome das imagens no diretório `neg` e a partir daí criar um arquivo txt com o caminho para cada imagem negativa.
 
Ex:
 
./neg/wally_001.jpg
./neg/wally_002.jpg
./neg/wally_003.jpg
...
./neg/wally_100.jpg
 
4 - O quarto passo, como a quantidade de imagens de treino são poucas (apenas 100, o indicado é ter ao menos 2500), vamos produzir mais algumas imagens positivas, apenas sobrepondo a imagem do wally nas imagens negativas. Para isso utilizamos o comando **opencv_createsamples**.
 
```
opencv_createsamples -img ./ReferenceData/img_wally_gray.jpg -bg ./bg.txt -info ./NewTrainingSet/info1.txt -pngoutput info -maxxangle 0.5 -maxyangle 0.5 -maxzangle 0.5 -num 200
```
 
Parâmetros:
 
* **img:** é o parâmetro para o caminho da imagem do objeto a ser identificado;
* **bg:** é o parâmetro para o arquivo que contém o caminho para todas as imagens negativas;
* **info:** é o parâmetro para o arquivo que vai conter o caminho para todas as imagens positivas criadas;
* **maxxangle, maxyangle, maxzangle:** são parâmetros para rotação da imagem do objeto nas imagens negativas;
* **num:** parâmetro que define o número de exemplos novos a serem criados;
 
5 - O quinto passo é juntar as imagens positivas que foram criadas com as que foram disponibilizadas para a resolução do problema. Para isso basta apenas juntar os arquivos de info (arquivo que contém o caminho para cada imagem positiva).
 
6 - Agora que temos imagens positivas, precisamos criar o arquivo vetorial, que é basicamente onde juntamos todas as nossas imagens positivas. Na verdade, usaremos opencv_createsamples novamente para isso!
 
```
opencv_createsamples -info ./NewTrainingSet/info.txt -num 226 -w 24 -h 24 -vec ./positives.vec
```
 
Parâmetros:
 
* **info:** parâmetro que define o caminho para todas as imagens positivas;
* **-num:** parâmetro que define o número de exemplos que estarão no arquivo vetorial;
* **w, h:** aconselha-se definir wxh, pelo menos, 20x20. É altamente importante que a proporção entre -w e -h seja a mesma (ou bastante parecida)
* **vec:** parâmetro para o caminho do arquivo vetorial produzido.
 
7 - Treinamento do modelo. Aqui, dizemos para onde queremos que os dados vão, onde está o arquivo vetorial (imagens positivas), onde está o arquivo de fundo (imagens negativas), quantas imagens positivas e negativas usar, quantos estágios e a largura e altura. Observe que usamos significativamente menos numPos do que temos. Isso é para abrir espaço para os estágios, que vão se somar a isso.
 
```
opencv_traincascade -data ./data -vec ./positives.vec -bg ./bg.txt  -numPos 200 -numNeg 100 -numStages 10 -w 24 -h 24  -maxFalseAlarmRate 0.001 -minHitRate: 0.999
```
 
Parâmetros:
 
* **data:** parâmetro que indica onde ficarão os arquivos de treino do modelo (estágios e classificador);
* **vec:** localização do arquivo vetorial;
* **bg:** localização do arquivo que contém o caminho para as imagens negativas;
* **numPos:** número de exemplos positivos para treinamento;
* **numNeg:** número de exemplos negativos para treinamento;
* **numStages:** número de estágios de treinamento;
* **maxFalseAlarmRate:** é usado para definir quantos recursos precisam ser adicionados. Na verdade, queremos que cada classificador fraco tenha uma taxa de acerto muito boa nos positivos e, em seguida, permita que eles removam as janelas negativas o mais rápido possível, mas fazendo melhor do que a adivinhação aleatória;
* **minHitRate:** é o parâmetro que nos garante que nossos dados de treinamento positivos produzam pelo menos uma saída de detecção decente;
 
8 - Após o treinamento o modelo, rodamos o modelo nos dados de teste. Para isto criamos um script `WIW.py` mas podemos também rodar o passo a passo no notebook `wally.ipynb`.
 
9 - Após executado  o modelo nos dados de teste, salvamos o resultado no diretório `Results` que contém os resultados do centróide de cada objeto identificado e as imagens com os retângulos identificados.


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