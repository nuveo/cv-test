# WheresWally Solution
---

O projeto foi desenvolvido na linguagem [Python](https://www.python.org/downloads/release/python-370/). O script é capaz de treinar um modelo conhecido como [Mask R-CNN](https://arxiv.org/pdf/1703.06870.pdf) que seja capaz de identificar o Wally nas imagens. Foi utilizado também o [Google Colaboratory](https://colab.research.google.com/) para programação do *script*, uma vez que a plataforma disponibiliza o uso de GPU e consequentemente facilita tanto na hora da implementação quanto no teste dos avaliadores.

# Descrição da solução

Esta solução utliza o [Detectron2](https://detectron2.readthedocs.io/en/latest/) (que por debaixo dos panos utiliza PyTorch), para treinar uma arquitetura de rede conhecida como Mask R-CNN. Esta arquitetura é uma extessão da Fast R-CNN, pois, além de retornar um *bounding box* de onde o objeto está localizado e a classe, este modelo também possui como saída a máscara da detecção. Ambas arquiteturas utilizam *Instance Segmentation*, que é o processo de segmentação que obtém a classe específica do objeto detectado.

### Tratamento dos Dados

Em *Deep Learning*, um importante passo na implementação de soluções é garantir que o seu conjunto de dados seja confiável. A qualidade do conjunto de treinamento implica diretamente na qualidade da solução. Portanto, o primeiro é analisar a base de treinamento e validar se, principalmente, as anotações estavam corretas. Utilizamos o [Labelme](https://github.com/wkentaro/labelme) para validar cada imagem, e como a especificação do projeto já havia alertado para possíveis não conformidades nos dados, de fato, as anotações das seguintes imagens precisaram ser corrigidas: `wally_007.jpg`, `wally_008.jpg`, `wally_045.jpg`, `wally_068.jpg`, `wally_077.jpg`, `wally_089.jpg` e `wally_100.jpg`.

### Treinamento do Modelo

Optamos por um modelo básico da mask r-cnn que utiliza uma resnet50 como arquitetura de extração de atributos. Partimos de um modelo pré-treinado com a base de dados do [COCO](https://cocodataset.org/#home), colocando em prática o conceito de *Transfer learning*. Na literatura, ao treinar inicialmente um modelo, buscamos atribuir parâmetros básicos e estabelecer um número de iterações que leve o modelo ao *overfitting*, ou seja, nosso modelo fica extremamente especialista no conjunto de treinamento. Ao analisar o treinamento e função de perda ao longo das iterações, podemos observar que a partir da iteração 800 nosso modelo não tem alteração substancial da sua *loss*, portanto, foi 800 o número de iterações suficiente que encontramos para um bom treinamento. A taxa inicial de aprendizagem foi ajustada para um valor não muito alto, pois já estamos aproveitando parte dos filtros treinados por outra base. Obviamente, o processo de definição dos melhores parâmetros pode ser extramente custoso, pois podemos ir ajustando cada parâmetro e fazendo um comparativo entre os modelos obtidos com base em uma métrica específica. Como nossa base é relativamente simples, não nos preocupamos em fazer um processo com muitos treinamentos e análises dos melhores hiperparâmetros, muito menos de técnicas de aumentos, como alteração de brilho, contraste, rotação, translação, dct, mixup, ruídos, etc. Mas todas essas técnicas são válidas para obtenção de modelos mais robustos

### Resultados

Ao final do treinamento, a *loss* estava muito baixa, o que mostra incialmente que o modelo foi bem ajustado para o conjunto de dados. 

    total_loss: 0.05807  loss_cls: 0.01024  loss_box_reg: 0.02192  loss_mask: 0.02395  loss_rpn_cls: 1.422e-05  loss_rpn_loc: 0.00182

A *loss* de classificação, do *bounding box* e da máscara estão bem baixas.

Como o problema busca obter os centróides de onde o Wally está na imagem, foi realizado um processo de inferência nos dados de treino e teste, e foram gerados os arquivos csv com os centróides de cada imagem de cada um dos conjuntos. Foi calculado a distância euclidiana média dos centróides de referência para os centróides obtidos com o modelo de classificação.
Em média, o valor encontrado é de 8.7 pixels. O valor é bom, mas poderia ser melhor, pois, o resultado da segmentação das imagens com fundo branco é uma segmentação nos contornos do Wally, e não uma segmentação retangular como está nas anotações, portanto, o centróide sofre variação. 

# Setup e Execução

## 2. Configure o Projeto

Como o script roda no Google Colab, basta apenas importar os arquivos `wheresWally-solution.ipynb` e `config.ini`.
### 2.1 Configure o arquivo `config.ini`
Após importar o arquivo `config.ini`, configure todos os parâmetros necessários para localização da base e arquivos de saída. Você também pode manter os parâmetros já definidos e colocar os dados de treino e teste no seu google drive, portanto, o caminho para os dados serão os únicos parâmetros necessários para alteração. No meu caso, as pastas `TrainingSet` e `TestSet` estão localizadas em `/content/drive/MyDrive/colab/dataset`. **Não se esqueça** de colocar o arquivo `coco-annotations.json` que está na raiz deste projeto para a pasta `/content/drive/MyDrive/colab/dataset/TrainingSet` do seu Google Drive.
Ao importar o arquivo `wheresWally-solution.ipynb` você perceberá que haverá um registro nas saídas de algumas células com relação ao treinamento executado.

## 3. Execute e obtenha os resultados
Com tudo importado e configurado, basta executar um `Run all (Ctrl+F9)` na aba de menu `Runtime` ou pelo atalho. O treinamento deve demorar em torno de 5 minutos para concluir. Ao final, serão gerados os arquivos de saída `wheres_wally_train_result.csv` e `wheres_wally_test_result.csv` com os centróides de cada um dos dados, estes arquivos estão na pasta `results` do projeto, juntamente com as imagens `wally_005_result.png` e `wally_010_result.png` que demonstram o resultado da detecção do modelo. Na pasta `output` estão os registros salvos do treinamento, estes registros podem ser carregados no tensorboard.

> Você precisará logar na sua conta google para habilitar o acesso ao google drive.
> Caso o ocorra `AttributeError: module 'PIL.TiffTags' has no attribute 'IFD'` basta executar um "Restart Runtime" e executar novamente todo o script. Isto ocorre pois toda vez que o pacote `PIL`é instalado, é necessário um `Restart Runtime`.

# Construído Com 
* [Python](https://www.python.org/downloads/release/python-370/)
* [Google Colaboratory](https://colab.research.google.com/)
* [Detectron2](https://detectron2.readthedocs.io/en/latest/)
* [Pandas](https://pandas.pydata.org/)
* [Scikit-learn](https://scikit-learn.org/stable/)
* [Labelme](https://github.com/wkentaro/labelme)
