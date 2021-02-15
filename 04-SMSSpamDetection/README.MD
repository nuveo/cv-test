# SMSSpamDetection Solution
---

Projeto desenvolvido em [Python](https://www.python.org/downloads/release/python-370/) para criação de um modelo capaz de classificar mensagens como 'ham' ou 'smap'.

# Descrição da solução

Esta solução utliza tarefas básicas de *Natural Language Processing (NLP)*. Uma biblioteca comumente usada para NLP em Python é a *Natural Language Toolkit* (NLTK).
Inicialmente, foi necessário carregar e processar os dados de entrada. Cada mensagem (linha) dos dados precisou passar por um preprocessamento, onde o objetivo era converter a mensagem original em dados que fossem possíveis para o treinamento de um modelo em *Machine Learning*.

### Preprocessamento
Primeiro passo é aplicar o método *tokenization*, que é responsável por quebrar o texto em elementos menores com significados, estes elementos são conhecidos como *tokens*. Este processo consiste em:
* Remover caracteres especiais e números das mensagens
* Garantir que todas as palavras estejam em letra minúscula
* Quebrar todas as mensagens em *tokens*
* Aplicar *lemmatization*, que é o processo de análise morfológica das palavras para remover terminações e obter o seu correto lema com base no contexto.

Por último, convertemos a matriz que possui como linhas os vetores de *tokens* em uma matriz sparsa que contém quantas vezes cada *token* apareceu em cada mensagem (linha).

Ao final deste processo, já podemos utilizar estes dados para treinar nosso modelo. Ainda cabe alguns processamentos nessa matriz sparsa, como uma normalização da ocorrência dos *tokens*, já que cada mensagem tem comprimento diferente. Todavia, este processo não foi necessário nesta solução.

### Treinamento do Modelo
O classificador utilizado foi o *Multinomial Naive Bayes* que é adequado para classificação com atributos discretos, como por exemplo, contagem de palavras para classificação de texto. Mantemos os parâmetros como *default* e ajustamos o modelo com o conjunto de dados. Treinamos 2 modelos, uma com a base originalmente desbalanceada, e outro com a base balanceada.


### Resultados
Para análise do modelo, utilizamos as métricas de *precision* e *recall*. Para este problema, são muito adequadas, já que uma dos requisitos do problema é que a chance de o classificador determinar um 'spam' como 'ham' deve ser menor do que classificar 'ham' como 'spam'. Portanto, devemos buscar que o nosso classificador tenha a métrica *recall* maior para a classe 'spam' do que 'ham'. Por exemplo, se para a classe 'spam' o *recall* é igual a 1, isso significa que ele é capaz de identificar todas as mensagens 'spams' do conjunto de dados. Este seria o cenário ideal, já que é tolerável não possuir um *precision* máximo (classificar alguns da classe 'ham' como 'spam'). Por este motivo, o modelo com a base balanceada apresenta resultado melhor para o problema, pois a classe 'spam' possui *recall* maior, se comparado ao modelo treinado com a base desbalanceada.

Os resultados aqui podem ser visualizados no terminal ou no arquivo de log `output.log` ao final da execução do programa. O arquivo `sms-hamspam-test-RESULT.csv` é o resultado pedido na especificação do problema. A primeira coluna diz respeito a classificação realizada pelo modelo e a segunda coluna corresponde às mensagens.

##### Modelo não balanceado
Métricas:
| class | precision (%) | recall (%) | f1-score (%) | support |
| :---: | :---: | :---: | :---: | :---: |
| ham   |  0.99 | 0.99  | 0.99 | 823 |
| spam  | 0.91  | 0.95  | 0.93 | 122 |


##### Modelo balanceado
Métricas:
| class | precision (%) | recall (%) | f1-score (%) | support |
| :---: | :---: | :---: | :---: | :---: |
| ham   |  1.00 | 0.92  | 0.96 | 823 |
| spam  | 0.64  | 0.98  | 0.77 | 122 |

# Setup e Execução
Os passos seguintes consideram que o sistema operacional seja o [Ubuntu 18.04.5 LTS](https://releases.ubuntu.com/18.04/).
## 1. Instalação das dependências
* [docker](https://docs.docker.com/install/linux/docker-ce/ubuntu/) v20.10.2
* [docker-compose](https://docs.docker.com/compose/install/) v1.25.3

## 2. Configure o Projeto
### 2.1 Configure o arquivo `config.ini`
No arquivo `config.ini` são definidos os caminhos para os arquivos de treino, teste, arquivo de saída e o percentual do dataset que será usado para validação. A pasta do projeto dentro do container é `/workspace`, por isso todos os caminhos já predefinidos estão dentro desta pasta. Os dados de treino e teste **precisam estar dentro da pasta do projeto**. Portanto, basta colocar a pasta com os dados e configurar o caminho de cada um deles no arquivo.

### 2.2 configure o arquivo `logger_config.yaml`
O arquivo de configuração `logger_config.yaml` define o formato, *handlers*, *loggers* e os respectivos níveis de saída para o arquivo de log e terminal. Já está basicamente configurado, não é necessário fazer qualquer alteração.

## 3. Execute e obtenha os resultados
Navegue até a pasta do projeto e execute os comandos:
    
    chmod +x run.sh
    ./run.sh

O programa será executado e o arquivo csv de saída com a classificação das mensagens, como foi pedido na especificação, será gerado de acordo com o caminho que foi configurado no config.ini. No arquivo `output.log` você verá o log da execução do programa, nele aparecerão as métricas dos dois modelos como foi mostrado no tópico de resultados.

# Construído Com 
* [Docker](https://www.docker.com/)
* [Docker-Compose](https://docs.docker.com/compose/)
* [Natural Language Toolkit](https://www.nltk.org/)
* [Pandas](https://pandas.pydata.org/)
* [Scikit-learn](https://scikit-learn.org/stable/)
* [Imbalanced-learn](https://pypi.org/project/imbalanced-learn/)
