import re
import nltk
import logging
import pandas as pd
import sys

nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler

from util.logger import setup_logger

class DataSet():
    """
    Classe para gerenciar e carregar base de dados
    
    Parameters:
        train_file: str
            Caminho para o arquivo csv de treino.
            
        col_name_sms: str, default='sms'
            Nome da coluna de mensagens do dataframe
            
        col_name_label: str, default='label'
            Nome da coluna de labels do dataframe
            
        val_percent: float, default=0.2
            Percentual do conjunto total de treinamento que será usado para validação.
    
    Attributes:
        data_train: ndarray de dimensões (n_mensagens, n_tokens)
            Dados de treino
            
        data_val: ndarray de dimensões (n_mensagens, n_tokens)
            Dados de validação
            
        label_train: ndarray de dimensões (n_mensagens, )
            Labels dos dados de treinamento
            
        label_val: ndarray de dimensões (n_mensagens, )
            Labels dos dados de validação
            
        transformer: CountVectorizer 
            conversor das mensagens de entrada em matriz sparsa no formato da data_train
            
        language: str
            Língua dos dados de entrada
        
    
    """
    
    def __init__(self, train_file, col_name_sms='sms', col_name_label='label', val_percent=0.2, language='english'):
        self.logger = logging.getLogger(name=__name__)
        self.data_train = None
        self.label_train = None
        self.data_val = None
        self.label_val = None
        self.transformer = None
        self.language = language
        self.load_dataset(train_file, col_name_sms, col_name_label, val_percent)
        

    def __sms_tokenization(self, sms):
            """ Aplica 'tokenization'. 
            Converte uma mensagem de texto em um vetor com os tokens.

            Args:
                sms (str): mensagens de texto

            Returns:
                list: retorna uma lista de strings representando os tokens. 
            """

            # Primeiro é necessário a remoção de caracteres especiais e números.
            # Utilizaresmo para isso uma expressão regular.
            sms_non_al = re.sub('[^A-Za-z]', ' ', sms)
            
            # Convertendo letras para caixa baixa e removendo "Stopwords" (palavras com muito pouco significado).
            sms_result_lower = [word.lower() for word in sms_non_al.split()]
            lemmatizer = WordNetLemmatizer()
            
            return [lemmatizer.lemmatize(word) for word in sms_result_lower if word not in stopwords.words(self.language)]
        

    def __count_vetorizer(self, dataset):
        """Gera vetor de atributos para cada linha de texto do dataframe.

        Args:
            dataset (dataframe): pandas dataframe.

        """

        # Uma forma de gerar essa informação é levando em consideração quantas vezes uma determinada palavra ocorre em cada mensagem.
        # Para isso, usaremos `CountVectorizer`, que gerará uma matriz bidimensional com os tokens e quantas vezes eles ocorrem em cada mensagem.
        transformer = CountVectorizer(analyzer=self.__sms_tokenization).fit(dataset)
        self.transformer = transformer
    
    
    def load_dataset(self, train_file:str, col_name_sms='sms', col_name_label='label', val_percent=0.2):
        """Carrega dados de treino e validação e aplica preprocessamento.

        Args:
            train_file (str): caminho para arquivo de treinamento
            col_name_sms (str, optional): string para o nome da coluna de mensagens
            col_name_label (str, optional): string para o nome da coluna de labels
            val_percent (float, optional): percentual do total de dados de treino separados para validação.

        """
        try:
            dataset_train = pd.read_csv(train_file, sep="\t", names=[col_name_label, col_name_sms])
        except FileNotFoundError as fnf_error:
            self.logger.exception("Arquivo não encontrado: \n {}".format(fnf_error), exc_info=False)
            sys.exit(1)
        
        # Aplica preprocessamento aos conjuntos de treino e validação, e cria o conversor
        # de tokens para atributos (`transformer`).
        self.__count_vetorizer(dataset_train[col_name_sms])
        
        # Separa dados em treino e validação.
        self.data_train, self.data_val, self.label_train, self.label_val = train_test_split(
            self.transformer.transform(dataset_train[col_name_sms]).toarray(), 
            dataset_train[col_name_label].to_numpy(), test_size=val_percent)
    
    
    def load_inference_dataset(self, file_path:str, col_name_sms='sms'):
        """
        Carrega dados para inferencia

        Args:
            file_path (str): caminho para arquivo csv com os dados a serem classificados.
            col_name_sms (str, optional): [description]. Defaults to 'sms'.

        Returns:
            inf_data (ndarray), dimensões (n_mensagens, n_tokens): dados para inferência.
            dataframe (dataframe): dataframe original dos dados carregados do arquivo csv.
        """
        try:
            dataframe = pd.read_csv(file_path, sep="\t", names=[col_name_sms])
        except FileNotFoundError as fnf_error:
            self.logger.exception("Arquivo não encontrado: \n {}".format(fnf_error), exc_info=False)
            sys.exit(1)
            
        return self.transformer.transform(dataframe[col_name_sms]).toarray(), dataframe
    
    
    def apply_random_undersampler(self):
        """
        Aplica UnderSampler para que o conjunto de dados tenha o mesmo número de elementos em todas as classes.

        Returns:
            x_train (ndarray): conjunto de treinamento
            y_train (ndarray): label do conjuento de treinamento.
        """
        rus = RandomUnderSampler()
        x_train, y_train = rus.fit_sample(self.data_train, self.label_train)
        return x_train, y_train