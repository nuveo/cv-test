from sklearn.naive_bayes import MultinomialNB

class SpamFilter():
    """
    Modelo de classificação probabilístico 'Multinomial Naive Bayes'.
    """
    def __init__(self):
        self.model = MultinomialNB()

    def train_model(self, x_train, y_train):
        """
        Treina modelo

        Args:
            x_train (ndarray), dimensões (n_mensagens, n_tokens): dados de treinamento.
            y_train (ndarray), dimensões (n_mensagens, ): labels dos dados de treinamento.
        """
        self.model.fit(x_train, y_train)
        
    def run_inference(self, inpt):
        """
        Executa inferência

        Args:
            inpt (ndarray), dimensões (n_mensagens, n_tokens): dados de inferência.

        Returns:
            labels (ndarray): classificação de cada vetor de atributos.
        """
        return self.model.predict(inpt)