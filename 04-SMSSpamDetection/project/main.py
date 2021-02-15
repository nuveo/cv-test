import argparse

from config import Config
from dataset import DataSet
from spam_filter import SpamFilter
from sklearn.metrics import classification_report
from util.logger import setup_logger

logger = setup_logger(name=__name__)

def run(train_file, test_file, output_file, val_percent):
    
    dataset = DataSet(train_file, val_percent=val_percent)
    
    # Treinamento com base balanceada
    logger.debug("Treinando modelo com base balanceada...")
    spam_filter_balanced = SpamFilter()
    spam_filter_balanced.train_model(dataset.data_train, dataset.label_train)
    
    val_predictions_a = spam_filter_balanced.run_inference(dataset.data_val)
    logger.debug("Modelo não balanceado \n {}".format(classification_report(dataset.label_val, val_predictions_a)))
    
    # Treinamento com base desbalanceada
    logger.debug("Treinando modelo com base desbalanceada...")
    spam_filter_unbalanced = SpamFilter()
    spam_filter_unbalanced.train_model(*dataset.apply_random_undersampler())
    
    val_predictions_b = spam_filter_unbalanced.run_inference(dataset.data_val)
    logger.debug("Modelo balanceado \n {}".format(classification_report(dataset.label_val, val_predictions_b)))
    
    # Inferencia
    x_inference, dataframe_inference = dataset.load_inference_dataset(test_file)
    predictions = spam_filter_balanced.run_inference(x_inference)
    dataframe_inference.insert(0, 'label', predictions)
    dataframe_inference.to_csv(output_file, index=False, sep="\t", header=False)
    
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-cf", "--configfile", default="/workspace/config.ini", help="caminho para arquivo de configuração.")
    args = vars(ap.parse_args())
    config_properties = Config(args["configfile"])
    
    logger.debug('Executando...')
    run(config_properties.train_file, 
        config_properties.test_file, 
        config_properties.output_file,
        config_properties.val_percent)
    logger.debug('Fim da execução!')
    
if __name__ == '__main__':
    main()