import configparser
import sys

from os import path
from util.logger import setup_logger

class Config():

    logger = setup_logger(name=__name__)
    def __init__(self, config_file_path):

        try:
            if path.isfile(config_file_path):

                config = configparser.ConfigParser()
                config.read(config_file_path)

                self.train_file = config['DEFAULT']['TrainFile']
                self.test_file = config['DEFAULT']['TestFile']
                self.output_file = config['DEFAULT']['OutputFile']
                self.val_percent = float(config['DEFAULT']['ValPercent'])
                
            else:
                raise FileNotFoundError('Arquivo de configuração não encontrado: \n {}'.format(config_file_path))
        except NameError as e:
            logger.exception("Parâmetro não encontrado no arquivo de configuração", exc_info=False)
            sys.exit(1)
        except FileNotFoundError as e:
            logger.exception(e, exc_info=False)
            sys.exit(1)