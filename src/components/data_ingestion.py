import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os,sys
from src.exceptions.expection import CustomException
from src.logger.custom_logging import logger
from src.entity.config_entity import DataIngestionConfig

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class DataIngestion:
    def __init__(self) -> None:
        self.data_ingestion_config=DataIngestionConfig()

    def initate_data_ingestion(self):
        try:
            logger.info('Reading Datasets')

            df=pd.read_csv(os.path.join('data','Training.csv'))
            sym_des = pd.read_csv(os.path.join('data','symtoms_df.csv'))
            precautions = pd.read_csv(os.path.join('data','precautions_df.csv'))
            workout = pd.read_csv(os.path.join('data','workout_df.csv'))
            description = pd.read_csv(os.path.join('data','description.csv'))
            medications = pd.read_csv(os.path.join('data','medications.csv'))
            diets = pd.read_csv(os.path.join('data','diets.csv'))

            logger.info('Creating Directories')

            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_file_path),exist_ok=True)

            logger.info('Saving Datasets')

            df.to_csv(self.data_ingestion_config.raw_file_path,index=False)
            sym_des.to_csv(self.data_ingestion_config.sys_file_path,index=False)
            precautions.to_csv(self.data_ingestion_config.precautions_file_path,index=False)
            workout.to_csv(self.data_ingestion_config.workout_file_path,index=False)
            description.to_csv(self.data_ingestion_config.desc_file_path,index=False)
            medications.to_csv(self.data_ingestion_config.medications_file_path,index=False)
            diets.to_csv(self.data_ingestion_config.diets_file_path,index=False)

            logger.info('train test split')

            train_set,test_set=train_test_split(df,test_size=.2,random_state=42)

            logger.info('Train test split of dataframe')

        

            train_set.to_csv(self.data_ingestion_config.train_file_path,index=False)

            test_set.to_csv(self.data_ingestion_config.test_file_path,index=False)

            logger.info('Saved train set and test set')

            logger.info('Data ingestion complete')

            return (
                self.data_ingestion_config.train_file_path,
                self.data_ingestion_config.test_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)    
    