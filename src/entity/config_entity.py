from dataclasses import dataclass
import os


@dataclass
class DataIngestionConfig:
    train_file_path=os.path.join('artifacts/data_ingestion','train.csv')
    test_file_path=os.path.join('artifacts/data_ingestion','test.csv')
    raw_file_path=os.path.join('artifacts/data_ingestion','raw.csv')
    sys_file_path=os.path.join('artifacts/data_ingestion','sys.csv')
    precautions_file_path=os.path.join('artifacts/data_ingestion','precautions_df.csv')
    workout_file_path=os.path.join('artifacts/data_ingestion','workout_df.csv')
    desc_file_path=os.path.join('artifacts/data_ingestion','description.csv')
    medications_file_path=os.path.join('artifacts/data_ingestion','medications.csv')
    diets_file_path=os.path.join('artifacts/data_ingestion','diets.csv')

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts/data_transformation', 'preprocessor.pkl')


@dataclass 
class ModelTrainerConfig:
    train_model_file_path=os.path.join('artifacts/model_trainer','model.pkl') 