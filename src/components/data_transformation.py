import sys
import pandas as pd
import numpy as np
from src.logger.custom_logging import logger
from src.exceptions.expection import CustomException
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.entity.config_entity import DataTransformationConfig
from src.utlis.utlis import save_obj

class DataTransformation:
    def __init__(self) -> None:
        self.data_transformation_config = DataTransformationConfig()

    def _create_preprocessor(self):
        try:
            logger.info('Creating data transformation pipeline')
            
            num_features = ['itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills', 'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition', 'spotting_ urination', 'fatigue', 'weight_gain', 'anxiety', 'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness', 'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes', 'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache', 'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine', 'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain', 'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes', 'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen', 'history_of_alcohol_consumption', 'fluid_overload.1', 'blood_in_sputum', 'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister', 'red_sore_around_nose', 'yellow_crust_ooze']


            # Define the pipeline for numerical features
            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy='median')),
                ("scaler", StandardScaler())
            ])

            # Create ColumnTransformer
            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, num_features)
            ], remainder='passthrough')

            return preprocessor

        except Exception as e:
            logger.error(f"Error in creating data transformation pipeline: {str(e)}")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path: str, test_path: str):
        try:
            logger.info('Reading train and test data')
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            # Verify target column exists
            target_column = 'prognosis'
            # if target_column not in train_data.columns or target_column not in test_data.columns:
            #     msg = f"Column '{target_column}' not found in the data."
            #     logger.error(msg)
            #     raise ValueError(msg)

            # Split data into features and target
            drop_columns = [target_column]
            input_feature_train_data = train_data.drop(columns=drop_columns)
            target_feature_train_data = train_data[target_column]
            input_feature_test_data = test_data.drop(columns=drop_columns)
            target_feature_test_data = test_data[target_column]

            # Encode target variable
            label_encoder = LabelEncoder()
            target_feature_train_encoded = label_encoder.fit_transform(target_feature_train_data)
            target_feature_test_encoded = label_encoder.transform(target_feature_test_data)

            # Apply preprocessing
            preprocessor = self._create_preprocessor()
            input_train_arr = preprocessor.fit_transform(input_feature_train_data)
            input_test_arr = preprocessor.transform(input_feature_test_data)

            # Combine features and target
            train_array = np.c_[input_train_arr, target_feature_train_encoded]
            test_array = np.c_[input_test_arr, target_feature_test_encoded]

            # Save preprocessor
            save_obj(file_path=self.data_transformation_config.preprocessor_obj_file_path, obj=preprocessor)

            return (train_array, test_array, self.data_transformation_config.preprocessor_obj_file_path)

        except ValueError as e:
            logger.error(f"ValueError occurred: {str(e)}")
            raise CustomException(e, sys)
        except Exception as e:
            logger.error(f"An unexpected error occurred: {str(e)}")
            raise CustomException(e, sys)
