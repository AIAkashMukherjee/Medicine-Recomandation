import os
import sys
import pandas as pd
import numpy as np
from src.logger.custom_logging import logger
from src.exceptions.expection import CustomException
from src.utlis.utlis import load_obj, symptoms_dict

class PredictionPipeline:
    def __init__(self) -> None:
        self.df = pd.read_csv('artifacts/data_ingestion/raw.csv')
        self.sym_des = pd.read_csv('artifacts/data_ingestion/sys.csv')
        self.precautions = pd.read_csv('artifacts/data_ingestion/precautions_df.csv')
        self.workout = pd.read_csv('artifacts/data_ingestion/workout_df.csv')
        self.description = pd.read_csv('artifacts/data_ingestion/description.csv')
        self.medications = pd.read_csv('artifacts/data_ingestion/medications.csv')
        self.diets = pd.read_csv('artifacts/data_ingestion/diets.csv')
        self.symptoms_dict = symptoms_dict

    def helper(self, dis):
        if dis is None or dis == "Unknown":
            return "Unknown", [], [], [], []

    # Make sure 'dis' is lowercase
        dis = dis.lower()

    # Get description of the disease
        desc_row = self.description[self.description['Disease'].str.lower() == dis]
        desc = " ".join(desc_row['Description'].values) if not desc_row.empty else "Description not available."

    # Get precautions related to the disease
        pre_rows = self.precautions[self.precautions['Disease'].str.lower() == dis]
        pre = [list(row) for row in pre_rows[['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].values] or []

    # Get medications for the disease
        med = self.medications[self.medications['Disease'].str.lower() == dis]['Medication'].tolist() or []

    # Get diet for the disease
        die = self.diets[self.diets['Disease'].str.lower() == dis]['Diet'].tolist() or []

    # Get workout recommendations for the disease
        wrkout = self.workout[self.workout['disease'].str.lower() == dis]['workout'].tolist() or []

        return desc, pre, med, die, wrkout


    def predict_disease(self, patient_symptoms):
        input_vector = np.zeros(len(self.symptoms_dict))
        for item in patient_symptoms:
            item = item.lower().strip()
            if item in self.symptoms_dict:
                input_vector[self.symptoms_dict[item]] = 1

        if not np.any(input_vector):
            print("No matching symptoms found. Please check the symptoms you entered.")
            return "Unknown Disease"

        try:
            preprocessor_obj_path = os.path.join("artifacts/data_transformation", "preprocessor.pkl")
            model_path = os.path.join("artifacts/model_trainer", "model.pkl")

            processor = load_obj(preprocessor_obj_path)
            model = load_obj(model_path)

            scaled = processor.transform([input_vector])
            prediction_index = model.predict(scaled)[0]

            return self.df['Disease'].iloc[prediction_index] if not self.df.empty else "Unknown Disease"
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            raise CustomException(e, sys)