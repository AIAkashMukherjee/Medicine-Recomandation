import os, sys
from src.logger.custom_logging import logger
from src.exceptions.expection import CustomException
from dataclasses import dataclass
from src.utlis.utlis import save_obj,model_evaluate
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from src.entity.config_entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self) :
        self.model_trainer_config=ModelTrainerConfig()   

    def initate_model_trainer(self,train_array,test_array):
        try:
            X_train,y_train,X_test,y_test=(
            train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
                )
            model = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "svc": SVC(kernel='linear')
            }

            

            model_report:dict=model_evaluate(X_train,y_train,X_test,y_test,model)
            print(model_report)
            print('\n====================================================================================\n')
            logger.info(f'Model Report : {model_report}')

            # To get best model score from dictionary 
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = model[best_model_name]

            print(f"Best Model Found, Model Name is: {best_model_name},Precision_Score: {best_model_score}")
            print("\n***************************************************************************************\n")
            logger.info(f"best model found, Model Name is {best_model_name}, Precison Score: {best_model_score}")


            save_obj(file_path=self.model_trainer_config.train_model_file_path,
                        obj = best_model
                        )
            

        except Exception as e:
            raise CustomException(e,sys)     