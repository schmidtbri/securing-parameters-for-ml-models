import os
import pandas as pd
import pickle
from io import BytesIO
import zipfile
from itsdangerous import Signer
from minio import Minio
from ml_base import MLModel

from diabetes_risk_model.prediction.schemas import DiabetesRiskModelInput, DiabetesRiskModelOutput, DiabetesRisk, \
    GeneralHealth, Age, Income


class DiabetesRiskModel(MLModel):
    """Prediction logic for the Diabetes Risk Model."""

    @property
    def display_name(self) -> str:
        """Return display name of model."""
        return "Diabetes Risk Model"

    @property
    def qualified_name(self) -> str:
        """Return qualified name of model."""
        return "diabetes_risk_model"

    @property
    def description(self) -> str:
        """Return description of model."""
        return "Model to predict the diabetes risk of a patient."

    @property
    def version(self) -> str:
        """Return version of model."""
        return "0.1.0"

    @property
    def input_schema(self):
        """Return input schema of model."""
        return DiabetesRiskModelInput

    @property
    def output_schema(self):
        """Return output schema of model."""
        return DiabetesRiskModelOutput

    def __init__(self, model_parameters_version: str,
                 model_files_bucket: str,
                 minio_url: str,
                 minio_access_key: str,
                 minio_secret_key: str,
                 parameters_signing_key: str):
        """Loads and verifies the model parameters."""
        # retrieving values from environment variables if the values provided have ${} around them
        if minio_access_key[0:2] == "${" and minio_access_key[-1] == "}":
            minio_access_key = os.environ[minio_access_key[2:-1]]

        if minio_secret_key[0:2] == "${" and minio_secret_key[-1] == "}":
            minio_secret_key = os.environ[minio_secret_key[2:-1]]

        if parameters_signing_key[0:2] == "${" and parameters_signing_key[-1] == "}":
            parameters_signing_key = os.environ[parameters_signing_key[2:-1]]

        minio_client = Minio(minio_url,
                             access_key=minio_access_key,
                             secret_key=minio_secret_key,
                             secure=False)
        try:
            # accessing the model file stored in minio
            response = minio_client.get_object(model_files_bucket,
                                               f"{self.qualified_name}-{self.version}-{model_parameters_version}.zip")
            zip_bytes = BytesIO(response.data)

            response.close()
            response.release_conn()

            # unzipping the parameters
            with zipfile.ZipFile(zip_bytes) as zf:
                if "signed_model.pkl" not in zf.namelist():
                    raise ValueError("Could not find signed model file in zip file.")
                signed_model_bytes = zf.read("signed_model.pkl")
        except Exception as e:
            raise RuntimeError("Could not access model file.") from e

        # checking the signed parameters
        signer = Signer(parameters_signing_key)
        unsigned_model_bytes = signer.unsign(signed_model_bytes)

        # unpickling the model object
        self._model = pickle.loads(unsigned_model_bytes)

    def predict(self, data: DiabetesRiskModelInput) -> DiabetesRiskModelOutput:
        """Make a prediction with the model.

        Params:
            data: Data for making a prediction with the model.

        Returns:
            The result of the prediction.

        """
        if type(data) is not DiabetesRiskModelInput:
            raise ValueError("Input must be of type 'DiabetesRiskModelInput'")

        X = pd.DataFrame([[
            None, None, None,
            data.body_mass_index,
            None, None, None, None, None, None, None, None, None,
            GeneralHealth.map(data.general_health),
            None, None, None, None,
            Age.map(data.age),
            None,
            Income.map(data.income)
        ]],
            columns=["HighBloodPressure", "HighCholesterol", "CholesterolChecked", "BMI", "Smoker", "Stroke",
                     "HeartDiseaseOrHeartAttack", "PhysicalActivity", "Fruits", "Veggies",
                     "HeavyAlchoholConsumption", "AnyHealthcare", "NoDoctorsVisitBecauseOfCost",
                     "GeneralHealth", "MentalHealth", "PhysicalHealth", "DifficultyWalking", "Sex",
                     "Age", "Education", "Income"])

        y_hat = float(self._model.predict(X)[0])

        return DiabetesRiskModelOutput(diabetes_risk=DiabetesRisk.map(y_hat))
