from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class GeneralHealth(str, Enum):
    """How would you say that in general your health is?"""
    EXCELLENT = "EXCELLENT"
    VERY_GOOD = "VERY_GOOD"
    GOOD = "GOOD"
    FAIR = "FAIR"
    POOR = "POOR"

    @staticmethod
    def map(value) -> float:
        mapping = {
            "EXCELLENT": 1.0,
            "VERY_GOOD": 2.0,
            "GOOD": 3.0,
            "FAIR": 4.0,
            "POOR": 5.0
        }
        return mapping[value]


class Age(str, Enum):
    """How old are you?"""
    EIGHTEEN_TO_TWENTY_FOUR = "EIGHTEEN_TO_TWENTY_FOUR"
    TWENTY_FIVE_TO_TWENTY_NINE = "TWENTY_FIVE_TO_TWENTY_NINE"
    THIRTY_TO_THIRTY_FOUR = "THIRTY_TO_THIRTY_FOUR"
    THIRTY_FIVE_TO_THIRTY_NINE = "THIRTY_FIVE_TO_THIRTY_NINE"
    FORTY_TO_FORTY_FOUR = "FORTY_TO_FORTY_FOUR"
    FORTY_FIVE_TO_FORTY_NINE = "FORTY_FIVE_TO_FORTY_NINE"
    FIFTY_TO_FIFTY_FOUR = "FIFTY_TO_FIFTY_FOUR"
    FIFTY_FIVE_TO_FIFTY_NINE = "FIFTY_FIVE_TO_FIFTY_NINE"
    SIXTY_TO_SIXTY_FOUR = "SIXTY_TO_SIXTY_FOUR"
    SIXTY_FIVE_TO_SIXTY_NINE = "SIXTY_FIVE_TO_SIXTY_NINE"
    SEVENTY_TO_SEVENTY_FOUR = "SEVENTY_TO_SEVENTY_FOUR"
    SEVENTY_FIVE_TO_SEVENTY_NINE = "SEVENTY_FIVE_TO_SEVENTY_NINE"
    EIGHTY_OR_OLDER = "EIGHTY_OR_OLDER"

    @staticmethod
    def map(value) -> float:
        mapping = {
            "EIGHTEEN_TO_TWENTY_FOUR": 1.0,
            "TWENTY_FIVE_TO_TWENTY_NINE": 2.0,
            "THIRTY_TO_THIRTY_FOUR": 3.0,
            "THIRTY_FIVE_TO_THIRTY_NINE": 4.0,
            "FORTY_TO_FORTY_FOUR": 5.0,
            "FORTY_FIVE_TO_FORTY_NINE": 6.0,
            "FIFTY_TO_FIFTY_FOUR": 7.0,
            "FIFTY_FIVE_TO_FIFTY_NINE": 8.0,
            "SIXTY_TO_SIXTY_FOUR": 9.0,
            "SIXTY_FIVE_TO_SIXTY_NINE": 10.0,
            "SEVENTY_TO_SEVENTY_FOUR": 11.0,
            "SEVENTY_FIVE_TO_SEVENTY_NINE": 12.0,
            "EIGHTY_OR_OLDER": 13.0
        }
        return mapping[value]


class Income(str, Enum):
    """What is your income?"""
    LESS_THAN_10K = "LESS_THAN_10K"
    BETWEEN_10K_AND_15K = "BETWEEN_10K_AND_15K"
    BETWEEN_15K_AND_20K = "BETWEEN_15K_AND_20K"
    BETWEEN_20K_AND_25K = "BETWEEN_20K_AND_25K"
    BETWEEN_25K_AND_35K = "BETWEEN_25K_AND_35K"
    BETWEEN_35K_AND_50K = "BETWEEN_35K_AND_50K"
    BETWEEN_50K_AND_75K = "BETWEEN_50K_AND_75K"
    SEVENTY_FIVE_THOUSAND_OR_MORE = "SEVENTY_FIVE_THOUSAND_OR_MORE"

    @staticmethod
    def map(value) -> float:
        mapping = {
            "LESS_THAN_10K": 1.0,
            "BETWEEN_10K_AND_15K": 2.0,
            "BETWEEN_15K_AND_20K": 3.0,
            "BETWEEN_20K_AND_25K": 4.0,
            "BETWEEN_25K_AND_35K": 5.0,
            "BETWEEN_35K_AND_50K": 6.0,
            "BETWEEN_50K_AND_75K": 7.0,
            "SEVENTY_FIVE_THOUSAND_OR_MORE": 8.0
        }
        return mapping[value]


class DiabetesRiskModelInput(BaseModel):
    body_mass_index: Optional[int] = Field(ge=15, le=60, description="Body Mass Index.")
    general_health: Optional[GeneralHealth] = Field(description="How would you say that in general your health is?")
    age: Optional[Age] = Field(description="How old are you?")
    income: Optional[Income] = Field(description="What is your income?")


class DiabetesRisk(str, Enum):
    """Risk of diabetes."""
    NO_DIABETES = "NO_DIABETES"
    DIABETES = "DIABETES"

    @staticmethod
    def map(value: float) -> str:
        mapping = {
            0.0: DiabetesRisk.NO_DIABETES,
            1.0: DiabetesRisk.DIABETES
        }
        return mapping[value]


class DiabetesRiskModelOutput(BaseModel):
    """Diabetes risk model output."""
    diabetes_risk: DiabetesRisk
