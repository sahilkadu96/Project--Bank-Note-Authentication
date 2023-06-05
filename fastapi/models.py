from pydantic import BaseModel, Field


class BankNote:
    variance: float
    skewness: float
    curtosis: float
    entropy: float
    def __init__(self, variance, skewness, curtosis, entropy):
        self.variance = variance
        self.skewness = skewness
        self.curtosis = curtosis
        self.entropy = entropy
     
class BankNoteRequest(BaseModel):
    variance: float = Field()
    skewness: float = Field()
    curtosis: float = Field()
    entropy: float = Field()