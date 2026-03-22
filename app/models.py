from pydantic import BaseModel
from typing import List

class Ticket(BaseModel):
    type: str
    date: str

class Customer(BaseModel):
    monthly_charges: float
    previous_month_charges: float
    contract_type: str
    tickets: List[Ticket]