from pydantic import BaseModel

class PromptStructure(BaseModel):
    clarity: list[str]= []
    descriptive: list[str]= []
    context: list[str]= []
    style: list[str]= []
    composition: list[str]= []
    lighting: list[str]= []
    technical: list[str]= []
    negative: list[str]= []

