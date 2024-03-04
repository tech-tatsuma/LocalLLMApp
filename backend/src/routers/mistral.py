from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel, Field
from typing import Dict
import csv
import output_parsers
from .functions.mistral import get_answer_with_search, get_answer_with_retrieval, get_answer_with_math

router = APIRouter()

@router.post("/search", response_model=Dict[str, str])
def askwithsearch(question: str):
    return get_answer_with_search(question)

@router.post("/retrieval", response_model=Dict[str, str])
def askwithretrieval(context: str, question: str):
    return get_answer_with_retrieval(context, question)

@router.post("/math", response_model=Dict[str, str])
def askwithqa(question: str):
    return get_answer_with_math(question)