import numpy as np
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator 
from pgmpy.inference import VariableElimination


heartDisease = pd.read_csv('./datasets/heart.csv') 
heartDisease = heartDisease.replace('?',np.nan)

print(f"Few examples from the dataset are given below : \n\n{heartDisease.head()}")

model = BayesianModel([('age','trestbps'),('age','fbs'),('sex','trestbps'),('exang','trestbps'),('trestbps','heartdisease'),
                       ('fbs','heartdisease'),('heartdisease','restecg'), ('heartdisease','thalach'),('heartdisease','chol')])


print('\nLearning CPD using Maximum likelihood estimators')
model.fit(heartDisease, estimator=MaximumLikelihoodEstimator)


print('Inferencing with Bayesian Network:') 
HeartDisease_infer = VariableElimination(model)


print('1. Probability of HeartDisease given Age=38') 
q = HeartDisease_infer.query(variables=['heartdisease'], evidence={'age':38})
print(q)
 
print('\n 2. Probability of HeartDisease given cholesterol=230') 
q=HeartDisease_infer.query(variables=['heartdisease'], evidence ={'chol':230})
print(q)