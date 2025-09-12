import pandas as pd

def load_lightcurve(file_path):  
  data = pd.read_csv(file_path)
  return data

def normalize(data, column='flux'):
  values = data[column].values
  norm = (values - min(values)) / (max(values) - min(values))
  data[column] = norm 
  return data 
