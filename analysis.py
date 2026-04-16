import os 
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()


key = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key= key)

model = genai.GenerativeModel('gemini-2.5-flash-lite')

def generate_summary(results_df):
    prompt = f''' you are a data scientist export.
    Here are the model results :

    {results_df.to_string()}

    1. Identify the best Model
    2. Explain why it is best
    3. summarise the performamce of the model'''

    response = model.generate_content(prompt)

    return response.text

def suggest_improvements(results_df):
    prompt = f''' you are a data scientist export.
    Here are the model results :

    {results_df.to_string()}

     suggest:
    - Ways to improve the model performance
    - Hyperparameter tuning and give range od values in each parameter
    - Better suitable algorithms for the given data
    - Data preprocessing improvements'''
     
    response = model.generate_content(prompt)

    return response.text