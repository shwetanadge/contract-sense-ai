from groq import Groq
from dotenv import load_dotenv
import os

#Load API key from .env file
load_dotenv()

#Create Groq Client
client = Groq(api_key = os.getenv("GROQ_API_KEY"))

#Make your first API call
response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
         {"role": "user", "content": "What is GDPR? Respond in JSON format with keys 'answer' and 'regulation_reference'"}
    ]
)

#Print the response
print(response.choices[0].message.content)