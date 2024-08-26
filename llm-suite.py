from openai import OpenAI
import os
import tempfile
from flask import Flask, request, jsonify
from PyPDF2 import PdfReader
import json
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 

def res_percentageMatch(file_path, jd):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    with open(file_path, 'rb') as file:
        pdf_text = ""
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            pdf_text += page.extract_text()

    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
    completion = client.chat.completions.create(
                    model="model-identifier",
                    messages=[          
                      { "role": "user", 
                        "content": 'Given a job description - {' + jd + '} and a resume - {' + pdf_text + '}, Provide resume match percentage in json format as output example like {"resume_match_percentage": "percentage"}. Give me result only no other words at all.',
                      }
                      ],
                    temperature=0.7,)
    res= str(completion.choices[0].message)
    start_index = res.find('{')
    end_index = res.rfind('}')
    json_string = res[start_index:end_index+1]
    json_string_main = json_string.strip()
    
    return str(json_string_main)
    
@app.route('/percentageMatch_resume', methods=['POST'])
def handle_percentageMatch_resume():
    # Assuming the PDF file is sent as form data in the request
    file = request.files['resume']
    jd = request.form['jd']
    _, file_extension = os.path.splitext(file.filename)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
    file.save(temp_file.name)

    # Parse the resume and return the structured JSON
    json_output = res_percentageMatch(temp_file.name,jd)
    
    # Clean up - close and delete the temporary file
    temp_file.close()
    os.unlink(temp_file.name)

    return json_output

if __name__ == "__main__":
    app.run(debug=True)