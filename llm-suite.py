from openai import OpenAI, OpenAIError
import os
import tempfile
from flask import Flask, request, jsonify
from PyPDF2 import PdfReader, PdfReadError
import json
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# resume percentage match func
def res_percentageMatch(file_path, jd):
    try:
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
                            "content": f'Given a job description - {{ {jd} }} and a resume - {{ {pdf_text} }}, Provide resume match percentage in json format as output example like {{"resume_match_percentage": "percentage"}}. Give me result only no other words at all.',
                          }
                          ],
                        temperature=0.7,)
        res = str(completion.choices[0].message)
        start_index = res.find('{')
        end_index = res.rfind('}')
        json_string = res[start_index:end_index+1]
        json_string_main = json_string.strip()
        return json_string_main
    except PdfReadError:
        return jsonify({"error": "Failed to read the PDF file."}), 400
    except OpenAIError as e:
        return jsonify({"error": f"OpenAI API error: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

# resume parser func
def parse_resume(file_path):
    try:
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
                            "content": f'Extract the following information from this resume text: {{ {pdf_text} }}. Make sure to find and include "name", "email", "contact_no", "degree" and "skills" in the JSON format. Only provide the JSON object as the output, with no additional text or new lines. The output must be a single line JSON object.',
                          }
                          ],
                        temperature=0.7,)
        res = str(completion.choices[0].message)
        start_index = res.find('{')
        end_index = res.rfind('}')
        json_string = res[start_index:end_index+1]
        json_string_main = json_string.strip()
        return completion.choices[0].message.content
    except PdfReadError:
        return jsonify({"error": "Failed to read the PDF file."}), 400
    except OpenAIError as e:
        return jsonify({"error": f"OpenAI API error: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

@app.route('/percentageMatch_resume', methods=['POST'])
def handle_percentageMatch_resume():
    try:
        file = request.files['resume']
        jd = request.form['jd']
        _, file_extension = os.path.splitext(file.filename)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
        file.save(temp_file.name)

        json_output = res_percentageMatch(temp_file.name, jd)
        
        temp_file.close()
        os.unlink(temp_file.name)

        return json_output
    except KeyError:
        return jsonify({"error": "Missing 'resume' file or 'jd' parameter in the request."}), 400
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

@app.route('/parse_resume', methods=['POST'])
def handle_parse_resume():
    try:
        file = request.files['resume']
        _, file_extension = os.path.splitext(file.filename)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
        file.save(temp_file.name)

        json_output = parse_resume(temp_file.name)
        
        temp_file.close()
        os.unlink(temp_file.name)
        return json_output
    except KeyError:
        return jsonify({"error": "Missing 'resume' file in the request."}), 400
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
