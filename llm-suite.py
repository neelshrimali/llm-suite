from tkinter import Image
from openai import OpenAI, OpenAIError
import os
import tempfile
from flask import Flask, request, jsonify
from PyPDF2 import PdfReader
import json
from flask_cors import CORS
import fitz  # PyMuPDF
import easyocr

app = Flask(__name__)
CORS(app)
reader = easyocr.Reader(['en'])

# OCR Doc for Extract Bill details
def parse_doc(str1):
    try:
        str1 = readDoc(str1)

        client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
        completion = client.chat.completions.create(
                        model="model-identifier",
                        messages=[          
                          { "role": "user", 
                            "content": f'This is my string -  {{{str1}}}, give me this {{ "total_amount" : "total_amount", "expense type" : "expense type" , "invoice_date" : "invoice_date"}} in json format. Infer the "expense_type" based on the context provided in the description. Give me result only no other words at all, not even new lines as well.',
                          }
                        ],
                        temperature=0.7)
        
        res = str(completion.choices[0].message)
        start_index = res.find('{')
        end_index = res.rfind('}')
        
        if start_index == -1 or end_index == -1:
            raise ValueError("JSON-like content not found in response.")
        
        json_string = res[start_index:end_index+1]
        json_string_main = json_string.strip()
        
        # Attempt to parse the JSON to ensure it's valid
        try:
            parsed_json = json.loads(json_string_main)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON: {e}")
        
        return str(parsed_json)

    except OpenAIError as e:
        return f"OpenAI API error: {e}"
    except ValueError as e:
        return f"Value error: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

def readDoc(str1):
    try:
        result = reader.readtext(str1)
        res = ''
        for bbox, text, prob in result:
            res += text + "\n"
        return res
    except Exception as e:
        return f"An error occurred while reading the document: {e}"

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

@app.route('/ocr_doc', methods=['POST'])
def handle_parse_doc():
    try:
        # Assuming the file is sent as form data in the request
        file = request.files['doc']
        _, file_extension = os.path.splitext(file.filename)
        file_extension = file_extension.lower()
    
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            file.save(temp_file.name)
            temp_file_path = temp_file.name

        if file_extension in ['.jpg', '.jpeg', '.png']:
            # Directly parse the image file
            json_output = parse_doc(temp_file_path)
        elif file_extension == '.pdf':
            # Convert PDF to an image (one-page PDF)
            with fitz.open(temp_file_path) as doc:
                page = doc.load_page(0)  # Load the first page
                pix = page.get_pixmap()  # Render page to an image
                
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as img_temp_file:
                    # pix.save(img_temp_file.name)
                    img_temp_file_path = img_temp_file.name
                    img.save(img_temp_file_path)
                    json_output = parse_doc(img_temp_file_path)
                    img_temp_file.close()
                    os.unlink(img_temp_file_path)
          
        else:
            return jsonify({"error": "Unsupported file type"}), 400

        # Clean up - close and delete the temporary file
        temp_file.close()
        os.unlink(temp_file_path)

    except Exception as e:
        print(str(e))
        return jsonify({"error": str(e)}), 500

    return jsonify(json_output)

if __name__ == "__main__":
    app.run(debug=True)
