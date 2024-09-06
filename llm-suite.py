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
import argparse
import PyPDF2
import re
import ollama
import torch

app = Flask(__name__)
CORS(app)
reader = easyocr.Reader(['en'])

# All Background Processes and function
# chat llm func
def chatllm(pmt):    
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
    completion = client.chat.completions.create(
                    model="model-identifier",
                    messages=[          
                      { "role": "user", 
                        "content": pmt,
                      }
                      ],
                    temperature=0.7,)   

    return completion.choices[0].message.content

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

# Single Document RAG func
# RAGApp code starting
###################################################################
###################################################################


def convert_pdf_to_text(file_path):
    if file_path:        
        with open(file_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            num_pages = len(pdf_reader.pages)
            text = ''
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                if page.extract_text():
                    text += page.extract_text() + " "
            
            # Normalize whitespace and clean up text
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Split text into chunks by sentences, respecting a maximum chunk size
            sentences = re.split(r'(?<=[.!?]) +', text)  # split on spaces following sentence-ending punctuation
            chunks = []
            current_chunk = ""
            for sentence in sentences:
                # Check if the current sentence plus the current chunk exceeds the limit
                if len(current_chunk) + len(sentence) + 1 < 1000:  # +1 for the space
                    current_chunk += (sentence + " ").strip()
                else:
                    # When the chunk exceeds 1000 characters, store it and start a new one
                    chunks.append(current_chunk)
                    current_chunk = sentence + " "
            if current_chunk:  # Don't forget the last chunk!
                chunks.append(current_chunk)
            vault_folder = "vaultFiles\\"
            files_and_dirs = os.listdir(vault_folder)
            files = [f for f in files_and_dirs if os.path.isfile(os.path.join(vault_folder, f))]
            vault_filename = os.path.join(vault_folder, f"{len(files)}_vault.txt")

            with open(vault_filename, "w", encoding="utf-8") as vault_file:
                for chunk in chunks:
                    # Write each chunk to its own line
                    vault_file.write(chunk.strip() + "\n")  # Two newlines to separate chunks

            # Create JSON response
            response_data = len(files)           
            
            return jsonify(response_data)

# ANSI escape codes for colors
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

# Function to open a file and return its contents as a string
def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

# Function to get relevant context from the vault based on user input
def get_relevant_context(rewritten_input, vault_embeddings, vault_content, top_k=5):
    if vault_embeddings.nelement() == 0:  # Check if the tensor has any elements
        return []
    # Encode the rewritten input
    input_embedding = ollama.embeddings(model='mxbai-embed-large', prompt=rewritten_input)["embedding"]
    # Compute cosine similarity between the input and vault embeddings
    cos_scores = torch.cosine_similarity(torch.tensor(input_embedding).unsqueeze(0), vault_embeddings)
    # Adjust top_k if it's greater than the number of available scores
    top_k = min(top_k, len(cos_scores))
    # Sort the scores and get the top-k indices
    top_indices = torch.topk(cos_scores, k=top_k)[1].tolist()
    # Get the corresponding context from the vault
    relevant_context = [vault_content[idx].strip() for idx in top_indices]
    return relevant_context

def rewrite_query(user_input_json, conversation_history, ollama_model):
    user_input = json.loads(user_input_json)["Query"]
    context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history[-2:]])
    prompt = f"""Rewrite the following query by incorporating relevant context from the conversation history.
    The rewritten query should:    

    - Preserve the core intent and meaning of the original query
    - Expand and clarify the query to make it more specific and informative for retrieving relevant context
    - Avoid introducing new topics or queries that deviate from the original query
    - DONT EVER ANSWER the Original query, but instead focus on rephrasing and expanding it into a new query
    
    Return ONLY the rewritten query text, without any additional formatting or explanations.
    
    Conversation History:
    {context}
    
    Original query: [{user_input}]
    
    Rewritten query: 
    """
    # Ollama
    # response = client.chat.completions.create(
    #     model=ollama_model,
    #     messages=[{"role": "system", "content": prompt}],
    #     max_tokens=200,
    #     n=1,
    #     temperature=0.1,
    # )

    # LM Studio
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
    response = client.chat.completions.create(
                    model="model-identifier",
                    messages=[          
                      { "role": "user", 
                        "content": prompt,
                      }
                      ],
                    temperature=0.7,
                    )
    
    rewritten_query = response.choices[0].message.content.strip()
    return json.dumps({"Rewritten Query": rewritten_query})
   
def ollama_chat(user_input, system_message, vault_embeddings, vault_content, ollama_model, conversation_history):
    conversation_history.append({"role": "user", "content": user_input})
    
    if len(conversation_history) > 1:
        query_json = {
            "Query": user_input,
            "Rewritten Query": ""
        }
        rewritten_query_json = rewrite_query(json.dumps(query_json), conversation_history, ollama_model)
        rewritten_query_data = json.loads(rewritten_query_json)
        rewritten_query = rewritten_query_data["Rewritten Query"]
    else:
        rewritten_query = user_input
    
    relevant_context = get_relevant_context(rewritten_query, vault_embeddings, vault_content)
    if relevant_context:
        context_str = "\n".join(relevant_context)
        
    
    user_input_with_context = user_input
    if relevant_context:
        user_input_with_context = user_input + "\n\nRelevant Context:\n" + context_str
    
    conversation_history[-1]["content"] = user_input_with_context
    
    messages = [
        {"role": "system", "content": system_message},
        *conversation_history
    ]
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")    
    response = client.chat.completions.create(
        # model=ollama_model,
        model="model-identifier",
        messages=messages,
        # max_tokens=2000,
    )
    res= str(response.choices[0].message.content)
    start_index = res.find('{')
    end_index = res.rfind('}')
    json_string = res[start_index:end_index+1]
    json_string_main = json_string.strip()

    conversation_history.append({"role": "assistant", "content": response.choices[0].message.content})    
    return json_string_main

parser = argparse.ArgumentParser(description="Ollama Chat")
parser.add_argument("--model", default="llama2", help="Ollama model to use (default: llama2)")
args = parser.parse_args()


conversation_history = []
system_message = "You are a helpful assistant that is an expert at extracting the most useful information from a given text. Also bring in extra relevant infromation to the user query from outside the given context."


###################################################################
###################################################################
# RAGApp code ending
###################################################################
###################################################################

# Api Gateways
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

@app.route('/chatllm', methods=['POST'])
def handle_chatllm():
    # Assuming the PDF file is sent as form data in the request
    pmt = request.form['prompt']  

    # Parse the resume and return the structured JSON
    json_output = chatllm(pmt)

    return json_output

@app.route('/rag_uploadPDF', methods=['POST'])
def handle_rag_uploadPDF():
    # Assuming the PDF file is sent as form data in the request
    file = request.files['doc']    
    _, file_extension = os.path.splitext(file.filename)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
    file.save(temp_file.name)

    # Parse the doc    
    json_output = convert_pdf_to_text(temp_file.name)    
    # Clean up - close and delete the temporary file
    temp_file.close()
    os.unlink(temp_file.name)
    return json_output

@app.route('/rag', methods=['POST'])
def handle_rag(): 
    user_input = request.form['prompt'] + ' - Please provide the answer in JSON format, ensuring it is not null.'

    id = str(request.form['fileid'])
    if not id:
        return jsonify({"error": "ID is required"}), 400
    
    vault_folder = "vaultFiles\\"    
    vault_filename = os.path.join(vault_folder, f"{id}_vault.txt")
    if not os.path.exists(vault_filename):
        return jsonify({"error": "Incorrect ID"}), 400
    
    vault_content = []
    vault_embeddings = []
    if os.path.exists(vault_filename):
        with open(vault_filename, "r", encoding='utf-8') as vault_file:
            vault_content = vault_file.readlines()
       
    for content in vault_content:
        response = ollama.embeddings(model='mxbai-embed-large', prompt=content)
        vault_embeddings.append(response["embedding"]) 
    vault_embeddings_tensor = torch.tensor(vault_embeddings)

    conversation_history = []
    # print(conversation_history)
    response = ollama_chat(user_input, system_message, vault_embeddings_tensor, vault_content, args.model, conversation_history)
    # print(NEON_GREEN + "Response: \n\n" + response + RESET_COLOR)
    return response

if __name__ == "__main__":
    app.run(debug=True)
