from flask import Flask, request, jsonify, send_from_directory
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from openai import OpenAI
from get_embedding_function import get_embedding_function
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)
CHROMA_PATH = "chroma"
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context, if query is not relavant then give answer as  'Invalid query' : {question}
"""

def query_rag(query_text: str):
    try:
        # Prepare the DB.
        embedding_function = get_embedding_function()
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

        # Search the DB.
        results = db.similarity_search_with_score(query_text, k=5)

        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)

        client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
        response_text = client.chat.completions.create(
            model="model-identifier",
            messages=[{
                "role": "user",
                "content": prompt,
            }],
            temperature=0.7
        )

        docs = [doc.metadata.get("id", None) for doc, score in results]
        sources = [doc.metadata.get("id", None) for doc, score in results]
        print(docs)
        # print(sources)
        response_content = response_text.choices[0].message.content
        for doc, score in results:
            print(f"Document Score:{score}")
        # Extract file names from sources
        file_names = [source.split("\\")[-1].split(":")[0] for source in sources if source]
        # file_names = list(set(file_names))
        file_names = file_names[:1]  # Limit to the first two files
        return response_content, file_names
    except Exception as e:
        raise e

@app.route('/MultiRAGquery', methods=['POST'])
def MultiRAGquery():
    query_text = request.form.get('query')
    if not query_text:
        return jsonify({'error': 'No query_text provided'}), 400

    try:
        response_content, sources = query_rag(query_text)
        download_urls = [f"/download/{filename}" for filename in sources]
        return jsonify({'response': response_content, 'sources': download_urls})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<path:filename>', methods=['GET'])
def download_file(filename):
    try:
        directory = os.path.join(app.root_path, 'data')
        return send_from_directory(directory, filename, as_attachment=True)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)
