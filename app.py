from flask import Flask, render_template, request, jsonify
from query_data import process_query
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    user_query = data.get('query', '')
    
    # Process the query using our existing RAG system
    response = process_query(user_query)
    
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 