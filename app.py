# app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import chatbot  # Import your chatbot module
import logging

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains (adjust as needed)

# Initialize the chatbot (ensure retriever is loaded)
# chatbot.chatbot is already initialized in chatbot.py

# Set up logging
logging.basicConfig(level=logging.INFO)

@app.route('/')
def home():
    return "RAG Secure Banking Chatbot API is running."

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        user_query = data.get("query", "")

        if not user_query:
            return jsonify({"error": "No query provided."}), 400

        # Get the chatbot's response
        response = chatbot.chatbot.ask(user_query)

        return jsonify({"answer": response}), 200

    except Exception as e:
        logging.exception("Error processing /ask request.")
        return jsonify({"error": str(e)}), 500  # Include error message in response
        # logging.exception("Error processing /ask request.")
        # return jsonify({"error": "An error occurred processing your request."}), 500

if __name__ == '__main__':
    # For development purposes only; use a proper WSGI server in production
    app.run(host='0.0.0.0', port=5000, debug=True)
