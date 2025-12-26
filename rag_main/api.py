from flask import Flask, request, jsonify
from rag_explainer import explain
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    query = data.get("query", "")
    answer = explain(query)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)
