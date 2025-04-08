import os
import cv2
import base64
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, util
from groq import Groq
from io import BytesIO
from PIL import Image
import faiss

# Load environment variables

app = Flask(__name__)
CORS(app)

# ================ MODEL 1: FAISS-based Model ================
# Paths to the saved FAISS index and metadata
faiss_index_path = "fruits_and_vegetables_dataset_index.faiss"
metadata_path = "fruits_and_vegetables_metadata.npy"

# Load the FAISS index and metadata if files exist
def load_faiss_index_and_metadata():
    try:
        index = faiss.read_index(faiss_index_path)
        metadata = np.load(metadata_path, allow_pickle=True).item()
        labels = metadata["labels"]
        image_paths = metadata["image_paths"]
        
        # Load the model
        model = SentenceTransformer("clip-ViT-L-14")
        
        # Recalibrate distance threshold
        D_train, _ = index.search(index.reconstruct_n(0, len(labels)), k=2)
        distances = D_train[:, 1]
        threshold = np.percentile(distances, 95)
        
        return index, labels, image_paths, model, threshold
    except Exception as e:
        print(f"Error loading FAISS model: {e}")
        return None, None, None, None, None

# Initialize FAISS model components
faiss_index, faiss_labels, image_paths, clip_model, threshold = load_faiss_index_and_metadata()

# Function to classify an image using FAISS
def classify_with_faiss(image):
    """Classifies an image using FAISS."""
    if None in (faiss_index, faiss_labels, clip_model, threshold):
        raise Exception("FAISS model not properly initialized")
        
    # Generate embedding for the test image
    test_embedding = clip_model.encode(image)
    test_embedding = test_embedding / np.linalg.norm(test_embedding)  # Normalize
    test_embedding = test_embedding.reshape(1, -1)  # Reshape for FAISS

    # Perform FAISS similarity search
    D, I = faiss_index.search(test_embedding.astype(np.float32), k=5)  # Retrieve top-5 matches

    # Check if the closest match exceeds the threshold
    if D[0][0] < threshold:
        predicted_label = faiss_labels[I[0][0]]
    else:
        predicted_label = "Unknown"

    return predicted_label, I[0][0]

# ================ MODEL 2: LLM-based Model ================
# Load Groq API key
client = Groq(api_key='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')

# Load labels from labels.txt
try:
    with open("labels.txt", "r", encoding="utf-8") as file:
        llm_labels = [line.strip() for line in file.readlines()]
except Exception as e:
    print(f"Error loading labels: {e}")
    llm_labels = []

# Load embedding model for similarity matching
try:
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    label_embeddings = embedding_model.encode(llm_labels, convert_to_tensor=True)
except Exception as e:
    print(f"Error loading embedding model: {e}")
    embedding_model = None
    label_embeddings = None

# Function to recognize fruit or vegetable using LLM
def recognize_with_llm(image):
    if not llm_labels:
        raise Exception("No labels loaded for LLM model")

    if embedding_model is None or label_embeddings is None:
        raise Exception("Embedding model not initialized for LLM model")

    # Convert image to base64
    _, buffer = cv2.imencode('.jpg', image)
    base64_image = base64.b64encode(buffer).decode("utf-8")

    label_list = ", ".join(llm_labels)

    # Groq model prompt
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                },
                {
                    "type": "text",
                    "text": f"Identify the fruit or vegetable in the image. Return only the name or the closest match from this list: {label_list}."
                }
            ]
        }
    ]

    try:
        completion = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=messages,
            temperature=0.3,
            max_tokens=50,
            top_p=1,
            stream=False,
        )
        prediction = completion.choices[0].message.content.strip()

        # Check if prediction is in labels
        if prediction in llm_labels:
            return prediction

        # If not in labels, find closest match
        pred_embedding = embedding_model.encode(prediction, convert_to_tensor=True)
        similarity_scores = util.pytorch_cos_sim(pred_embedding, label_embeddings)[0]
        best_match_idx = similarity_scores.argmax().item()
        best_match_score = similarity_scores[best_match_idx].item()

        if best_match_score > 0.7:
            return llm_labels[best_match_idx]
        else:
            return None
    except Exception as e:
        raise Exception(f"LLM model prediction failed: {e}")

# ================ Helper Functions ================
# Helper function to convert image formats
def convert_to_cv2_image(file):
    # Handle file object from request.files
    if hasattr(file, 'read'):
        file_data = file.read()
        return cv2.imdecode(np.frombuffer(file_data, np.uint8), cv2.IMREAD_COLOR)
    # Handle PIL Image
    elif isinstance(file, Image.Image):
        open_cv_image = np.array(file)
        # Convert RGB to BGR (OpenCV uses BGR)
        return open_cv_image[:, :, ::-1].copy()
    else:
        raise ValueError("Unsupported image format")

def convert_to_pil_image(file):
    # Handle file object from request.files
    if hasattr(file, 'read'):
        file_data = file.read()
        # Make a copy of the data if needed
        if isinstance(file, BytesIO):
            file.seek(0)
        return Image.open(BytesIO(file_data)).convert('RGB')
    # Handle OpenCV image
    elif isinstance(file, np.ndarray):
        return Image.fromarray(cv2.cvtColor(file, cv2.COLOR_BGR2RGB))
    else:
        raise ValueError("Unsupported image format")

# ================ API Endpoints ================
# API endpoint for Model 1 (FAISS)
@app.route("/api/upload", methods=["POST"])
def upload():
    try:
        # Check if the request contains a file
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400
        
        file = request.files['file']
        
        # Make a copy of the file data
        file_copy = BytesIO(file.read())
        file_copy.seek(0)
        file.seek(0)
        
        # Convert to PIL format for CLIP/FAISS
        try:
            pil_image = Image.open(file_copy).convert('RGB')
            
            # Get prediction
            predicted_label, closest_idx = classify_with_faiss(pil_image)
            closest_image = image_paths[closest_idx] if image_paths else None
            
            # Return in the format expected by the React component
            return jsonify({
                "fruits_vegetables": predicted_label,
                "closest_image": closest_image
            }), 200
        except Exception as e:
            print(f"FAISS model error: {str(e)}")
            return jsonify({"error": f"FAISS model error: {str(e)}"}), 500
            
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return jsonify({"error": str(e)}), 500

# API endpoint for Model 2 (LLM)
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Check if image file exists in request
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
            
        file = request.files['image']
        
        # Convert to CV2 format for the LLM model
        try:
            cv_image = convert_to_cv2_image(file)
            if cv_image is None:
                return jsonify({"error": "Invalid image format"}), 400
                
            # Get prediction from LLM model
            result = recognize_with_llm(cv_image)
            
            # Return prediction
            if result:
                return jsonify({"prediction": result})
            else:
                return jsonify({"prediction": "Unknown"})
        except Exception as e:
            print(f"LLM model error: {str(e)}")
            return jsonify({"error": f"LLM model error: {str(e)}"}), 500
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Home route
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

# Status endpoint to check model availability
@app.route("/api/status", methods=["GET"])
def status():
    faiss_available = None not in (faiss_index, faiss_labels, clip_model)
    llm_available = GROQ_API_KEY is not None and embedding_model is not None
    
    return jsonify({
        "faiss_model": "available" if faiss_available else "unavailable",
        "llm_model": "available" if llm_available else "unavailable"
    })

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)