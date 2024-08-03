import os
import logging
from typing import List, Tuple
import numpy as np
from google.cloud import storage
import pyarrow.feather as feather
import tempfile
from quart import Quart, request, jsonify
from threading import Lock
from datetime import timedelta
import aiohttp
import asyncio
import threading
import multiprocessing
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Quart(__name__)

storage_client = storage.Client()
dataset_cache = {}
cache_lock = Lock()

async def download_dataset(dataset_id: str) -> bytes:
    bucket_name = "ai-drive-psg-2024-us-central1-public"
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(dataset_id)
    
    expiration = timedelta(minutes=15)  # Set the expiration time to 15 minutes
    signed_url = blob.generate_signed_url(expiration=expiration)
    
    async with aiohttp.ClientSession() as session:
        async with session.get(signed_url) as resp:
            return await resp.read()

async def download_and_load_dataset(dataset_id: str) -> Tuple[np.ndarray, List[str]]:
    with cache_lock:
        if dataset_id in dataset_cache:
            return dataset_cache[dataset_id]

    try:
        file_content = await download_dataset(dataset_id)
        
        # Create a temporary file and write the content
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name
        
        logger.info(f"Temporary file created at: {temp_file_path}")
        
        # Read the Feather file directly into a DataFrame
        df = feather.read_feather(temp_file_path)
        
        # Ensure the DataFrame contains the necessary columns for vectors
        if df.empty or df.shape[1] == 0:
            raise ValueError("Dataset is empty or does not contain vector data.")
        
        vectors = np.vstack(df.iloc[:, 0].values)
        
        # Normalize if necessary
        norms = np.linalg.norm(vectors, axis=1)
        if not np.allclose(norms, 1.0):
            vectors = vectors / norms[:, np.newaxis]
        
        with cache_lock:
            dataset_cache[dataset_id] = (vectors, df.columns.tolist())
        
        # Clean up the temporary file
        os.unlink(temp_file_path)
        logger.info(f"Temporary file removed: {temp_file_path}")
        
        return vectors, df.columns.tolist()
    except Exception as e:
        logger.error(f"Error loading dataset {dataset_id}: {str(e)}")
        logger.exception("Traceback:")
        raise Exception("Internal server error") from e

def knn_search(vectors: np.ndarray, query_vector: np.ndarray, k: int = 10) -> List[int]:
    """
    Perform KNN search using cosine similarity (dot product)
    """
    # Ensure the query vector is normalized
    query_vector = query_vector / np.linalg.norm(query_vector)
    
    # Compute dot products between the query vector and all vectors
    dot_products = np.dot(vectors, query_vector)
    
    # Get the indices of the top k vectors with the highest dot products
    top_k_indices = np.argsort(dot_products)[::-1][:k]
    
    return top_k_indices.tolist()

def process_knn_search(vectors: np.ndarray, query_vector: np.ndarray, k: int) -> List[int]:
    return knn_search(vectors, query_vector, k)

@app.route("/KNN_search", methods=["POST"])
async def knn_search_endpoint():
    try:
        request_data = await request.get_json()
        dataset_id = request_data['dataset_id']
        query_vector = np.array(request_data['query_vector'], dtype=np.float32)
        
        logger.info(f"Received request for dataset: {dataset_id}")
        
        vectors, _ = await download_and_load_dataset(dataset_id)
        
        if query_vector.shape[0] != vectors.shape[1]:
            return jsonify({"detail": f"Query vector dimension mismatch. Expected {vectors.shape[1]}, got {query_vector.shape[0]}"}), 400

        # Use asynchronous function to perform KNN search
        loop = asyncio.get_event_loop()
        nearest_neighbors = await loop.run_in_executor(None, process_knn_search, vectors, query_vector, 10)
        
        logger.info(f"KNN search completed for dataset: {dataset_id}")
        return jsonify({"nearest_neighbor": {"indices": nearest_neighbors}})
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        logger.exception("Traceback:")
        return jsonify({"detail": "Internal server error"}), 500

@app.route("/threaded_KNN_search", methods=["POST"])
async def threaded_knn_search_endpoint():
    try:
        request_data = await request.get_json()
        dataset_id = request_data['dataset_id']
        query_vector = np.array(request_data['query_vector'], dtype=np.float32)

        logger.info(f"Received request for dataset: {dataset_id}")

        # Start a new thread for downloading the dataset
        def load_dataset():
            return asyncio.run(download_and_load_dataset(dataset_id))

        thread = threading.Thread(target=load_dataset)
        thread.start()
        thread.join()  # Wait for the thread to finish

        vectors, _ = dataset_cache[dataset_id]

        if query_vector.shape[0] != vectors.shape[1]:
            return jsonify({"detail": f"Query vector dimension mismatch. Expected {vectors.shape[1]}, got {query_vector.shape[0]}"}), 400

        # Perform KNN search
        nearest_neighbors = knn_search(vectors, query_vector)

        logger.info(f"KNN search completed for dataset: {dataset_id}")
        return jsonify({"nearest_neighbor": {"indices": nearest_neighbors}})
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        logger.exception("Traceback:")
        return jsonify({"detail": "Internal server error"}), 500

@app.route("/multiprocessing_KNN_search", methods=["POST"])
async def multiprocessing_knn_search_endpoint():
    try:
        request_data = await request.get_json()
        dataset_id = request_data['dataset_id']
        query_vector = np.array(request_data['query_vector'], dtype=np.float32)

        logger.info(f"Received request for dataset: {dataset_id}")

        vectors, _ = await download_and_load_dataset(dataset_id)

        if query_vector.shape[0] != vectors.shape[1]:
            return jsonify({"detail": f"Query vector dimension mismatch. Expected {vectors.shape[1]}, got {query_vector.shape[0]}"}), 400

        # Use multiprocessing to parallelize the KNN search
        def parallel_knn_search(vectors, query_vector, k):
            return knn_search(vectors, query_vector, k)

        with multiprocessing.Pool(processes=4) as pool:
            result = pool.apply_async(parallel_knn_search, args=(vectors, query_vector, 10))
            nearest_neighbors = result.get()

        logger.info(f"KNN search completed for dataset: {dataset_id}")
        return jsonify({"nearest_neighbor": {"indices": nearest_neighbors}})
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        logger.exception("Traceback:")
        return jsonify({"detail": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
