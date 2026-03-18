import requests
import json
import time
import concurrent.futures

# Configuration
API_URL = "http://localhost:8000/v1/score"
MODEL_NAME = "qwen3-0-6b"
LABEL_TOKEN_IDS = [9693, 2152] 

# Benchmark Settings
NUM_REQUESTS = 20       # Total number of API calls for the benchmark
NUM_ITEMS = 80       # Number of items per request
QUERY_TOKENS = 1000     # Approximate token length of the query
ITEM_TOKENS = 26        # Approximate token length of each item
CONCURRENCY = 2        # Number of parallel requests

def generate_dummy_text(target_tokens):
    """
    Generates a string of approximately `target_tokens` length.
    Approximation: 1 "word" + space ~= 1 token (rough heuristic).
    """
    word = "test "
    multiplier = max(1, target_tokens)
    return (word * multiplier).strip()

def prepare_payload():
    """Pre-generates the static payload to avoid measuring client-side generation time."""
    print("🔹 Generating dummy payload data...")
    query_text = generate_dummy_text(QUERY_TOKENS)
    
    # Generate unique items to ensure the server processes them individually
    items_list = [f"{i}_{generate_dummy_text(ITEM_TOKENS)}" for i in range(NUM_ITEMS)]
    
    return {
        "query": query_text,
        "items": items_list,
        "label_token_ids": LABEL_TOKEN_IDS,
        "model": MODEL_NAME,
        "apply_softmax": True
    }

def send_request(session, payload):
    """
    Sends a single request using a shared session.
    Returns True if successful, False otherwise.
    """
    headers = {"Content-Type": "application/json"}
    try:
        response = session.post(API_URL, json=payload, headers=headers)
        response.raise_for_status()
        return True
    except Exception as e:
        print(f"❌ Request Failed: {e}")
        return False

def run_benchmark():
    # 1. Prepare Data
    payload = prepare_payload()
    
    # Calculate approximate size for info
    payload_json = json.dumps(payload)
    payload_size_mb = len(payload_json) / (1024 * 1024)
    
    print(f"\n--- Configuration ---")
    print(f"  Requests:       {NUM_REQUESTS}")
    print(f"  Items/Req:      {NUM_ITEMS}")
    print(f"  Query Tokens:   ~{QUERY_TOKENS}")
    print(f"  Item Tokens:    ~{ITEM_TOKENS}")
    print(f"  Concurrency:    {CONCURRENCY}")
    print(f"  Payload Size:   ~{payload_size_mb:.2f} MB")

    # 2. Warm-up Phase
    print(f"\n🔥 Warming up (sending 1 request)...")
    try:
        # We use a session to reuse TCP connections (keep-alive)
        with requests.Session() as session:
            start_warmup = time.perf_counter()
            success = send_request(session, payload)
            warmup_time = time.perf_counter() - start_warmup
            
            if success:
                print(f"✅ Warm-up complete in {warmup_time:.4f}s")
            else:
                print(f"❌ Warm-up failed. Aborting benchmark.")
                return
    except Exception as e:
        print(f"❌ Warm-up crashed: {e}")
        return

    # 3. Execution Phase
    print(f"\n🚀 Starting Benchmark ({NUM_REQUESTS} requests)...")
    
    successful_requests = 0
    start_time = time.perf_counter()
    
    with requests.Session() as session:
        # Using ThreadPoolExecutor to limit concurrency to 4
        with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
            # Submit all tasks to the executor
            futures = [executor.submit(send_request, session, payload) for _ in range(NUM_REQUESTS)]
            
            # Wait for completion and count successes as they finish
            for future in concurrent.futures.as_completed(futures):
                if future.result():
                    successful_requests += 1

    end_time = time.perf_counter()
    total_time = end_time - start_time

    # 4. Calculation & Reporting
    if successful_requests == 0:
        print("❌ All benchmark requests failed.")
        return

    rps = successful_requests / total_time
    total_items_processed = successful_requests * NUM_ITEMS
    items_per_second = total_items_processed / total_time
    
    print("\n" + "="*40)
    print(f"📊 BENCHMARK RESULTS")
    print("="*40)
    print(f"Total Time:       {total_time:.4f} s")
    print(f"Success Rate:     {successful_requests}/{NUM_REQUESTS}")
    print("-" * 20)
    print(f"RPS (Requests/s): {rps:.2f}")
    print(f"IPS (Items/s):    {items_per_second:.2f}")
    print("="*40)

if __name__ == "__main__":
    run_benchmark()