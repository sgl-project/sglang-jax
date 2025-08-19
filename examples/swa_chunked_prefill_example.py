#!/usr/bin/env python3

"""
Example demonstrating SWA (Sliding Window Attention) with Chunked Prefill in sgl-jax

This example shows how to start the server with SWA enabled and process long sequences efficiently.
"""

import subprocess
import sys
import time
import requests
import json


def start_server_with_swa():
    """Start the sgl-jax server with SWA and chunked prefill enabled"""
    
    cmd = [
        sys.executable, "-m", "sgl_jax.launch_server",
        
        # Model configuration
        "--model-path", "meta-llama/Llama-2-7b-chat-hf",
        "--host", "127.0.0.1",
        "--port", "30000",
        
        # SWA (Sliding Window Attention) configuration
        "--enable-swa",                      # Enable SWA
        "--attention-window-size", "1024",   # Size of sliding window for attention
        "--swa-chunk-size", "256",           # Chunk size for SWA eviction
        
        # Chunked Prefill configuration  
        "--chunked-prefill-size", "512",     # Size of prefill chunks
        "--enable-mixed-chunked-prefill",    # Allow mixing prefill and decode in same batch
        
        # Memory and performance settings
        "--max-prefill-tokens", "2048",      # Maximum tokens for prefill
        "--page-size", "1",                  # Page size for memory management
        "--disable-radix-cache",             # Use ChunkCache instead of RadixCache
        
        # Other settings
        "--dtype", "bfloat16",
        "--stream-interval", "1"
    ]
    
    print("Starting sgl-jax server with SWA and chunked prefill...")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )


def test_long_sequence_processing(base_url="http://127.0.0.1:30000"):
    """Test processing of long sequences with SWA"""
    
    # Create a very long prompt to test chunked prefill and SWA
    long_prompt = """
    Artificial Intelligence (AI) is a revolutionary field of computer science that aims to create 
    machines capable of performing tasks that typically require human intelligence. The concept of 
    AI has evolved significantly since its inception in the 1950s, when pioneers like Alan Turing 
    and John McCarthy laid the groundwork for what would become one of the most transformative 
    technologies of our time.
    
    The history of AI can be divided into several distinct periods, each characterized by different 
    approaches and levels of optimism. The early years, often referred to as the "Golden Age" of AI 
    (1956-1974), were marked by tremendous enthusiasm and ambitious goals. Researchers believed that 
    human-level AI was just around the corner, and significant investments were made in AI research.
    
    During this period, fundamental concepts such as neural networks, expert systems, and natural 
    language processing were developed. The Dartmouth Conference in 1956, organized by John McCarthy, 
    Marvin Minsky, Nathaniel Rochester, and Claude Shannon, is widely considered the birth of AI as 
    a field of study. This conference brought together researchers who shared a vision of creating 
    machines that could simulate human intelligence.
    
    However, the initial optimism soon gave way to what became known as the "AI Winter" (1974-1980), 
    a period characterized by reduced funding and diminished interest in AI research. This downturn 
    occurred because many of the early promises of AI had not been fulfilled, and the limitations 
    of the existing approaches became apparent. The computational power available at the time was 
    insufficient for the complex tasks that researchers were attempting to solve.
    
    The field experienced a renaissance in the 1980s with the emergence of expert systems, which 
    were designed to emulate the decision-making abilities of human experts in specific domains. 
    Companies began to see the commercial potential of AI, leading to increased investment and the 
    establishment of AI-focused companies. However, this period was followed by another AI Winter 
    in the late 1980s and early 1990s, as expert systems proved to be brittle and difficult to 
    maintain.
    
    The modern era of AI began in the 1990s and has been characterized by more pragmatic approaches 
    and steady progress rather than grand promises. The development of machine learning algorithms, 
    particularly those based on statistical methods, marked a significant shift in the field. 
    Researchers began to focus on creating systems that could learn from data rather than being 
    explicitly programmed with rules.
    """ * 3  # Repeat to make it very long
    
    print("Testing long sequence processing with SWA...")
    print(f"Prompt length: {len(long_prompt)} characters")
    
    try:
        response = requests.post(
            f"{base_url}/generate",
            json={
                "text": long_prompt,
                "sampling_params": {
                    "max_new_tokens": 200,
                    "temperature": 0.1,
                    "top_p": 0.9
                }
            },
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            generated_text = result.get("text", "")
            print("✓ Long sequence processing successful!")
            print(f"Generated text length: {len(generated_text)} characters")
            print(f"Generated text preview: {generated_text[:200]}...")
            return True
        else:
            print(f"✗ Request failed with status {response.status_code}: {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ Request failed with exception: {e}")
        return False


def test_streaming_with_swa(base_url="http://127.0.0.1:30000"):
    """Test streaming generation with SWA"""
    
    print("\nTesting streaming generation with SWA...")
    
    prompt = "Explain the benefits of sliding window attention in large language models: " * 20
    
    try:
        response = requests.post(
            f"{base_url}/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "max_new_tokens": 150,
                    "temperature": 0.2,
                    "stream": True
                }
            },
            stream=True,
            timeout=120
        )
        
        if response.status_code == 200:
            print("✓ Streaming request initiated successfully!")
            
            tokens_received = 0
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode('utf-8'))
                        if 'text' in data:
                            tokens_received += 1
                            if tokens_received <= 5:  # Show first few tokens
                                print(f"  Received token {tokens_received}: {data['text'][:50]}...")
                    except:
                        continue
            
            print(f"✓ Streaming completed! Received {tokens_received} token updates")
            return True
        else:
            print(f"✗ Streaming request failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ Streaming request failed with exception: {e}")
        return False


def test_multiple_concurrent_requests(base_url="http://127.0.0.1:30000"):
    """Test multiple concurrent requests with SWA"""
    
    print("\nTesting multiple concurrent requests with SWA...")
    
    import threading
    import time
    
    results = []
    
    def make_request(request_id, prompt_length_multiplier):
        base_prompt = "Generate a detailed explanation about machine learning concepts. " * prompt_length_multiplier
        
        try:
            response = requests.post(
                f"{base_url}/generate",
                json={
                    "text": base_prompt,
                    "sampling_params": {
                        "max_new_tokens": 50,
                        "temperature": 0.3
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                results.append(f"✓ Request {request_id} (length {prompt_length_multiplier}x) succeeded")
            else:
                results.append(f"✗ Request {request_id} failed with status {response.status_code}")
                
        except Exception as e:
            results.append(f"✗ Request {request_id} failed with exception: {e}")
    
    # Start multiple threads with different prompt lengths
    threads = []
    for i in range(3):
        thread = threading.Thread(
            target=make_request, 
            args=(i+1, (i+1) * 10)  # Different prompt lengths
        )
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Print results
    for result in results:
        print(f"  {result}")
    
    success_count = sum(1 for r in results if r.startswith("✓"))
    print(f"✓ Concurrent requests test: {success_count}/{len(results)} succeeded")
    
    return success_count == len(results)


def main():
    """Main example function"""
    
    print("SWA (Sliding Window Attention) with Chunked Prefill Example")
    print("=" * 60)
    
    # Start the server
    server_process = start_server_with_swa()
    
    try:
        print("Waiting for server to start...")
        time.sleep(30)  # Give server time to initialize
        
        # Run tests
        print("\nRunning example tests...")
        
        success_count = 0
        total_tests = 3
        
        if test_long_sequence_processing():
            success_count += 1
        
        if test_streaming_with_swa():
            success_count += 1
            
        if test_multiple_concurrent_requests():
            success_count += 1
        
        print(f"\n" + "=" * 60)
        print(f"Example Results: {success_count}/{total_tests} tests passed")
        
        if success_count == total_tests:
            print("🎉 All SWA example tests passed!")
            print("\nSWA with chunked prefill is working correctly!")
        else:
            print("❌ Some tests failed.")
            
        print(f"\n📝 Server logs (last 20 lines):")
        print("-" * 40)
        
        # Print some server logs
        server_process.poll()
        if server_process.returncode is None:
            try:
                stdout, stderr = server_process.communicate(timeout=1)
                if stderr:
                    print(stderr[-2000:])  # Last 2000 chars of logs
            except subprocess.TimeoutExpired:
                pass
        
    except KeyboardInterrupt:
        print("\n🛑 Example interrupted by user")
        
    finally:
        print("\n🔧 Shutting down server...")
        server_process.terminate()
        try:
            server_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server_process.kill()
            server_process.wait()
        print("✓ Server shutdown complete")


if __name__ == "__main__":
    main()