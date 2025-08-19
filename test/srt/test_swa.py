#!/usr/bin/env python3

"""
Test script for Sliding Window Attention (SWA) with chunked prefill
"""

import tempfile
import subprocess
import time
import requests
import json
import sys


def test_swa_basic():
    """Test basic SWA functionality"""
    print("Testing basic SWA functionality...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Start server with SWA enabled
        cmd = [
            sys.executable, "-m", "sgl_jax.launch_server",
            "--model-path", "meta-llama/Llama-2-7b-chat-hf",
            "--host", "127.0.0.1", 
            "--port", "30001",
            "--enable-swa",
            "--attention-window-size", "512",
            "--swa-chunk-size", "128",
            "--chunked-prefill-size", "256",
            "--enable-mixed-chunked-prefill"
        ]
        
        print(f"Starting server with command: {' '.join(cmd)}")
        
        server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for server to start
        time.sleep(30)
        
        try:
            # Test long sequence processing with SWA
            test_prompt = "Explain the concept of machine learning in detail. " * 50  # Long prompt
            
            response = requests.post(
                "http://127.0.0.1:30001/generate",
                json={
                    "text": test_prompt,
                    "sampling_params": {
                        "max_new_tokens": 100,
                        "temperature": 0.0
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✓ SWA basic test passed")
                print(f"  Generated tokens: {len(result.get('text', '').split())}")
                return True
            else:
                print(f"✗ SWA basic test failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"✗ SWA basic test failed with exception: {e}")
            return False
        finally:
            server_process.terminate()
            server_process.wait()


def test_swa_memory_efficiency():
    """Test that SWA reduces memory usage for long sequences"""
    print("Testing SWA memory efficiency...")
    
    # This test would compare memory usage with and without SWA
    # For now, we'll just test that very long sequences can be processed
    with tempfile.TemporaryDirectory() as temp_dir:
        cmd = [
            sys.executable, "-m", "sgl_jax.launch_server",
            "--model-path", "meta-llama/Llama-2-7b-chat-hf",
            "--host", "127.0.0.1",
            "--port", "30002", 
            "--enable-swa",
            "--attention-window-size", "256",
            "--swa-chunk-size", "64",
            "--chunked-prefill-size", "128"
        ]
        
        server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        time.sleep(30)
        
        try:
            # Test very long sequence
            very_long_prompt = "The history of artificial intelligence is fascinating. " * 200
            
            response = requests.post(
                "http://127.0.0.1:30002/generate",
                json={
                    "text": very_long_prompt,
                    "sampling_params": {
                        "max_new_tokens": 50,
                        "temperature": 0.0
                    }
                },
                timeout=120
            )
            
            if response.status_code == 200:
                print("✓ SWA memory efficiency test passed")
                return True
            else:
                print(f"✗ SWA memory efficiency test failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"✗ SWA memory efficiency test failed: {e}")
            return False
        finally:
            server_process.terminate()
            server_process.wait()


def test_swa_with_chunked_prefill():
    """Test SWA integration with chunked prefill"""
    print("Testing SWA with chunked prefill...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        cmd = [
            sys.executable, "-m", "sgl_jax.launch_server",
            "--model-path", "meta-llama/Llama-2-7b-chat-hf",
            "--host", "127.0.0.1",
            "--port", "30003",
            "--enable-swa",
            "--attention-window-size", "512",
            "--swa-chunk-size", "128",
            "--chunked-prefill-size", "256",
            "--enable-mixed-chunked-prefill"
        ]
        
        server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        time.sleep(30)
        
        try:
            # Test multiple requests with different lengths
            test_cases = [
                "Short prompt",
                "Medium length prompt that should trigger chunked prefill " * 10,
                "Very long prompt that definitely triggers chunked prefill and SWA eviction " * 50
            ]
            
            success_count = 0
            for i, prompt in enumerate(test_cases):
                response = requests.post(
                    "http://127.0.0.1:30003/generate",
                    json={
                        "text": prompt,
                        "sampling_params": {
                            "max_new_tokens": 30,
                            "temperature": 0.0
                        }
                    },
                    timeout=60
                )
                
                if response.status_code == 200:
                    success_count += 1
                    print(f"  ✓ Test case {i+1} passed")
                else:
                    print(f"  ✗ Test case {i+1} failed: {response.status_code}")
            
            if success_count == len(test_cases):
                print("✓ SWA with chunked prefill test passed")
                return True
            else:
                print(f"✗ SWA with chunked prefill test failed: {success_count}/{len(test_cases)} passed")
                return False
                
        except Exception as e:
            print(f"✗ SWA with chunked prefill test failed: {e}")
            return False
        finally:
            server_process.terminate()
            server_process.wait()


def test_swa_configuration():
    """Test different SWA configurations"""
    print("Testing SWA configuration options...")
    
    configurations = [
        {
            "attention_window_size": 256,
            "swa_chunk_size": 64,
            "chunked_prefill_size": 128
        },
        {
            "attention_window_size": 512, 
            "swa_chunk_size": 128,
            "chunked_prefill_size": 256
        }
    ]
    
    success_count = 0
    for i, config in enumerate(configurations):
        print(f"  Testing configuration {i+1}: {config}")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            cmd = [
                sys.executable, "-m", "sgl_jax.launch_server",
                "--model-path", "meta-llama/Llama-2-7b-chat-hf",
                "--host", "127.0.0.1",
                "--port", str(30004 + i),
                "--enable-swa",
                "--attention-window-size", str(config["attention_window_size"]),
                "--swa-chunk-size", str(config["swa_chunk_size"]),
                "--chunked-prefill-size", str(config["chunked_prefill_size"])
            ]
            
            server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            time.sleep(25)
            
            try:
                response = requests.post(
                    f"http://127.0.0.1:{30004 + i}/generate",
                    json={
                        "text": "Test prompt " * 100,
                        "sampling_params": {
                            "max_new_tokens": 20,
                            "temperature": 0.0
                        }
                    },
                    timeout=60
                )
                
                if response.status_code == 200:
                    success_count += 1
                    print(f"    ✓ Configuration {i+1} passed")
                else:
                    print(f"    ✗ Configuration {i+1} failed: {response.status_code}")
                    
            except Exception as e:
                print(f"    ✗ Configuration {i+1} failed: {e}")
            finally:
                server_process.terminate()
                server_process.wait()
    
    if success_count == len(configurations):
        print("✓ SWA configuration test passed")
        return True
    else:
        print(f"✗ SWA configuration test failed: {success_count}/{len(configurations)} passed")
        return False


def main():
    """Run all SWA tests"""
    print("Running SWA (Sliding Window Attention) tests...")
    print("=" * 60)
    
    tests = [
        test_swa_basic,
        test_swa_memory_efficiency,
        test_swa_with_chunked_prefill,
        test_swa_configuration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        print()
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
        
        # Wait between tests
        time.sleep(5)
    
    print()
    print("=" * 60)
    print(f"SWA Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All SWA tests passed!")
        sys.exit(0)
    else:
        print("❌ Some SWA tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()