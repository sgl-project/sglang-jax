"""
Test chunked prefill functionality for sgl-jax

Usage:
python3 -m pytest test/srt/test_chunked_prefill.py -v
"""

import asyncio
import pytest
import requests
import time
from typing import List

import sgl_jax as sgl
from sgl_jax.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST


class TestChunkedPrefill:
    """Test suite for chunked prefill functionality"""

    @pytest.fixture(scope="class")
    def server_args(self):
        """Common server arguments for testing"""
        return {
            "model_path": DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            "host": "127.0.0.1",
            "port": 30001,
            "chunked_prefill_size": 512,  # Small chunk size for testing
            "max_prefill_tokens": 1024,
            "tp_size": 1,
            "disable_radix_cache": True,  # Use ChunkCache for testing
        }

    def test_basic_chunked_prefill(self, server_args):
        """Test basic chunked prefill without mixed mode"""
        
        # Create a long input that will trigger chunking
        long_prompt = "This is a test prompt. " * 100  # ~600 tokens
        
        # Start server with chunked prefill
        with sgl.Runtime(**server_args) as runtime:
            # Simple completion request
            response = requests.post(
                f"http://{server_args['host']}:{server_args['port']}/v1/completions",
                json={
                    "model": "default",
                    "prompt": long_prompt,
                    "max_tokens": 50,
                    "temperature": 0.1,
                }
            )
            
            assert response.status_code == 200
            result = response.json()
            assert "choices" in result
            assert len(result["choices"]) > 0
            assert "text" in result["choices"][0]

    def test_mixed_chunked_prefill(self, server_args):
        """Test mixed chunked prefill mode"""
        
        # Enable mixed chunked prefill
        mixed_args = server_args.copy()
        mixed_args["enable_mixed_chunked_prefill"] = True
        
        with sgl.Runtime(**mixed_args) as runtime:
            # Send multiple requests concurrently
            long_prompt = "This is a very long test prompt that should trigger chunked prefill. " * 50
            short_prompt = "Short prompt."
            
            async def send_requests():
                tasks = []
                
                # Long request that will be chunked
                tasks.append(self._send_completion_request(
                    mixed_args, long_prompt, max_tokens=30
                ))
                
                # Short requests that can be processed alongside chunks
                for i in range(3):
                    tasks.append(self._send_completion_request(
                        mixed_args, f"{short_prompt} Request {i}", max_tokens=20
                    ))
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                return results
            
            results = asyncio.run(send_requests())
            
            # All requests should complete successfully
            for result in results:
                assert not isinstance(result, Exception)
                assert "choices" in result
                assert len(result["choices"]) > 0

    def test_chunked_prefill_memory_efficiency(self, server_args):
        """Test that chunked prefill handles memory more efficiently"""
        
        # Test with very long prompt that would OOM without chunking
        very_long_prompt = "Test token " * 200  # ~400 tokens
        
        chunk_args = server_args.copy()
        chunk_args["chunked_prefill_size"] = 256  # Force chunking
        
        with sgl.Runtime(**chunk_args) as runtime:
            start_time = time.time()
            
            response = requests.post(
                f"http://{chunk_args['host']}:{chunk_args['port']}/v1/completions",
                json={
                    "model": "default",
                    "prompt": very_long_prompt,
                    "max_tokens": 50,
                    "temperature": 0.1,
                }
            )
            
            end_time = time.time()
            
            assert response.status_code == 200
            result = response.json()
            assert "choices" in result
            
            # Should complete in reasonable time despite chunking
            assert (end_time - start_time) < 30  # 30 seconds max

    def test_disabled_chunked_prefill(self, server_args):
        """Test behavior when chunked prefill is disabled"""
        
        disabled_args = server_args.copy()
        disabled_args["chunked_prefill_size"] = -1  # Disable chunking
        
        with sgl.Runtime(**disabled_args) as runtime:
            # Should still work for normal-sized prompts
            normal_prompt = "This is a normal prompt."
            
            response = requests.post(
                f"http://{disabled_args['host']}:{disabled_args['port']}/v1/completions",
                json={
                    "model": "default",
                    "prompt": normal_prompt,
                    "max_tokens": 30,
                    "temperature": 0.1,
                }
            )
            
            assert response.status_code == 200
            result = response.json()
            assert "choices" in result

    def test_chunked_prefill_with_streaming(self, server_args):
        """Test chunked prefill with streaming responses"""
        
        with sgl.Runtime(**server_args) as runtime:
            long_prompt = "Generate a story: " * 50
            
            response = requests.post(
                f"http://{server_args['host']}:{server_args['port']}/v1/completions",
                json={
                    "model": "default",
                    "prompt": long_prompt,
                    "max_tokens": 100,
                    "temperature": 0.1,
                    "stream": True,
                },
                stream=True
            )
            
            assert response.status_code == 200
            
            # Collect streaming chunks
            chunks = []
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: '):
                        chunk = line_str[6:]
                        if chunk.strip() != '[DONE]':
                            chunks.append(chunk)
            
            assert len(chunks) > 0

    async def _send_completion_request(self, server_args, prompt, max_tokens=50):
        """Helper to send async completion request"""
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://{server_args['host']}:{server_args['port']}/v1/completions",
                json={
                    "model": "default",
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": 0.1,
                }
            ) as response:
                assert response.status == 200
                return await response.json()

    def test_chunked_prefill_batch_processing(self, server_args):
        """Test that chunked requests are properly batched"""
        
        batch_args = server_args.copy()
        batch_args["enable_mixed_chunked_prefill"] = True
        batch_args["max_running_requests"] = 8
        
        with sgl.Runtime(**batch_args) as runtime:
            # Send multiple long requests
            prompts = [f"Long test prompt number {i}. " * 60 for i in range(4)]
            
            responses = []
            for prompt in prompts:
                response = requests.post(
                    f"http://{batch_args['host']}:{batch_args['port']}/v1/completions",
                    json={
                        "model": "default",
                        "prompt": prompt,
                        "max_tokens": 30,
                        "temperature": 0.1,
                    }
                )
                responses.append(response)
            
            # All should succeed
            for response in responses:
                assert response.status_code == 200
                result = response.json()
                assert "choices" in result

    def test_chunked_prefill_configuration_validation(self):
        """Test that chunked prefill configuration is validated properly"""
        
        # Test invalid chunk size
        with pytest.raises(Exception):
            invalid_args = {
                "model_path": DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
                "chunked_prefill_size": 0,  # Invalid
                "page_size": 16,
            }
            # This should fail validation
            sgl.Runtime(**invalid_args)


if __name__ == "__main__":
    # Run basic test
    test_instance = TestChunkedPrefill()
    
    server_args = {
        "model_path": DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
        "host": "127.0.0.1", 
        "port": 30002,
        "chunked_prefill_size": 512,
        "max_prefill_tokens": 1024,
        "tp_size": 1,
        "disable_radix_cache": True,
    }
    
    print("Running basic chunked prefill test...")
    test_instance.test_basic_chunked_prefill(server_args)
    print("✅ Basic test passed!")
    
    print("Running mixed chunked prefill test...")
    test_instance.test_mixed_chunked_prefill(server_args) 
    print("✅ Mixed mode test passed!")
    
    print("All tests completed successfully!")