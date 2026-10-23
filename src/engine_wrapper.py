class LLMEngine:
    def __init__(self, max_batch_size: int = 16):
        self.max_batch_size = max_batch_size
        self.current_time = 0.0
        
        # Hardware simulation constants
        self.time_per_prefill_token = 0.0005  # 0.5ms per token in prompt
        self.time_per_decode_step = 0.02      # 20ms base decode time
        self.batch_penalty = 0.002            # 2ms extra per sequence in batch

    def advance_time(self, seconds: float):
        self.current_time += seconds

    def prefill(self, requests: list) -> float:
        """
        Simulates the prefill phase (processing the prompt).
        In a real setup: Call llama_decode on the prompt tokens and allocate KV cache.
        """
        if not requests:
            return 0.0
            
        total_input_tokens = sum(req.input_len for req in requests)
        prefill_time = total_input_tokens * self.time_per_prefill_token
        self.advance_time(prefill_time)
        return prefill_time

    def decode_step(self, active_requests: list) -> list:
        """
        Simulates one token generation step for all active sequences in the batch.
        In a real setup: Call llama_decode with 1 token per sequence, update KV cache.
        Returns a list of requests that have finished generating.
        """
        if not active_requests:
            return []

        # Time taken depends on how many sequences are being decoded simultaneously
        decode_time = self.time_per_decode_step + (len(active_requests) * self.batch_penalty)
        self.advance_time(decode_time)

        completed = []
        for req in active_requests:
            req.generated_tokens += 1
            if req.generated_tokens >= req.target_output_len:
                completed.append(req)
                
        return completed