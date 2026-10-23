from src.request_queue import RequestQueue
from src.engine_wrapper import LLMEngine
from src.metrics import MetricsTracker, RequestMetric

class ContinuousBatchScheduler:
    def __init__(self, engine: LLMEngine, queue: RequestQueue, metrics: MetricsTracker, max_batch_size: int = 8):
        self.engine = engine
        self.queue = queue
        self.metrics = metrics
        self.max_batch_size = max_batch_size

    def run(self):
        active_reqs = []
        active_metrics = {}

        while self.queue.has_pending() or active_reqs:
            self.queue.update_queue(self.engine.current_time)
            
            # 1. Fill empty slots in the active batch
            available_slots = self.max_batch_size - len(active_reqs)
            if available_slots > 0 and self.queue.waiting:
                new_batch = self.queue.waiting[:available_slots]
                self.queue.waiting = self.queue.waiting[available_slots:]
                
                for req in new_batch:
                    active_metrics[req.id] = RequestMetric(
                        request_id=req.id, arrival_time=req.arrival_time, start_time=self.engine.current_time
                    )
                
                # Prefill the newly injected sequences
                self.engine.prefill(new_batch)
                
                for req in new_batch:
                    active_metrics[req.id].first_token_time = self.engine.current_time
                    active_reqs.append(req)

            # 2. If nothing is active and nothing is waiting, advance time to next arrival
            if not active_reqs:
                if self.queue.incoming:
                    self.engine.current_time = self.queue.incoming[0].arrival_time
                continue

            # 3. Decode ONE step for all active sequences
            completed = self.engine.decode_step(active_reqs)
            
            # 4. Immediately evict finished sequences (Continuous Batching logic)
            for req in completed:
                metric = active_metrics[req.id]
                metric.end_time = self.engine.current_time
                metric.input_tokens = req.input_len
                metric.output_tokens = req.generated_tokens
                self.metrics.log_request(metric)
                
                active_reqs.remove(req)
                del active_metrics[req.id]