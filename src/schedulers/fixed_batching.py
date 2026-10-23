from src.request_queue import RequestQueue
from src.engine_wrapper import LLMEngine
from src.metrics import MetricsTracker, RequestMetric

class FixedBatchScheduler:
    def __init__(self, engine: LLMEngine, queue: RequestQueue, metrics: MetricsTracker, batch_size: int = 8):
        self.engine = engine
        self.queue = queue
        self.metrics = metrics
        self.batch_size = batch_size

    def run(self):
        while self.queue.has_pending():
            self.queue.update_queue(self.engine.current_time)
            
            # Wait until we have exactly batch_size requests (or no more incoming)
            if len(self.queue.waiting) < self.batch_size and self.queue.incoming:
                # Fast-forward time to next arrival to prevent infinite CPU spinning
                self.engine.current_time = self.queue.incoming[0].arrival_time
                continue
            
            if not self.queue.waiting:
                continue

            # Take fixed batch
            batch = self.queue.waiting[:self.batch_size]
            self.queue.waiting = self.queue.waiting[self.batch_size:]
            
            active_metrics = {}
            for req in batch:
                active_metrics[req.id] = RequestMetric(
                    request_id=req.id, arrival_time=req.arrival_time, start_time=self.engine.current_time
                )

            # Prefill Phase
            self.engine.prefill(batch)
            for req in batch:
                active_metrics[req.id].first_token_time = self.engine.current_time

            # Decode Phase - strict fixed batch runs until ALL are done
            active_reqs = batch[:]
            while active_reqs:
                completed = self.engine.decode_step(active_reqs)
                for req in completed:
                    metric = active_metrics[req.id]
                    metric.end_time = self.engine.current_time
                    metric.input_tokens = req.input_len
                    metric.output_tokens = req.generated_tokens
                    self.metrics.log_request(metric)
                    active_reqs.remove(req)