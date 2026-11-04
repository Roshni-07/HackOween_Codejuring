from api_budget import APIBudgetTracker

tracker = APIBudgetTracker()

print("Can make API request?", tracker.can_make_request())

# Simulate Claude API usage
sample_usage = {
    "input_tokens": 100000,
    "output_tokens": 20000,
    "cache_creation_input_tokens": 5000,
    "cache_read_input_tokens": 10000
}

tracker.record_usage(sample_usage)

print("\nBudget test completed!")