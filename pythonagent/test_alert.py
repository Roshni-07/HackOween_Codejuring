from api_budget import APIBudgetTracker

# Create a fresh tracker with a $40 budget
tracker = APIBudgetTracker(
    budget=40.0,
    alert_limit=30.0
)

# Simulate API usage costing $31
sample_usage = {
    "input_tokens": 0,
    "output_tokens": 0,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0
}

# Directly simulate spending for this alert test
tracker.spent = 29.00

# Add usage worth $2
sample_usage["input_tokens"] = 666667

tracker.record_usage(sample_usage)

print("\nAlert test completed!")