
import json
from pathlib import Path
from datetime import datetime, timezone


class APIBudgetTracker:

    def __init__(self, budget=40.0, alert_limit=30.0):

        self.budget = budget
        self.alert_limit = alert_limit

        # Claude 3.5 Sonnet standard rates per million tokens
        self.input_rate = 3.00
        self.output_rate = 15.00
        self.cache_write_rate = 3.75
        self.cache_read_rate = 0.30

        self.spent = 0.0

        self.log_dir = Path("Logs/api_budget")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = self.log_dir / "usage.jsonl"

        # Recover previous spending when restarting
        if self.log_file.exists():
            with open(self.log_file, "r", encoding="utf-8") as file:
                for line in file:
                    if line.strip():
                        entry = json.loads(line)
                        self.spent += entry["cost"]

    def calculate_cost(self, usage):

        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)

        cache_write = usage.get(
            "cache_creation_input_tokens", 0
        )

        cache_read = usage.get(
            "cache_read_input_tokens", 0
        )

        cost = (
            input_tokens * self.input_rate
            + output_tokens * self.output_rate
            + cache_write * self.cache_write_rate
            + cache_read * self.cache_read_rate
        ) / 1_000_000

        return cost

    def record_usage(self, usage):

        cost = self.calculate_cost(usage)
        self.spent += cost

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "usage": usage,
            "cost": round(cost, 8),
            "total_spent": round(self.spent, 8),
            "remaining": round(max(0, self.budget - self.spent), 8)
        }

        with open(self.log_file, "a", encoding="utf-8") as file:
            file.write(json.dumps(entry) + "\n")

        print(f"API cost: ${cost:.6f}")
        print(f"Total spent: ${self.spent:.4f}")
        print(f"Remaining: ${max(0, self.budget - self.spent):.4f}")

        if self.spent >= self.budget:
            print("BUDGET LIMIT REACHED! Stop new API requests.")

        elif self.spent >= self.alert_limit:
            print("WARNING: API spending crossed $30!")

        return entry

    def can_make_request(self):

        if self.spent >= self.budget:
            print("API request blocked: Budget exhausted.")
            return False

        return True

