import os
import unittest

from tests.llama_server_test_utils import (
    extract_token_count,
    extract_tokens_per_second,
    post_json,
    start_llama_server,
)


class LlamaServerSingleRequestTest(unittest.TestCase):
    def test_single_request_tokens_per_second(self):
        prompt = os.environ.get(
            "LLAMA_PROMPT",
            "Write a short paragraph about why concurrency helps throughput.",
        )
        n_predict = int(os.environ.get("LLAMA_N_PREDICT", "128"))

        with start_llama_server() as server:
            response = post_json(
                f"{server['base_url']}/completion",
                {
                    "prompt": prompt,
                    "n_predict": n_predict,
                    "temperature": 0.2,
                    "stream": False,
                },
            )

        token_count = extract_token_count(response)
        tokens_per_second = extract_tokens_per_second(response)

        self.assertGreater(token_count, 0, "Expected tokens in response.")
        self.assertGreater(tokens_per_second, 0.0, "Expected tokens per second > 0.")
        print(
            "single_request "
            f"tokens={token_count} tokens_per_second={tokens_per_second:.2f}"
        )


if __name__ == "__main__":
    unittest.main()
