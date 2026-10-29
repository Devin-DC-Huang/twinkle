import inspect
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from twinkle_agentic.async_rl.score_worker import ScoreWorker


class CompletionLoggingTest(unittest.TestCase):
    def test_thinking_only_sample_retains_prompt_and_reasoning(self):
        messages = [{'role': 'user', 'content': 'Question'},
                    {'role': 'assistant', 'content': '', 'reasoning_content': 'Still thinking'}]
        sample = SimpleNamespace(messages=messages, decoded='', response_token_ids=[1, 2], truncated=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'completions.jsonl'
            inspect.unwrap(ScoreWorker.log_completions)(None, [sample], [[0.0]], 1, str(path))
            record = json.loads(path.read_text())
        self.assertEqual(record['prompt'], messages[:1])
        self.assertEqual(record['reasoning_content'], 'Still thinking')
        self.assertEqual(record['completion'], '')
        self.assertEqual(record['messages'], messages)
        self.assertEqual(record['completion_length'], 2)
        self.assertTrue(record['truncated'])
        self.assertEqual(record['rewards'], [0.0])


if __name__ == '__main__':
    unittest.main()
