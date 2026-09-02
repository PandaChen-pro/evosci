import io
import unittest

from evosci.llm import OpenAICompatibleBackend


class StreamingTests(unittest.TestCase):
    def test_chat_completion_stream_is_reassembled(self) -> None:
        stream = io.BytesIO(
            b'data: {"choices":[{"delta":{"content":"{\\"status\\":"}}]}\n\n'
            b'data: {"choices":[{"delta":{"content":"\\"ok\\"}"}}]}\n\n'
            b'data: [DONE]\n\n'
        )
        self.assertEqual(
            OpenAICompatibleBackend._read_stream(stream), '{"status":"ok"}'
        )


if __name__ == "__main__":
    unittest.main()
