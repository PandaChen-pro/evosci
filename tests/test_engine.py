import json
import tempfile
import unittest
from pathlib import Path

from evosci.config import EvoSciConfig
from evosci.engine import EvoSciEngine


class EngineTests(unittest.TestCase):
    def test_two_round_end_to_end_run_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = EvoSciConfig.from_dict({
                "run": {
                    "rounds": 2,
                    "team_size": 3,
                    "ideas_per_round": 3,
                    "problem_count": 2,
                    "reviewer_count": 2,
                    "random_seed": 7,
                    "output_dir": directory,
                },
                "evolution": {"crossover_count": 1, "variation_count": 1},
            })
            engine = EvoSciEngine(config)
            state, run_dir = engine.run(
                "neural network grokking", ["computer science", "physics"]
            )
            self.assertEqual(len(state.rounds), 2)
            self.assertTrue(all(len(result.evaluated_ideas) == 3 for result in state.rounds))
            self.assertTrue((run_dir / "state.json").exists())
            self.assertTrue((run_dir / "graph.json").exists())
            self.assertTrue((run_dir / "report.md").exists())
            loaded = EvoSciEngine.load_state(run_dir)
            self.assertEqual(loaded.to_dict(), state.to_dict())

            resumed = EvoSciEngine.resume(run_dir)
            resumed_state, same_dir = resumed.run(
                loaded.topic, loaded.disciplines, run_dir=run_dir, state=loaded
            )
            self.assertEqual(same_dir, run_dir)
            self.assertEqual(len(resumed_state.rounds), 2)
            config_snapshot = json.loads((run_dir / "config.json").read_text())
            self.assertEqual(config_snapshot["run"]["rounds"], 2)


if __name__ == "__main__":
    unittest.main()
