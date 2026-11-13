"""Exercise launcher functions locally, without importing/contacting Modal."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import tempfile
from unittest.mock import Mock, patch

LAUNCHER = Path(__file__).resolve().parents[1] / "modal_app.py"


def launcher_functions(**globals_):
    tree = ast.parse(LAUNCHER.read_text(encoding="utf-8"))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    for node in functions:
        node.decorator_list = []
    namespace = {"ROOT": "/vol", "Path": Path, **globals_}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(LAUNCHER), "exec"), namespace)
    return namespace


@unittest.skipUnless(LAUNCHER.exists(), "Launcher tests run in the local source checkout")
class ModalLauncherTests(unittest.TestCase):
    def test_reused_workers_refresh_before_reading_new_outputs(self):
        # CPU worker is already warm when another container commits evaluation.
        # A later invocation must observe that commit, including on GPU reuse.
        for worker in ("cpu_job", "gpu_job"):
            with self.subTest(worker=worker):
                durable, mounted, events = {}, {}, []
                def reload():
                    events.append("reload")
                    mounted.clear()
                    mounted.update(durable)
                def commit():
                    events.append("commit")
                    durable.update(mounted)
                namespace = launcher_functions(volume=SimpleNamespace(reload=reload, commit=commit))
                def child(args, cwd, env, commit_callback):
                    events.append("child")
                    self.assertEqual(mounted, durable)
                    if "task2-statistics" in args:
                        self.assertEqual(mounted["evaluation"], "verified results")
                    return 0
                with patch("pa3.process.run_with_commits", side_effect=child):
                    namespace[worker]("status", {})
                    durable["evaluation"] = "verified results"
                    namespace[worker]("task2-statistics", {})
                self.assertEqual(events.count("reload"), 2)
                for i, event in enumerate(events):
                    if event == "child":
                        self.assertEqual(events[i-1], "reload")

    def test_failed_refresh_never_launches_work(self):
        volume = SimpleNamespace(reload=Mock(side_effect=RuntimeError("storage unavailable")), commit=Mock())
        namespace = launcher_functions(volume=volume)
        with patch("pa3.process.run_with_commits") as child:
            with self.assertRaisesRegex(RuntimeError, "storage unavailable"):
                namespace["cpu_job"]("task2-statistics", {})
            child.assert_not_called()

    def test_remote_failure_is_propagated_to_cli(self):
        namespace = launcher_functions()
        namespace["orchestrate"] = SimpleNamespace(remote=Mock(side_effect=RuntimeError("statistics failed")))
        with self.assertRaisesRegex(RuntimeError, "statistics failed"):
            namespace["main"](command="all")

    def test_statistics_reuses_saved_predictions_without_gpu(self):
        from pa3.config import IMAGE_CONDITIONS
        from pa3.io import Store, write_parquet, read_json, sha256
        from pa3.task2 import statistics
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "runs/main/task2-evaluation/196"
            rows = []
            for suite in ("grounding_test", "natural_test"):
                for condition in IMAGE_CONDITIONS:
                    for answer in ("yes", "no"):
                        rows.append(dict(suite=suite, condition=condition, type="yes/no",
                            pair_id=suite, score=1., canonical_match=True,
                            normalized_prediction=answer, partner_match=False,
                            nll=.2, generation_seconds=.01))
            predictions = source / "examples.parquet"
            write_parquet(predictions, rows)
            Store(root).finish(source, "existing-evaluation", [predictions])
            before = sha256(predictions)
            with patch("pa3.task2.load_final", side_effect=AssertionError("Must not load GPU model")):
                statistics(root)
                output = source / "statistics/summary.json"
                self.assertEqual(len(read_json(output)), 16)
                saved_time = output.stat().st_mtime_ns
                statistics(root)
                self.assertEqual(output.stat().st_mtime_ns, saved_time)
            self.assertEqual(sha256(predictions), before)


if __name__ == "__main__":
    unittest.main()
