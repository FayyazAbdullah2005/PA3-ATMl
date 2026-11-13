import contextlib
import io
import sys
import tempfile
import unittest
from pa3.process import run_with_commits


class ProcessTests(unittest.TestCase):
    def test_commit_handshake_survives_progress_output_and_rejects_wrong_token(self):
        code="""import os,sys
token=os.environ['PA3_COMMIT_TOKEN']
sys.stdout.write('progress without newline')
sys.stdout.flush()
print('\\n__PA3_COMMIT__:wrong-token',flush=True)
print('\\n__PA3_COMMIT__:'+token,flush=True)
ack=sys.stdin.readline().strip()
assert ack=='COMMIT_OK:'+token,ack
print('finished',flush=True)
"""
        committed=[]
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()) as output:
            status=run_with_commits([sys.executable,"-c",code],tmp,{},lambda:committed.append(True))
        self.assertEqual(status,0)
        self.assertEqual(committed,[True])
        self.assertIn("finished",output.getvalue())

    def test_commit_failure_stops_child_instead_of_hanging(self):
        code="""import os,sys
print('\\n__PA3_COMMIT__:'+os.environ['PA3_COMMIT_TOKEN'],flush=True)
sys.stdin.readline()
"""
        def fail():
            raise RuntimeError("simulated storage failure")
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError,"simulated storage failure"):
                run_with_commits([sys.executable,"-c",code],tmp,{},fail)


if __name__=="__main__":
    unittest.main()

