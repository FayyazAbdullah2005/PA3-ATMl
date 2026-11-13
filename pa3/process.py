"""Local subprocess/commit handshake, independent of the Modal SDK."""
import os
import subprocess
import uuid


def run_with_commits(args,cwd,environment,commit):
    token=uuid.uuid4().hex
    env={**os.environ,**environment,"PA3_COMMIT_PROTOCOL":"1","PA3_COMMIT_TOKEN":token}
    with subprocess.Popen(args,cwd=cwd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT,text=True,encoding="utf-8",errors="replace",
                          bufsize=1,env=env) as process:
        try:
            for line in process.stdout:
                if line.strip()=="__PA3_COMMIT__:"+token:
                    commit()
                    process.stdin.write("COMMIT_OK:"+token+"\n")
                    process.stdin.flush()
                else:
                    print(line,end="",flush=True)
            return process.wait()
        except BaseException:
            # Never leave an expensive child waiting forever after a persistence
            # or parent-side error. Its earlier committed artifacts remain valid.
            process.stdin.close()
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise

