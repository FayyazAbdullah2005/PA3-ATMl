"""Read committed progress through Modal's storage API; launches no containers."""
import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import PurePosixPath
import time


def read_text(volume, path):
    from modal.exception import NotFoundError
    try:
        return b"".join(volume.read_file(path)).decode("utf-8")
    except (NotFoundError, FileNotFoundError):
        return ""


def show(volume, run_id):
    from modal.exception import NotFoundError
    print(f"\n{datetime.now():%H:%M:%S} - saved cloud progress for {run_id}", flush=True)
    events = read_text(volume, "/orchestration.jsonl")
    for line in reversed(events.splitlines()):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("options", {}).get("run_id") == run_id:
            label = next((k for k in ("starting", "completed", "reused_without_gpu") if k in event), None)
            if label:
                print(f"Last recorded stage: {label} {event[label]}")
                break
    base = f"/runs/{run_id}"
    counts = Counter()
    strategies = {"parallel0", "parallel1", "parallel2", "long", "revision"}
    try:
        for entry in volume.iterdir(base + "/task3-live", recursive=True):
            path = PurePosixPath(entry.path)
            if path.name == "DONE.json" and path.parent.name in strategies:
                counts[str(path.parent.parent)] += 1
    except (NotFoundError, FileNotFoundError):
        pass
    total = sum(counts.values())
    complete = sum(n == 5 for n in counts.values())
    print(f"Task 3 live: {total}/250 responses saved ({total/250:.1%}); {complete}/50 questions have all 5 responses.")
    for tokens in (196, 49):
        training = base + f"/task2/{tokens}"
        done = read_text(volume, training + "/DONE.json")
        records = read_text(volume, training + "/training.jsonl").splitlines()
        latest = None
        for line in reversed(records):
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "step" in candidate:
                latest = candidate
                break
        label = "COMPLETE" if done else f"update {latest['step']}/4500" if latest else "no saved training updates yet"
        print(f"Task 2 ({tokens} tokens): {label}")
    completion = read_text(volume, base + "/completion.json")
    if completion and json.loads(completion).get("complete"):
        print("ALL REQUIRED COMPUTATIONAL OUTPUTS VERIFIED COMPLETE")
    print("Saved progress can lag active generation. Counts are not a time estimate.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="main")
    parser.add_argument("--watch", action="store_true", help="Refresh every 60 seconds; Ctrl+C stops only this viewer")
    args = parser.parse_args()
    import modal
    volume = modal.Volume.from_name("atml-pa3-artifacts")
    try:
        while True:
            show(volume, args.run_id)
            if not args.watch:
                break
            time.sleep(60)
    except KeyboardInterrupt:
        print("\nViewer stopped. The cloud run is unaffected.")


if __name__ == "__main__":
    main()
