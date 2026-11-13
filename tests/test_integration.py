import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from pa3.io import (write_json,read_json,write_parquet,read_parquet,Store,digest,
                    append_event,read_events)


def fixture_tokenizer():
    from transformers import AutoTokenizer
    local=Path(__file__).resolve().parents[1]/"artifacts"/"test-fixtures"/"tokenizer"
    if local.exists():
        return AutoTokenizer.from_pretrained(local)
    if Path("/vol/data/pins.json").exists():
        from pa3.text import tokenizer_for
        return tokenizer_for("/vol")
    return None


class IntegrationTests(unittest.TestCase):
    def test_audit_checks_permanent_checkpoint_content(self):
        from pa3.cli import execute
        from pa3.io import sha256
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            directory=root/"runs"/"main"/"task1"/"full"/"checkpoints"
            directory.mkdir(parents=True)
            checkpoint=directory/"step-000001.pt"
            checkpoint.write_bytes(b"synthetic checkpoint data")
            write_json(directory/"step-000001.json",{"file":checkpoint.name,"sha256":sha256(checkpoint)})
            execute("audit",root)
            checkpoint.write_bytes(b"corrupted")
            with self.assertRaisesRegex(RuntimeError,"Integrity audit failed"):
                execute("audit",root)

    def test_partial_event_tail_is_repaired_before_append(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"events.jsonl"
            append_event(p,{"step":1})
            with p.open("ab") as f:
                f.write(b'{"incomplete":')
            append_event(p,{"step":2})
            self.assertEqual(read_events(p),[{"step":1},{"step":2}])

    def test_real_qwen_chat_and_visual_masks(self):
        from pa3.text import supervised_example
        from pa3.vision import vision_batch
        tok=fixture_tokenizer()
        if tok is None:
            self.skipTest("Real tokenizer is checked by Modal's CPU preflight after model preparation")
        row=supervised_example(tok,"What is 2+2?","4")
        target=[x for x in row["labels"] if x!=-100]
        self.assertEqual(tok.decode(target),"4<|im_end|>")
        class Features:
            tokens=196
            def get(self,image_id):
                return torch.zeros(196,768)
        e={"image_id":1,"type":"presence","question":"Does this image contain a cat?","answer":"yes"}
        batch=vision_batch([e],tok,Features())
        self.assertEqual(int(batch["visual_mask"].sum()),196)
        self.assertTrue(bool((batch["labels"][batch["visual_mask"]]==-100).all()))
        omitted=vision_batch([e],tok,Features(),no_image=True)
        self.assertEqual(batch["input_ids"].shape[1]-omitted["input_ids"].shape[1],196)

    @unittest.skipUnless(importlib.util.find_spec("lm_eval"),"Harness is installed in the Modal environment")
    def test_harness_real_api_filters_token_flags_and_request_reuse(self):
        import datasets
        from transformers import Qwen3Config,Qwen3ForCausalLM
        from lm_eval.api.task import ConfigurableTask
        from pa3.eval1 import benchmark
        tok=fixture_tokenizer()
        if tok is None:
            self.skipTest("Run prepare-models on Modal first")
        torch.set_num_threads(2)
        cfg=Qwen3Config(vocab_size=len(tok),hidden_size=32,intermediate_size=48,num_hidden_layers=1,
            num_attention_heads=2,num_key_value_heads=1,head_dim=16,max_position_embeddings=256,
            pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id,tie_word_embeddings=True)
        model=Qwen3ForCausalLM(cfg).eval()
        def dataset(**kwargs):
            return datasets.DatasetDict({"test":datasets.Dataset.from_list([
                {"question":"What is 1+1?","answer":"2"},{"question":"What is 2+2?","answer":"4"}])})
        def process(doc,results):
            return {"acc":float(results[0]==doc["answer"])}
        config={"task":"gsm8k_cot","custom_dataset":dataset,"test_split":"test",
            "output_type":"generate_until","doc_to_text":"Q: {{question}}\nA:","doc_to_target":"answer",
            "process_results":process,"metric_list":[{"metric":"acc","aggregation":"mean","higher_is_better":True}],
            "generation_kwargs":{"max_gen_toks":2,"do_sample":False,"until":["<|im_end|>"]},
            "filter_list":[{"name":name,"filter":[{"function":"take_first"}]} for name in ("strict-match","flexible-extract")]}
        task=ConfigurableTask(config=config)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            write_json(root/"data"/"pins.json",{"synthetic":True})
            with patch("pa3.benchmarks.tasks",return_value={"gsm8k_cot":task}):
                # Override the production cap for this synthetic API exercise only.
                import lm_eval
                original=lm_eval.simple_evaluate
                def limited(**kwargs):
                    kwargs["gen_kwargs"]={"max_gen_toks":2,"do_sample":False}
                    return original(**kwargs)
                with patch("lm_eval.simple_evaluate",side_effect=limited):
                    benchmark(root,"main","baseline","gsm8k_cot",smoke=True,loader=lambda:(model,tok))
            d=root/"runs"/"eval-smoke"/"task1-evaluation"/"baseline"/"gsm8k_cot"
            self.assertEqual(len(read_json(d/"all_filters.json")),4)
            samples=read_json(d/"examples.json")
            self.assertEqual(len(samples),2)
            self.assertTrue(all(isinstance(r["hit_max_new_tokens"],bool) for r in samples))
            self.assertTrue(all("extracted_answer" in r for r in samples))
            from pa3.eval1 import persistent_hflm
            from lm_eval.api.instance import Instance
            lm=persistent_hflm(model,tok,d,lambda:None)
            prompts=[row["arguments"][0] for row in samples]
            requests=[Instance(request_type="generate_until",doc={},arguments=tuple(arguments),idx=0) for arguments in prompts]
            with patch.object(lm,"_model_generate",side_effect=AssertionError("Completed request was recomputed")):
                cached=lm.generate_until(requests)
            self.assertEqual(len(cached),2)

    def test_complete_offline_and_adaptive_pipelines_on_synthetic_pools(self):
        from pa3.task3 import offline,adaptive
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source=root/"data"/"task3"
            manifests={}
            for dataset in ("math","mathvista"):
                ids=["dev1","dev2","eval1","eval2"]
                orders={qid:[np.random.default_rng(i).permutation(32).tolist() for i in range(10)] for qid in ids}
                manifests[dataset]={"question_ids":ids,"orders":orders,"splits":{"development":ids[:2],"evaluation":ids[2:]},"live_ids":ids[2:]}
                for qid in ids:
                    rows=[]
                    for i in range(32):
                        row={"normalized_answer":"1" if dataset=="math" else ("A" if i%3 else "B"),
                            "correct":True if dataset=="math" else bool(i%3),"retained_tokens":3,
                            "logp":[-.2,-.3,-.4],"kl_u_p":[1.+i/10,2.,3.],"rollout_id":i,
                            "question_text":"Synthetic test only","gold_answer":"1"}
                        if dataset=="mathvista":
                            row.update(img_attn_total=[.2,.4,.3],prompt_text_attn_total=[.8,.5,.5],
                                       generated_text_attn_total=[0.,.1,.2])
                        rows.append(row)
                    write_parquet(source/dataset/digest(qid)[:20]/"rollouts.parquet",rows)
            write_json(source/"manifest.json",manifests)
            Store(root).finish(source,"synthetic",[source/"manifest.json"])
            offline(root,"synthetic")
            adaptive(root,"synthetic")
            d=root/"runs"/"synthetic"
            self.assertTrue(Store(root).done(d/"task3-offline"))
            self.assertEqual(len(read_json(d/"task3-adaptive"/"selected_policies.json")),4)
            held=read_parquet(d/"task3-adaptive"/"held_out.parquet")
            self.assertEqual(len(held),2*2*2*10)
            self.assertTrue(all(r["N"] in (4,8,16,32) for r in held))
            # Rerunning finished units must not reopen/recompute rollout pools.
            with patch("pa3.task3.offline_question",side_effect=AssertionError("Recomputed completed question")):
                offline(root,"synthetic")


if __name__=="__main__":
    unittest.main()

