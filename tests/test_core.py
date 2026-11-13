"""Meaningful numerical, recovery, isolation, and causality regression tests."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import warnings
import numpy as np
import torch
from torch import nn
from pa3.io import Store,write_json,read_json,atomic_path,digest,write_parquet,read_parquet
from pa3.models import LoRALinear,install_lora,merge_lora,unique_parameters
from pa3.scoring import (last_boxed,vqa_normalize,vqa_soft,group_answers,rwc_weights,vote,
                         select,mean_scores,bootstrap_difference,mathvista_normalize)
from pa3.train import Trainer,seed_all,token_losses,collate_text,optimizer_groups,schedule_factor
from pa3.task3 import prefix_state,policy_result,offline_question
from pa3.data2 import object_pools,draw_grounding,select_natural
from pa3.task2 import stage_position,draw_effective_batch,development_steps


class ArtifactTests(unittest.TestCase):
    def test_atomic_write_preserves_old_file_after_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"data.json"
            write_json(p,{"v":1})
            with self.assertRaises(RuntimeError):
                with atomic_path(p) as partial:
                    partial.write_text("broken",encoding="utf-8")
                    raise RuntimeError("simulated interruption")
            self.assertEqual(read_json(p),{"v":1})

    def test_completed_units_verify_content_and_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"unit"
            write_json(p/"x.json",{"v":1})
            store=Store(tmp)
            store.finish(p,"sig",[p/"x.json"])
            self.assertTrue(store.done(p,"sig"))
            with self.assertRaises(RuntimeError):
                store.done(p,"other")
            write_json(p/"x.json",{"v":2})
            with self.assertRaises(RuntimeError):
                store.done(p,"sig")

    def test_parquet_unicode_and_null_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"x.parquet"
            rows=[{"answer":"α = ½","score":None,"ids":[1,2]}]
            write_parquet(p,rows)
            self.assertEqual(read_parquet(p),rows)


class NumericalTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        seed_all(123)

    def test_token_mask_and_causal_shift(self):
        logits=torch.zeros(1,4,5,requires_grad=True)
        labels=torch.tensor([[-100,-100,2,3]])
        sums,counts=token_losses(logits,labels)
        self.assertEqual(counts.tolist(),[2])
        self.assertAlmostEqual(float(sums.detach()),2*np.log(5),places=6)
        sums.sum().backward()
        self.assertEqual(float(logits.grad[0,0].abs().sum()),0)
        self.assertGreater(float(logits.grad[0,1].abs().sum()),0)
        self.assertEqual(float(logits.grad[0,3].abs().sum()),0)

    def test_accumulation_matches_effective_batch_for_both_weightings(self):
        labels=torch.tensor([[-100,1,2,-100],[-100,2,3,4],[-100,1,-100,-100]])
        initial=torch.randn(3,4,7)
        for mode in ("tokens","examples"):
            full=initial.clone().requires_grad_()
            sums,counts=token_losses(full,labels)
            loss=sums.sum()/counts.sum() if mode=="tokens" else (sums/counts).mean()
            loss.backward()
            accumulated=initial.clone().requires_grad_()
            for i in range(3):
                a,b=token_losses(accumulated[i:i+1],labels[i:i+1])
                part=a.sum()/counts.sum() if mode=="tokens" else (a/b).sum()/3
                part.backward()
            torch.testing.assert_close(full.grad,accumulated.grad)

    def test_double_sided_projection_constraints_and_merge(self):
        base=nn.Linear(9,7,bias=False)
        u,_,vh=torch.linalg.svd(base.weight.detach(),full_matrices=False)
        lora=LoRALinear(base,3,6,dropout=0,bases=(u[:,:2],vh[:2].T),generator=torch.Generator().manual_seed(7))
        with torch.no_grad():
            lora.B.normal_()
        delta=lora.delta()
        self.assertLess(float((u[:,:2].T@delta).detach().norm()),1e-5)
        self.assertLess(float((delta@vh[:2].T).detach().norm()),1e-5)
        model=nn.Sequential(lora).eval()
        x=torch.randn(2,5,9)
        before=model(x)
        merge_lora(model)
        torch.testing.assert_close(before,model(x),atol=2e-6,rtol=2e-6)

    def test_initial_lora_factors_are_identical_with_or_without_projection(self):
        base=nn.Linear(9,7,bias=False)
        a=LoRALinear(copy.deepcopy(base),3,6,generator=torch.Generator().manual_seed(7))
        u,_,vh=torch.linalg.svd(base.weight.detach(),full_matrices=False)
        b=LoRALinear(copy.deepcopy(base),3,6,bases=(u[:,:2],vh[:2].T),generator=torch.Generator().manual_seed(7))
        torch.testing.assert_close(a.A,b.A)
        torch.testing.assert_close(a.B,b.B)

    def test_l2_sp_exact_gradient_and_tied_parameter_accounting(self):
        model=nn.Module()
        model.a=nn.Linear(3,3,bias=True)
        model.b=nn.Linear(3,3,bias=False)
        model.b.weight=model.a.weight
        params=unique_parameters(model)
        self.assertEqual(len(params),2)
        references={n:p.detach().clone()-0.2 for n,p in params}
        omega=sum(.5*(p-references[n]).square().sum() for n,p in params)
        omega.backward()
        for n,p in params:
            torch.testing.assert_close(p.grad,p-references[n])

    def test_cosine_endpoints_and_stage_boundaries(self):
        self.assertAlmostEqual(schedule_factor(38,1250,38),1)
        self.assertAlmostEqual(schedule_factor(1250,1250,38),0)
        self.assertAlmostEqual(schedule_factor(750,750,38,.1),.1)
        self.assertEqual(stage_position(2249),(0,2249))
        self.assertEqual(stage_position(2250),(1,0))
        self.assertEqual(stage_position(3750),(2,0))
        self.assertEqual(len(development_steps()),12)


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def run_model(self,directory,steps,restore=False):
        seed_all(42)
        model=nn.Sequential(nn.Linear(4,8),nn.Dropout(.2),nn.Tanh(),nn.Linear(8,4))
        optimizer=torch.optim.AdamW(optimizer_groups(model,.01))
        trainer=Trainer(model,optimizer,directory,{"test":"resume"},full_state=True)
        if restore:
            self.assertTrue(trainer.restore())
        x=torch.arange(24,dtype=torch.float32).reshape(2,3,4)/20
        labels=torch.tensor([[-100,1,2],[-100,2,3]])
        batches=lambda step:[{"x":x,"labels":labels}]
        forward=lambda batch:token_losses(model(batch["x"]),batch["labels"])
        trainer.run(steps,batches,forward,lambda step:.01,save_every=1)
        return model,trainer

    def test_resume_restores_optimizer_rng_and_data_position(self):
        with tempfile.TemporaryDirectory() as tmp:
            uninterrupted,_=self.run_model(Path(tmp)/"a",4)
            self.run_model(Path(tmp)/"b",2)
            resumed,trainer=self.run_model(Path(tmp)/"b",4,restore=True)
            for a,b in zip(uninterrupted.parameters(),resumed.parameters()):
                torch.testing.assert_close(a,b,rtol=0,atol=0)
            self.assertEqual(trainer.step,4)
            self.assertEqual(trainer.extra["examples_seen"],8)

    def test_corrupt_latest_checkpoint_falls_back_to_verified_older_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,trainer=self.run_model(tmp,2)
            latest=trainer.checkpoints.latest()
            latest[0].write_bytes(b"incomplete")
            # Step-2 final also has a rolling recovery at step 1.
            fallback=trainer.checkpoints.latest()
            self.assertIsNotNone(fallback)
            self.assertLess(fallback[1]["step"],2)


class ScoringTests(unittest.TestCase):
    def test_balanced_last_box_extractor(self):
        self.assertEqual(last_boxed(r"a \boxed{1} b \boxed{\frac{1}{2}}"),r"\frac{1}{2}")
        for text in ("42",r"\boxed{}",r"\boxed{1} then \boxed{bad"):
            self.assertIsNone(last_boxed(text))

    def test_vqa_normalization_and_leave_one_out_soft_scores(self):
        self.assertEqual(vqa_normalize("The TWO cats!"),"2 cats")
        self.assertEqual(vqa_normalize("1,000"),"1000")
        self.assertEqual(vqa_normalize("3.14"),"3.14")
        self.assertAlmostEqual(vqa_soft("yes",["yes"]+["no"]*9),.3)
        self.assertAlmostEqual(vqa_soft("yes",["yes"]*2+["no"]*8),.6)
        self.assertAlmostEqual(vqa_soft("yes",["yes"]*3+["no"]*7),.9)
        self.assertEqual(vqa_soft("yes",["yes"]*4+["no"]*6),1)
        self.assertEqual(vqa_soft("",[""]*10),0)

    def test_mathvista_option_numeric_and_list_rules(self):
        meta={"question_type":"multi_choice","choices":["red","blue"]}
        self.assertEqual(mathvista_normalize("(B)",meta),"B")
        self.assertEqual(mathvista_normalize("blue",meta),"B")
        self.assertEqual(mathvista_normalize("3.0",{"answer_type":"integer"}),"3")
        self.assertEqual(mathvista_normalize("3.14159",{"answer_type":"float","precision":2}),"3.14")

    def test_ties_missing_answers_and_average_ranks(self):
        groups,reps=group_answers([None,"a","", "b"],lambda a,b:a==b)
        self.assertEqual(groups,[0,1,0,2])
        self.assertIsNone(reps[0])
        self.assertEqual(vote([1,2])[0],1)
        np.testing.assert_allclose(rwc_weights([5,5,1]),[.75,.75,0])
        self.assertEqual(select([0],{"certainty":[0]},"certainty_rwc"),0)

    def test_prefix_calibration_and_fusion_identity(self):
        raw=[{"logp":[-.2,-.8],"kl_u_p":[0.,2.],"img_attn_total":[.8,.2],
              "prompt_text_attn_total":[.2,.8],"generated_text_attn_total":[0.,0.]},
             {"logp":[-.5,-.5],"kl_u_p":[1.,3.],"img_attn_total":[.4,.6],
              "prompt_text_attn_total":[.6,.4],"generated_text_attn_total":[0.,0.]}]
        scores=mean_scores(raw,True)
        pooled=np.array([0.,2.,1.,3.])
        low,high=np.quantile(pooled,[.05,.95])
        for i,r in enumerate(raw):
            c=np.clip((np.array(r["kl_u_p"])-low)/(high-low),0,1)
            v=np.array(r["img_attn_total"])
            covariance=np.mean((c-c.mean())*(v-v.mean()))
            self.assertAlmostEqual(scores["token_fusion"][i]-scores["visconf"][i],covariance)
        # Adding an unseen extreme candidate must not affect an already computed prefix.
        first=mean_scores(raw[:1],True)
        self.assertNotAlmostEqual(float(first["visconf"][0]),float(scores["visconf"][0]))

    def test_bootstrap_is_paired_and_reproducible(self):
        a=bootstrap_difference([1,0,1],[0,0,1],100)
        self.assertEqual(a,bootstrap_difference([1,0,1],[0,0,1],100))
        self.assertAlmostEqual(a["difference"],1/3)

    def test_selection_rejects_gold_data(self):
        with self.assertRaises(ValueError):
            prefix_state([{"correct":True}],"math")

    def test_adaptive_stopping_does_not_depend_on_correctness(self):
        states=[{"N":n,"unique_leader":True,"nonempty_leader":True,"agreement":.75,
                 "quality_agrees":True,"margin":.3,"correct":False} for n in (4,8,16,32)]
        self.assertEqual(policy_result(states,.75,.25)["N"],4)
        changed=[{**r,"correct":True} for r in states]
        self.assertEqual(policy_result(changed,.75,.25)["N"],4)
        self.assertEqual(policy_result(states,.875,.25)["N"],32)
        self.assertEqual(policy_result(states)["N"],32)


class DataTests(unittest.TestCase):
    def test_grounding_pair_generation_and_exclusion(self):
        instances={"categories":[{"id":1,"name":"person"}],"annotations":[
            {"image_id":1,"category_id":1},{"image_id":2,"category_id":1},{"image_id":2,"category_id":1}]}
        pools=object_pools(instances,[1,2,3,4])
        rng=np.random.default_rng(1)
        pair=draw_grounding(pools,"counting",rng)
        self.assertNotEqual(pair["examples"][0]["answer"],pair["examples"][1]["answer"])
        self.assertIn("people",pair["examples"][0]["question"])
        pair=draw_grounding(pools,"presence",rng,excluded={4})
        self.assertEqual(pair["examples"][1]["image_id"],3)

    def test_caption_curriculum_has_exactly_two_passes(self):
        data={"captions":{"train":[{"caption_id":i} for i in range(18000)]}}
        for epoch in (0,1):
            found=[]
            for step in range(epoch*1125,(epoch+1)*1125):
                batch,counts=draw_effective_batch(data,step)
                found.extend(e["caption_id"] for e in batch)
            self.assertEqual(sorted(found),list(range(18000)))


if __name__=="__main__":
    unittest.main()

