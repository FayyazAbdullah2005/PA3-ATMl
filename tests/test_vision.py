import copy
import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from pa3.vision import VLM,Features
from pa3.io import write_json


class VisionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)

    def test_spatial_pooling_averages_correct_two_by_two_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)
            (d/"s").mkdir()
            x=np.arange(196,dtype=np.float16).reshape(1,196,1)*np.ones((1,1,768),dtype=np.float16)
            np.save(d/"s"/"features.npy",x)
            write_json(d/"index.json",{"1":{"shard":"s","row":0}})
            features=Features(d,49)
            result=features.get(1)
            self.assertEqual(tuple(result.shape),(49,768))
            self.assertAlmostEqual(float(result[0,0]),(0+1+14+15)/4)
            self.assertAlmostEqual(float(result[1,0]),(2+3+16+17)/4)
            features.close()

    def test_connector_gradients_frozen_backbone_and_cached_generation(self):
        cfg=Qwen3Config(vocab_size=37,hidden_size=1024,intermediate_size=64,num_hidden_layers=1,
            num_attention_heads=8,num_key_value_heads=2,head_dim=128,max_position_embeddings=128,
            pad_token_id=0,eos_token_id=36,tie_word_embeddings=True)
        lm=Qwen3ForCausalLM(cfg)
        lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
        lm.config.use_cache=False
        model=VLM(lm)
        features=torch.randn(1,4,768)
        batch={"input_ids":torch.tensor([[1,0,0,0,0,2,3]]),"attention_mask":torch.ones(1,7,dtype=torch.long),
               "visual_mask":torch.tensor([[False,True,True,True,True,False,False]]),"features":features}
        model(batch).logits[:,-1].square().mean().backward()
        self.assertGreater(float(model.connector[0].weight.grad.norm()),0)
        for name,p in model.lm.named_parameters():
            self.assertIsNone(p.grad,name)
        model.eval()
        with torch.inference_mode():
            embeds,mask,positions=model.embedding_inputs(batch)
            output=model.lm.generate(inputs_embeds=embeds,attention_mask=mask,max_new_tokens=3,do_sample=False)
        self.assertLessEqual(output.shape[1],3)
        self.assertGreater(output.shape[1],0)
        # No-image batches are shorter and contain no residual image placeholders.
        no_image={"input_ids":torch.tensor([[1,2,3]]),"attention_mask":torch.ones(1,3,dtype=torch.long),
                  "visual_mask":torch.zeros(1,3,dtype=torch.bool),"features":None}
        self.assertEqual(model.embedding_inputs(no_image)[0].shape[1],3)


if __name__=="__main__":
    unittest.main()

