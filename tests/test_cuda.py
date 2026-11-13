"""Tiny optional CUDA tests: no downloaded weights, no assignment generations."""
from pathlib import Path
import tempfile
import unittest
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from pa3.config import TARGETS
from pa3.models import install_lora
from pa3.train import Trainer,seed_all,token_losses,optimizer_groups


@unittest.skipUnless(torch.cuda.is_available(),"CUDA tests run only when a local GPU is available")
class CudaTests(unittest.TestCase):
    def test_fp16_autocast_with_fp32_adapters_and_optimizer_states(self):
        torch.set_num_threads(2)
        seed_all(42)
        config=Qwen3Config(vocab_size=97,hidden_size=32,intermediate_size=48,num_hidden_layers=1,
            num_attention_heads=2,num_key_value_heads=1,head_dim=16,max_position_embeddings=64,
            pad_token_id=0,eos_token_id=96,tie_word_embeddings=True)
        model=Qwen3ForCausalLM(config).half().to("cuda")
        model.requires_grad_(False)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
        model.enable_input_require_grads()
        install_lora(model,TARGETS,4,8)
        optimizer=torch.optim.AdamW(optimizer_groups(model,1e-3))
        with tempfile.TemporaryDirectory() as tmp:
            trainer=Trainer(model,optimizer,tmp,{"test":"tiny-cuda"})
            ids=torch.tensor([[1,2,3,4,5,6,7,8]],device="cuda")
            labels=ids.clone()
            labels[:,:3]=-100
            def forward(batch):
                return token_losses(model(batch["ids"],use_cache=False).logits,batch["labels"])
            trainer.run(1,lambda step:[{"ids":ids,"labels":labels}],forward,lambda step:1e-3)
            self.assertEqual(trainer.step,1)
            for p in model.parameters():
                if p.requires_grad:
                    self.assertEqual(p.dtype,torch.float32)
                    self.assertIsNotNone(p.grad)
            for state in optimizer.state.values():
                self.assertEqual(state["exp_avg"].dtype,torch.float32)
                self.assertEqual(state["exp_avg_sq"].dtype,torch.float32)
        del model,optimizer,trainer
        torch.cuda.empty_cache()


if __name__=="__main__":
    unittest.main()

