"""Assignment constants. Changes to experimental settings require a new run ID."""
SEED = 6304
QWEN = "Qwen/Qwen3-0.6B"
SIGLIP = "google/siglip-base-patch16-224"
LIVE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
LIVE_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
MATH_DATASET = "DigitalLearningGmbH/MATH-lighteval"
MATH_REVISION = "0530c78699ea5e8eb5530600900e1f328b48acad"
ROLLOUT_FOLDER = "https://drive.google.com/drive/folders/1utX-Gfi9vvaoi1mi0pzNpDGL1o1fAh23"
QUALITATIVE_URL = "https://raw.githubusercontent.com/emergent-misalignment/emergent-misalignment/main/evaluation/first_plot_questions.yaml"
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
VISION_TARGETS = TARGETS[:4]
T1_METHODS = {
    "full": {"lr": 2e-5, "rank": 0, "alpha": 0, "l2": 0.0},
    "l2sp": {"lr": 2e-5, "rank": 0, "alpha": 0, "l2": 1e-3},
    "lora4": {"lr": 2e-4, "rank": 4, "alpha": 8, "l2": 0.0},
    "lora16": {"lr": 2e-4, "rank": 16, "alpha": 32, "l2": 0.0},
    "oplora": {"lr": 2e-4, "rank": 16, "alpha": 32, "l2": 0.0},
}
T1_MILESTONES = (312, 625, 937, 1250)
T2_STAGES = (
    {"name": "caption", "updates": 2250, "connector_lr": 1e-3, "adapter_lr": 0.0,
     "mixture": {"caption": 1.0}},
    {"name": "grounding", "updates": 1500, "connector_lr": 1e-4, "adapter_lr": 2e-5,
     "mixture": {"presence": .5, "counting": .25, "caption": .25}},
    {"name": "natural", "updates": 750, "connector_lr": 5e-5, "adapter_lr": 1e-5,
     "mixture": {"natural": .7, "presence": .1, "counting": .1, "caption": .1}},
)
BUDGETS = (1, 2, 4, 8, 16, 32)
T1_TASKS = {"gsm8k_cot": ("test", 1319), "hellaswag": ("validation", 10042),
            "boolq": ("validation", 3270), "arc_easy": ("test", 2376)}
IMAGE_CONDITIONS = ("correct", "swap", "mismatched", "none")
T1_SHARED={"examples":40000,"updates":1250,"microbatch":2,"accumulation":16,"max_length":2048,
    "optimizer":"AdamW","betas":[.9,.999],"epsilon":1e-8,"weight_decay":.01,
    "weight_decay_exclusions":"biases and parameters with fewer than two dimensions",
    "schedule":"cosine to zero","warmup_updates":38,"gradient_clip":1.0,
    "gradient_checkpointing":True,"computation":"FP16 autocast + dynamic scaling; no quantization",
    "master_and_optimizer_dtype":"FP32","adapter_backbone_dtype":"FP16",
    "loss":"assistant response + end-of-turn; valid-token mean over effective batch",
    "lora_dropout":.05,"targets":list(TARGETS),"protection_rank":128,"recovery_every":25}
T2_SHARED={"microbatch":8,"accumulation":2,"optimizer":"AdamW","betas":[.9,.999],"epsilon":1e-8,
    "weight_decay":.01,"gradient_clip":1.0,"gradient_checkpointing":True,
    "computation":"FP16 autocast + dynamic scaling; no quantization","trainable_and_optimizer_dtype":"FP32",
    "backbone_dtype":"FP16","caps":{"question":80,"caption":64,"answer":16},
    "loss":"per-example supervised-token mean, then effective-batch example mean",
    "schedule":"ceil(5% stage updates) warmup, cosine to 10% peak","stage_optimizer_policy":"retain existing moments",
    "lora_rank":16,"lora_alpha":32,"lora_dropout":.05,"targets":list(VISION_TARGETS),"recovery_every":25}

