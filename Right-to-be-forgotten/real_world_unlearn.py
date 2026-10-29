import json
import os
import shutil
import sys
import warnings
from pathlib import Path

import datasets
import hydra
import torch
import transformers
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from dataset import TextForgetDatasetQA, custom_data_collator_forget
from trainer import CustomTrainerForgetting
from utils import get_model_identifiers_from_yaml, set_random_seed

from my_models.my_llama import LlamaForCausalLM
warnings.filterwarnings('ignore')

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "workspace") not in sys.path:
    sys.path.insert(0, str(REPO / "workspace"))

from da_asu.protocol import normalize_layers_id, require_local_snapshot
from gold_rank.lock import materialize_gold_rank
from gold_rank.preflight import enable_offline


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def _verdict_path(da):
    candidates = []
    if da.get("verdict_path"):
        candidates.append(Path(da.verdict_path))
    candidates.append(REPO / "workspace/C/derived/m0_verdict.json")
    candidates.append(REPO / "workspace/results/direction_c/m0_verdict.json")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _da_asu_mode(cfg):
    da = cfg.get("da_asu")
    if da is None or not da.enabled:
        return None
    if da.mode not in {"anchor", "logit_control"}:
        raise SystemExit(f"da_asu.mode must be anchor or logit_control, got {da.mode}")
    if not da.reason_path or not Path(da.reason_path).is_file():
        raise SystemExit("da_asu.reason_path must be an existing local JSONL")
    if not da.allow_without_m0:
        verdict_path = _verdict_path(da)
        if verdict_path is None:
            raise SystemExit("M0 verdict missing. Anchor training stays off.")
        verdict = json.loads(verdict_path.read_text())
        if verdict.get("decision") != "proceed":
            raise SystemExit(f"M0 decision is {verdict.get('decision')}. Anchor training stays off.")
    return da.mode


@hydra.main(version_base=None, config_path="config", config_name="real_world")
def main(cfg):
    enable_offline()
    num_devices = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))

    seed = cfg.seed
    set_random_seed(seed)
    model_id = require_local_snapshot(cfg.model_path)
    cfg.layers_id = normalize_layers_id(cfg.layers_id)
    forget_file = Path(cfg.data_path) / f"{cfg.split}.json"
    retain_file = Path(cfg.data_path) / f"{cfg.retain}.json"
    if not forget_file.is_file() or not retain_file.is_file():
        raise SystemExit(f"Refusing to download data. Missing {forget_file} or {retain_file}.")
    da_mode = _da_asu_mode(cfg)
    gold_rank = materialize_gold_rank(cfg.get("gold_rank"), forget_file)
    if gold_rank is not None:
        cfg.gold_rank.enabled = True
        cfg.gold_rank.mode = gold_rank["mode"]
        cfg.gold_rank.k = gold_rank["k"]
        cfg.gold_rank.delta = gold_rank["delta"]
        cfg.gold_rank.penalty = gold_rank["penalty"]
        cfg.gold_rank.seed = gold_rank["seed"]

    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    config = AutoConfig.from_pretrained(model_id, local_files_only=True)

    curr_save_dir = cfg.save_dir

    if os.path.exists(os.path.join(curr_save_dir, 'eval_results-last', 'unlearning_results.txt')):
        print(f'Task already unlearned.')
        exit()

    if local_rank == 0:
        Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
        with open(f"{cfg.save_dir}/config.yaml", "w") as file:
            OmegaConf.save(cfg, file)

    forget_data = datasets.load_dataset('json', data_files=str(forget_file), split='train')
    retain_data = datasets.load_dataset('json', data_files=str(retain_file), split='train')
    reason_data = None
    reason_max_length = None
    if da_mode is not None:
        reason_data = datasets.load_dataset('json', data_files=cfg.da_asu.reason_path, split='train')
        reason_max_length = cfg.da_asu.reason_max_length

    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token

    torch_format_dataset = TextForgetDatasetQA(tokenizer=tokenizer,
                                               model_family=cfg.model_family,
                                               forget_data=forget_data,
                                               retain_data=retain_data,
                                               max_length=300,
                                               mask=cfg.mask,
                                               seed=cfg.seed,
                                               retain_seed_mode="index_only",
                                               reason_data=reason_data,
                                               reason_max_length=reason_max_length)

    batch_size = cfg.batch_size
    gradient_accumulation_steps = cfg.gradient_accumulation_steps
    steps_per_epoch = len(
        torch_format_dataset) // (batch_size * gradient_accumulation_steps * num_devices)

    max_steps = int(cfg.num_epochs * len(torch_format_dataset)) // (
                batch_size * gradient_accumulation_steps * num_devices)
    warmup_steps = steps_per_epoch if steps_per_epoch > 1 else 0

    if cfg.save_steps == 'steps_per_epoch':
        save_steps = steps_per_epoch
    elif cfg.save_steps == 'last':
        save_steps = max_steps
    else:
        save_steps = cfg.save_steps

    if local_rank == 0:
        print("Saving to: ", curr_save_dir)

    # load the config files for deepspeed
    if cfg.use_LoRA:
        ds_config = 'config/ds_config/lora.json'
    else:
        ds_config = 'config/ds_config/llama3.json'

    training_args = transformers.TrainingArguments(
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        learning_rate=cfg.lr,
        bf16=True,
        bf16_full_eval=True,
        output_dir=curr_save_dir,
        optim="paged_adamw_32bit",
        deepspeed=ds_config,
        save_steps=save_steps,
        save_only_model=True,
        ddp_find_unused_parameters=False,
        weight_decay=cfg.weight_decay,
        seed=cfg.seed,
        eval_strategy="no",
        report_to="none",
    )

    # load target LLM
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        attn_implementation='flash_attention_2',
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    model.generation_config.do_sample = True

    if model_cfg["gradient_checkpointing"] == "true":
        model.gradient_checkpointing_enable()

    # Configure LoRA parameters
    if cfg.use_LoRA:
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            target_modules=find_all_linear_names(model),
            r=cfg.LoRA.r,
            lora_alpha=cfg.LoRA.alpha,
            lora_dropout=cfg.LoRA.dropout,
        )
        model = get_peft_model(model, peft_config)

    # load reference model
    if 'ASU' in cfg.forget_loss:
        # arch = AutoConfig.from_pretrained(model_path).architectures[0]
        reference_model = LlamaForCausalLM.from_pretrained(
            model_id,
            config=config,
            attn_implementation='flash_attention_2',
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
        print("Custom Model Loaded")
        
    else:
        reference_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=config,
            attn_implementation='flash_attention_2',
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )

    reference_model = reference_model.eval()

    trainer = CustomTrainerForgetting(
        model=model,
        tokenizer=tokenizer,
        train_dataset=torch_format_dataset,
        eval_dataset=torch_format_dataset,
        # the callback for computing metrics, None in this case since you're doing it in your callback
        compute_metrics=None,
        # callbacks=[GlobalStepDeletionCallback],
        args=training_args,
        data_collator=custom_data_collator_forget,
        loss_type=cfg.forget_loss,
        ref_model=reference_model,
        attention_temp=cfg.attention_temp,
        layers_id=cfg.layers_id,
        beta=cfg.beta,
        forget_coeff=cfg.forget_coeff,
        regularization_coeff=cfg.regularization_coeff,
        gold_rank=gold_rank,
        da_asu_mode=da_mode,
        da_asu_mu=cfg.da_asu.mu if da_mode is not None else 1.0,
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

    print('Start Training ...')
    # Start training
    trainer.train()

    if local_rank == 0:
        if os.path.exists(os.path.join(curr_save_dir, f'checkpoint-{max_steps}')):
            shutil.move(os.path.join(curr_save_dir, f'checkpoint-{max_steps}'),
                        os.path.join(curr_save_dir, f'checkpoint-last'))


if __name__ == "__main__":
    main()
