"""
SAGE CLI — Interactive Terminal Interface
==========================================
Provides a REPL with slash-commands for training, fine-tuning, quantization,
RAG toggling, and real-time chat with streaming output.
"""

import sys
import os
import torch
from typing import Optional

from .config import SageConfig
from .model import SageModel
from .data import SageTokenizer
from .train import train
from .inference import generate
from .finetune import finetune_instruction, DEMO_INSTRUCTION_SAMPLES
from .optimize import quantize_int8
from .memory import RAGManager, ConversationHistory
from .utils import setup_logger, save_checkpoint, load_checkpoint
from . import __version__

logger = setup_logger("sage.cli")

# ===================================================================
# Banner
# ===================================================================

BANNER = r"""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     ███████  █████   ██████  ███████                         ║
║     ██      ██   ██ ██       ██                              ║
║     ███████ ███████ ██   ███ █████                           ║
║          ██ ██   ██ ██    ██ ██                              ║
║     ███████ ██   ██  ██████  ███████                         ║
║                                                              ║
║     Self-Adaptive General Engine  v{version}                   ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""


def print_banner(model: SageModel, config: SageConfig) -> None:
    """Display startup banner with model statistics."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(BANNER.format(version=__version__))
    print(f"  Model params  : {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"  Trainable     : {trainable_params:,}")
    print(f"  Context length: {config.max_seq_len}")
    print(f"  Device        : {config.device}")
    print(f"  Layers: {config.n_layers}  |  Heads: {config.n_heads}  |  Experts: {config.n_experts}")
    print()
    print("  Type /help for commands, or start chatting!\n")


# ===================================================================
# Help text
# ===================================================================

HELP_TEXT = """
Available Commands:
  /train [steps]     Train the model (default: 100 steps)
  /finetune [steps]  Instruction-tune with LoRA (default: 200 steps)
  /save              Save current model checkpoint
  /load              Load latest checkpoint
  /quantize          Quantize model to INT8 (CPU only)
  /rag on|off        Enable/disable retrieval-augmented generation
  /rag add <text>    Add a document for RAG retrieval
  /clear             Clear conversation history
  /help              Show this message
  /exit              Exit SAGE
"""


# ===================================================================
# Command handlers
# ===================================================================

def handle_train(model, config, tokenizer, args):
    """Handle /train [steps]"""
    steps = 100
    if args:
        try:
            steps = int(args[0])
        except ValueError:
            print(f"  Invalid step count: {args[0]}")
            return model

    print(f"\n  Starting training for {steps} steps …\n")
    model = train(model, config, total_steps=steps, tokenizer=tokenizer, resume=True)

    # Show a quick sample after training
    print("\n  --- Sample generation after training ---")
    generate(model, tokenizer, "Once upon a time", max_new_tokens=80, stream=True, device=config.device)
    print()
    return model


def handle_finetune(model, config, tokenizer, args):
    """Handle /finetune [steps]"""
    steps = 200
    if args:
        try:
            steps = int(args[0])
        except ValueError:
            print(f"  Invalid step count: {args[0]}")
            return model

    print(f"\n  Starting instruction fine-tuning for {steps} steps (LoRA) …\n")
    model = finetune_instruction(
        model, config,
        samples=DEMO_INSTRUCTION_SAMPLES,
        total_steps=steps,
        use_lora=True,
        tokenizer=tokenizer,
    )

    print("\n  --- Sample after fine-tuning ---")
    prompt = "### Instruction:\nWhat is the speed of light?\n\n### Response:\n"
    generate(model, tokenizer, prompt, max_new_tokens=100, stream=True, device=config.device)
    print()
    return model


def handle_save(model, config):
    """Handle /save"""
    path = save_checkpoint(model, None, 0, config.checkpoint_dir)
    print(f"  Model saved to {path}")


def handle_load(model, config):
    """Handle /load"""
    model, _, step = load_checkpoint(model, None, config.checkpoint_dir, device=str(config.device))
    model = model.to(config.device)
    print(f"  Model loaded (step {step})")
    return model


def handle_quantize(model):
    """Handle /quantize"""
    print("  Quantizing model to INT8 (model will be on CPU) …")
    model = quantize_int8(model)
    print("  Quantization complete.")
    return model


def handle_rag(rag_manager: RAGManager, args):
    """Handle /rag on|off|add <text>"""
    if not args:
        state = "enabled" if rag_manager.enabled else "disabled"
        print(f"  RAG is currently {state} ({rag_manager.store.size} chunks indexed)")
        return

    subcmd = args[0].lower()
    if subcmd == "on":
        rag_manager.toggle(True)
        print("  RAG enabled.")
    elif subcmd == "off":
        rag_manager.toggle(False)
        print("  RAG disabled.")
    elif subcmd == "add":
        text = " ".join(args[1:])
        if text:
            rag_manager.add_documents([text])
            print(f"  Document added. Store now has {rag_manager.store.size} chunks.")
        else:
            print("  Usage: /rag add <your document text here>")
    else:
        print("  Usage: /rag on|off|add <text>")


# ===================================================================
# Main REPL
# ===================================================================

def main() -> None:
    """Entry point for the SAGE interactive CLI."""
    config = SageConfig()
    tokenizer = SageTokenizer()

    # Ensure vocab_size matches the tokenizer
    config.vocab_size = tokenizer.vocab_size

    print("  Initializing SAGE model …")
    model = SageModel(config)
    model = model.to(config.device)

    # Attempt to load existing checkpoint
    model, _, loaded_step = load_checkpoint(
        model, None, config.checkpoint_dir, device=str(config.device)
    )
    if loaded_step > 0:
        print(f"  Resumed from checkpoint at step {loaded_step}")

    print_banner(model, config)

    # Initialize RAG and conversation history
    rag_manager = RAGManager(model, tokenizer, config.device)
    history = ConversationHistory(tokenizer, max_tokens=config.max_seq_len - 128)

    # ---------- One-liner CLI arguments ----------
    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()
        args = sys.argv[2:]
        if cmd == "--train":
            handle_train(model, config, tokenizer, args)
        elif cmd == "--finetune":
            handle_finetune(model, config, tokenizer, args)
        elif cmd == "--quantize":
            handle_quantize(model)
        else:
            print(f"  Unknown argument: {cmd}")
            print("  Usage: --train [steps] | --finetune [steps] | --quantize")
        return

    # ---------- REPL loop ----------
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!")
            break

        if not user_input:
            continue

        # ---------- Slash commands ----------
        if user_input.startswith("/"):
            parts = user_input.split()
            cmd = parts[0].lower()
            args = parts[1:]

            if cmd == "/exit":
                print("  Goodbye!")
                break
            elif cmd == "/help":
                print(HELP_TEXT)
            elif cmd == "/train":
                model = handle_train(model, config, tokenizer, args)
            elif cmd == "/finetune":
                model = handle_finetune(model, config, tokenizer, args)
            elif cmd == "/save":
                handle_save(model, config)
            elif cmd == "/load":
                model = handle_load(model, config)
                # Re-attach to RAG manager since model changed
                rag_manager.model = model
            elif cmd == "/quantize":
                model = handle_quantize(model)
                rag_manager.model = model
            elif cmd == "/rag":
                handle_rag(rag_manager, args)
            elif cmd == "/clear":
                history.clear()
                print("  Conversation history cleared.")
            else:
                print(f"  Unknown command: {cmd}. Type /help for a list.")
            continue

        # ---------- Chat mode ----------
        # Build prompt with history and optional RAG context
        rag_context = rag_manager.retrieve_context(user_input)
        prompt = history.build_prompt(user_input, rag_context=rag_context)

        history.add("user", user_input)

        print("SAGE: ", end="", flush=True)
        response = generate(
            model,
            tokenizer,
            prompt,
            max_new_tokens=256,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            stream=True,
            device=config.device,
        )

        # Extract only the SAGE response part from the full generation
        if "SAGE:" in response:
            reply = response.split("SAGE:")[-1].strip()
        else:
            reply = response[len(prompt):].strip()

        history.add("assistant", reply)


if __name__ == "__main__":
    main()
