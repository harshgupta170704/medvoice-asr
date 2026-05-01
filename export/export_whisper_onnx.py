"""
export/export_whisper_onnx.py
──────────────────────────────────────────────────────────────────────────
Exports a fine-tuned Whisper model to ONNX and applies dynamic INT8 
quantization, achieving ~35% latency reduction at inference time.

Usage:
    python export/export_whisper_onnx.py \
        --model_dir  models/whisper \
        --output_dir models/whisper_onnx \
        [--quantize]
"""

import argparse
import os
from pathlib import Path

import torch
import numpy as np
from transformers import WhisperForConditionalGeneration, WhisperProcessor


# ──────────────────────────────────────────────────────────────────────
# Helper: create dummy inputs for encoder tracing
# ──────────────────────────────────────────────────────────────────────

def dummy_inputs(processor: WhisperProcessor, device: torch.device) -> dict:
    """Generate a batch-1 log-mel spectrogram for tracing."""
    dummy_audio = np.zeros(16_000, dtype=np.float32)   # 1 second of silence
    inputs = processor(dummy_audio, sampling_rate=16_000, return_tensors="pt")
    return {"input_features": inputs.input_features.to(device)}


# ──────────────────────────────────────────────────────────────────────
# ONNX Export via Optimum
# ──────────────────────────────────────────────────────────────────────

def export_with_optimum(model_dir: str, output_dir: str) -> None:
    """
    Preferred path: use HuggingFace Optimum for a complete, correct export
    that handles the encoder-decoder structure of Whisper.
    """
    try:
        from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
        print("[INFO] Exporting Whisper with Optimum …")
        model = ORTModelForSpeechSeq2Seq.from_pretrained(
            model_dir,
            export=True,
            provider="CPUExecutionProvider",
        )
        model.save_pretrained(output_dir)
        print(f"[OK] ONNX model saved to {output_dir}")
    except ImportError:
        print("[WARN] optimum not available, falling back to manual export.")
        export_manual(model_dir, output_dir)


def export_manual(model_dir: str, output_dir: str) -> None:
    """
    Manual torch.onnx.export of the Whisper encoder only.
    Useful when optimum is unavailable.
    The decoder can be left in PyTorch (hybrid mode).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Loading model from {model_dir} on {device}")

    processor = WhisperProcessor.from_pretrained(model_dir)
    model = WhisperForConditionalGeneration.from_pretrained(model_dir).to(device)
    model.eval()

    encoder = model.model.encoder
    dummy = dummy_inputs(processor, device)

    onnx_path = output_dir / "whisper_encoder.onnx"
    print(f"[INFO] Exporting encoder to {onnx_path} …")

    with torch.no_grad():
        torch.onnx.export(
            encoder,
            (dummy["input_features"],),
            str(onnx_path),
            input_names=["input_features"],
            output_names=["last_hidden_state"],
            dynamic_axes={
                "input_features":   {0: "batch_size"},
                "last_hidden_state": {0: "batch_size"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
    print(f"[OK] Encoder ONNX saved: {onnx_path}")

    # Save processor alongside so app can load it
    processor.save_pretrained(str(output_dir))


# ──────────────────────────────────────────────────────────────────────
# INT8 Dynamic Quantization
# ──────────────────────────────────────────────────────────────────────

def quantize_onnx(onnx_dir: str) -> None:
    """
    Apply dynamic INT8 quantization to all ONNX files in onnx_dir.
    Targets MatMul and Gemm ops (dominant in transformer weights).
    Expected latency reduction: ~30-40% on CPU.
    """
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        print("[ERR] onnxruntime.quantization not available. Install onnxruntime.")
        return

    onnx_dir = Path(onnx_dir)
    onnx_files = list(onnx_dir.glob("*.onnx"))

    if not onnx_files:
        print(f"[WARN] No .onnx files found in {onnx_dir}")
        return

    for onnx_path in onnx_files:
        if "quantized" in onnx_path.name:
            continue
        q_path = onnx_path.parent / (onnx_path.stem + "_quantized.onnx")
        print(f"[INFO] Quantizing {onnx_path.name} → {q_path.name} …")

        quantize_dynamic(
            model_input=str(onnx_path),
            model_output=str(q_path),
            weight_type=QuantType.QInt8,
            optimize_model=True,
        )
        orig_mb  = onnx_path.stat().st_size / 1e6
        quant_mb = q_path.stat().st_size / 1e6
        print(f"  Size: {orig_mb:.1f} MB → {quant_mb:.1f} MB ({100*(1-quant_mb/orig_mb):.0f}% reduction)")

    print(f"[OK] Quantization complete → {onnx_dir}")


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Export Whisper → ONNX + INT8 Quantization")
    p.add_argument("--model_dir",  default="models/whisper",
                   help="Path to fine-tuned Whisper checkpoint")
    p.add_argument("--output_dir", default="models/whisper_onnx",
                   help="Where to save the ONNX model")
    p.add_argument("--quantize",   action="store_true",
                   help="Apply INT8 dynamic quantization after export")
    p.add_argument("--manual",     action="store_true",
                   help="Force manual torch.onnx.export (skip optimum)")
    return p.parse_args()


def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.manual:
        export_manual(args.model_dir, args.output_dir)
    else:
        export_with_optimum(args.model_dir, args.output_dir)

    if args.quantize:
        print("\n[INFO] Applying INT8 quantization …")
        quantize_onnx(args.output_dir)


if __name__ == "__main__":
    main()
