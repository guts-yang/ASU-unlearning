"""INT4 / NF4 quantization wrappers. Do not instantiate a 7B model this round."""

import argparse


def bitsandbytes_load_kwargs(kind):
    kind = kind.lower()
    if kind == "nf4":
        return {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
        }
    if kind == "int4":
        return {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "fp4",
            "bnb_4bit_use_double_quant": False,
        }
    if kind == "int8":
        return {"load_in_8bit": True}
    raise ValueError(f"unknown quant kind: {kind}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="QRA INT4 channel (no 7B this round).")
    parser.add_argument("--model", default="")
    parser.add_argument("--kind", default="nf4", choices=["nf4", "int4", "int8"])
    args = parser.parse_args(argv)
    cfg = bitsandbytes_load_kwargs(args.kind)
    if not args.model:
        print(f"dry-run quant config: {cfg}")
        return 0
    raise RuntimeError("Real quantized load is deferred. Do not instantiate 7B this round.")


if __name__ == "__main__":
    raise SystemExit(main())
