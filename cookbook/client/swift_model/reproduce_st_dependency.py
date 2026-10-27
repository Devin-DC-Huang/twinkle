# Copyright (c) ModelScope Contributors. All rights reserved.
"""Reproduce the pinned Swift ST constructor failure without HTTP or Ray.

Pass an existing SentenceTransformer directory to --model for an offline run.
The default downloads the same model used by the planned CS acceptance.
"""

import argparse

import twinkle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = parser.parse_args()
    from swift.dev.model.multi_lora_sentence_transformer_model import MultiLoraSentenceTransformerModel

    twinkle.initialize(mode="local")
    MultiLoraSentenceTransformerModel(model_id=args.model, mixed_precision="no", max_loras=2, max_r=4)


if __name__ == "__main__":
    main()
