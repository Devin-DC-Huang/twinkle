# Copyright (c) ModelScope Contributors. All rights reserved.
"""Exercise dev task entry points on real tiny decoder models without downloads."""
import copy

import pytest
import torch


@pytest.mark.parametrize('task', ['embedding', 'reranker', 'seq_cls'])
def test_dev_task_forward_backward_and_update(tmp_path, monkeypatch, task):
    from tokenizers import Tokenizer
    from tokenizers.models import WordPiece
    from transformers import (PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM,
                              Qwen2ForSequenceClassification)
    from twinkle import Platform, initialize
    from twinkle.model.transformers.transformers import TransformersModel
    monkeypatch.setenv('ACCELERATE_USE_CPU', 'true')
    monkeypatch.setattr(Platform, 'get_local_device', lambda *args, **kwargs: 'cpu')
    initialize(mode='local')
    config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=2, pad_token_id=0,
                         num_labels=3 if task == 'seq_cls' else 1)
    model_cls = Qwen2ForCausalLM if task == 'embedding' else Qwen2ForSequenceClassification
    model_cls(config).save_pretrained(tmp_path)
    vocab = {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4}
    vocab.update({f't{i}': i for i in range(5, 32)})
    PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordPiece(vocab, unk_token='[UNK]')),
                           pad_token='[PAD]', unk_token='[UNK]').save_pretrained(tmp_path)
    model = TransformersModel(model_id=str(tmp_path), mixed_precision='no', gradient_checkpointing=False)
    model.set_processor('InputProcessor', padding_side='right')
    if task == 'embedding':
        model.set_loss('InfonceLoss', temperature=0.1)
        labels = [1, 0, 0, 1, 0, 0]
    elif task == 'reranker':
        model.set_loss('PointwiseRerankerLoss')
        labels = [1., 0., 0., 1., 0., 0.]
    else:
        model.set_loss('SeqClsLoss', problem_type='single_label_classification', num_labels=3)
        labels = [0, 1, 2, 0, 1, 2]
    model.set_optimizer('AdamW', lr=1e-3)
    inputs = [{'input_ids': [2, 5+i, 3], 'attention_mask': [1] * 3, 'labels': [label]}
              for i, label in enumerate(labels)]
    before = {name: p.detach().clone() for name, p in model.model.named_parameters()}
    result = model.forward_backward(inputs=copy.deepcopy(inputs), task=task)
    assert torch.isfinite(torch.tensor(result['loss']))
    model.clip_grad_and_step()
    assert any(not torch.equal(before[name], p) for name, p in model.model.named_parameters())
    result = model.forward_only(inputs=copy.deepcopy(inputs), task=task, return_logits=True)
    output = result['embeddings' if task == 'embedding' else 'logits']
    assert output.shape[0] == len(inputs) and torch.isfinite(output).all()
