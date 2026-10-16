# Copyright (c) ModelScope Contributors. All rights reserved.
from types import SimpleNamespace

import numpy as np
import pytest

from twinkle.data_format import SampledSequence, SampleResponse, SamplingParams
from twinkle.sampler.vllm_sampler.vllm_sampler import vLLMSampler


@pytest.mark.asyncio
@pytest.mark.parametrize('labels', [None, [-100, 2], np.array([-100, 2])])
async def test_prompt_label_normalization_preserves_existing_arrays(labels):
    sampler = vLLMSampler.__new__(vLLMSampler)

    async def generate(**kwargs):
        return SampleResponse(prompt_token_ids=[1, 2], sequences=[SampledSequence(
            tokens=[3], logprobs=[[(3, -0.1)]], stop_reason='length')])

    sampler.engine = SimpleNamespace(sample=generate)
    sampler.template = SimpleNamespace(
        get_vllm_input_ids=lambda ids: ids,
        concat_input_feature=lambda feat, tokens: feat,
        decode=lambda tokens: 'result')
    feat = {'input_ids': [1, 2]}
    if labels is not None:
        feat['labels'] = labels
    result = await sampler._sample_single(feat, SamplingParams(max_tokens=1))
    expected = [-100, -100] if labels is None else list(labels)
    assert result.sequences[0].new_input_feature['labels'] == expected
    if labels is not None:
        assert feat['labels'] is labels
