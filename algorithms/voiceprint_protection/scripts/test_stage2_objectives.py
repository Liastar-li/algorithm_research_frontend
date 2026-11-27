"""Stage 2 论文目标与 GAN 梯度隔离的轻量回归测试。"""

import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stage2 import losses as loss_module
from stage2.common import perturb_embedding
from stage2.pipeline import Stage2Pipeline
from stage2.train import compute_losses, make_scaler, train_step


class FakeEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, waveform, lengths=None):
        self.calls.append(waveform.detach().clone())
        features = torch.stack((waveform.mean(-1), waveform.square().mean(-1)), dim=1)
        return torch.nn.functional.normalize(features, dim=1)


class FakeStage1(torch.nn.Module):
    def content_code(self, mel, embedding):
        content = mel.mean(1) + embedding.mean(1, keepdim=True)
        return content.unsqueeze(1).expand(-1, mel.shape[1], -1)

    def decode(self, mel, embedding, content):
        shift = embedding.mean(1, keepdim=True).unsqueeze(-1)
        return mel + shift + 2.0 * content.mean(1, keepdim=True)


class FakeVocoder(torch.nn.Module):
    def forward(self, mel):
        return -mel[..., 0]


class FakeDiscriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(80, 1, bias=False)

    def forward(self, mel):
        return self.head(mel.mean(1))


def make_args(**overrides):
    values = dict(alpha=0.1, lambda_de=5.0, lambda_per=0.0, lambda_mask=0.0,
                  lambda_gan=1.0, no_mel_loss=True, accum_steps=1,
                  amp=False, clip_grad=0.0)
    values.update(overrides)
    return SimpleNamespace(**values)


def make_pipeline(split_space=False):
    speaker = FakeEncoder()
    stage1 = FakeStage1()
    vocoder = FakeVocoder()
    de_speaker = FakeEncoder() if split_space else speaker
    return SimpleNamespace(
        speaker=speaker,
        de_speaker=de_speaker,
        stage1=stage1,
        vocoder=vocoder,
        mel=lambda waveform: waveform.unsqueeze(-1).expand(-1, -1, 80),
        vocode=vocoder,
        align=Stage2Pipeline.align,
        thresholds=lambda waveform: None,
        content=stage1.content_code,
    )


class Stage2ObjectiveTests(unittest.TestCase):
    def test_perturbation_matches_paper_addition(self):
        embedding = torch.tensor([[1.0, 0.0]])
        delta = torch.tensor([[0.0, 1.0]])
        torch.testing.assert_close(perturb_embedding(embedding, delta, 0.1),
                                   torch.tensor([[1.0, 0.1]]))

    def setUp(self):
        torch.manual_seed(7)
        self.waveforms = torch.tensor([[0.2, 0.4, 0.8, 1.2],
                                       [0.5, 0.7, 0.9, 1.1]])
        self.lengths = torch.tensor([4, 4])
        self.batch = (self.waveforms, self.lengths, None)
        self.generator = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.generator.weight.copy_(torch.eye(2))
        self.discriminator = FakeDiscriminator()

    def test_defense_reference_is_vocoded_original(self):
        pipeline = make_pipeline(split_space=True)
        out = compute_losses(self.batch, pipeline, self.generator, self.discriminator,
                             make_args(), torch.device("cpu"), measure_dual=True)
        reconstructed = pipeline.vocode(pipeline.mel(self.waveforms))
        torch.testing.assert_close(pipeline.de_speaker.calls[0], reconstructed)
        torch.testing.assert_close(out["de"], out["sim_clone"])
        self.assertNotEqual(float(out["sim_clone"].detach()),
                            float(out["sim_prot"].detach()))

    def test_discriminator_receives_only_its_own_gradient(self):
        pipeline = make_pipeline()
        args = make_args()
        optimizer_g = torch.optim.AdamW(self.generator.parameters(), lr=0.01)
        optimizer_d = torch.optim.AdamW(self.discriminator.parameters(), lr=0.01)
        expected_discriminator = copy.deepcopy(self.discriminator)

        last, _ = train_step(self.batch, pipeline, self.generator, self.discriminator,
                             optimizer_g, optimizer_d, make_scaler(False), args,
                             torch.device("cpu"))
        expected_loss = loss_module.discriminator_gan_loss(
            expected_discriminator(last["original_mel"]),
            expected_discriminator(last["protected_mel"]),
        )
        expected_loss.backward()

        self.assertIsNotNone(self.generator.weight.grad)
        for actual, expected in zip(self.discriminator.parameters(),
                                    expected_discriminator.parameters()):
            torch.testing.assert_close(actual.grad, expected.grad)

    def test_accumulated_generator_steps_keep_discriminator_gradient_isolated(self):
        pipeline = make_pipeline()
        args = make_args(accum_steps=2)
        optimizer_g = torch.optim.AdamW(self.generator.parameters(), lr=0.01)
        optimizer_d = torch.optim.AdamW(self.discriminator.parameters(), lr=0.01)
        expected_discriminator = copy.deepcopy(self.discriminator)

        last, _ = train_step(self.batch, pipeline, self.generator, self.discriminator,
                             optimizer_g, optimizer_d, make_scaler(False), args,
                             torch.device("cpu"))
        expected_loss = loss_module.discriminator_gan_loss(
            expected_discriminator(last["original_mel"]),
            expected_discriminator(last["protected_mel"]),
        )
        expected_loss.backward()

        for actual, expected in zip(self.discriminator.parameters(),
                                    expected_discriminator.parameters()):
            torch.testing.assert_close(actual.grad, expected.grad)


if __name__ == "__main__":
    unittest.main()
