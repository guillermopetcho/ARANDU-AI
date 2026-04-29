"""Tests del LR Scheduler de AranduSSL.

Valida las propiedades del schedule warmup-lineal + decaimiento cosenoidal
construido por engine/scheduler.py::build_scheduler().

Cómo correr:
    pytest tests/test_scheduler.py -v

Requisitos: torch (ya requerido por el proyecto)
"""
import math
import sys
import os

import pytest
import torch
import torch.nn as nn

# Resolver el path del proyecto para que los imports de engine/ funcionen
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.scheduler import build_scheduler, momentum_update


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def optimizer_and_scheduler():
    """Optimizer y scheduler con configuración representativa de producción."""
    base_lr   = 0.0225
    w_steps   = 1765    # ~5 epochs × 353 steps/epoch
    t_steps   = 70600   # 200 epochs × 353 steps/epoch

    model = nn.Linear(10, 10)
    opt   = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=0.9)

    # Replicar el parche de initial_lr que hace train.py
    for pg in opt.param_groups:
        pg['initial_lr'] = base_lr

    scheduler = build_scheduler(opt, w_steps, t_steps, final_lr_ratio=0.0)
    return opt, scheduler, base_lr, w_steps, t_steps


# ---------------------------------------------------------------------------
# Propiedades del Warmup
# ---------------------------------------------------------------------------

class TestWarmup:
    def test_lr_starts_at_1_percent(self, optimizer_and_scheduler):
        """Al inicio (step 0), el LR debe ser ~1% del LR base (factor 0.01)."""
        opt, scheduler, base_lr, *_ = optimizer_and_scheduler
        lr_step0 = opt.param_groups[0]['lr']
        assert abs(lr_step0 - base_lr * 0.01) < 1e-7, (
            f"LR en step 0 debería ser ~{base_lr * 0.01:.6f}, got {lr_step0:.6f}"
        )

    def test_lr_is_monotonically_increasing_during_warmup(self, optimizer_and_scheduler):
        """Durante el warmup el LR debe crecer de forma monótona."""
        opt, scheduler, base_lr, w_steps, _ = optimizer_and_scheduler
        lrs = [opt.param_groups[0]['lr']]
        for _ in range(w_steps):
            scheduler.step()
            lrs.append(opt.param_groups[0]['lr'])

        for i in range(len(lrs) - 1):
            assert lrs[i] <= lrs[i + 1] + 1e-9, (
                f"LR no monótono en warmup: step {i}={lrs[i]:.6f} > step {i+1}={lrs[i+1]:.6f}"
            )

    def test_lr_reaches_base_after_warmup(self, optimizer_and_scheduler):
        """Al finalizar el warmup, el LR debe alcanzar el LR base (100%)."""
        opt, scheduler, base_lr, w_steps, _ = optimizer_and_scheduler
        for _ in range(w_steps):
            scheduler.step()
        lr_end_warmup = opt.param_groups[0]['lr']
        assert abs(lr_end_warmup - base_lr) < 1e-5, (
            f"LR al finalizar warmup debería ser ~{base_lr:.6f}, got {lr_end_warmup:.6f}"
        )


# ---------------------------------------------------------------------------
# Propiedades del Decaimiento Cosenoidal
# ---------------------------------------------------------------------------

class TestCosineDecay:
    def test_lr_decreases_after_warmup(self, optimizer_and_scheduler):
        """Post-warmup, el LR debe decrecer en cada paso."""
        opt, scheduler, _, w_steps, t_steps = optimizer_and_scheduler
        for _ in range(w_steps):
            scheduler.step()

        decay_steps = t_steps - w_steps
        lrs = [opt.param_groups[0]['lr']]
        for _ in range(min(decay_steps, 500)):  # Muestrear los primeros 500 pasos
            scheduler.step()
            lrs.append(opt.param_groups[0]['lr'])

        for i in range(len(lrs) - 1):
            assert lrs[i] >= lrs[i + 1] - 1e-9, (
                f"LR no monótono decreciente post-warmup: step {i}={lrs[i]:.6f} < step {i+1}={lrs[i+1]:.6f}"
            )

    def test_lr_reaches_zero_at_end(self, optimizer_and_scheduler):
        """Al final del training, el LR debe llegar a 0 (final_lr_ratio=0.0)."""
        opt, scheduler, _, _, t_steps = optimizer_and_scheduler
        for _ in range(t_steps):
            scheduler.step()
        lr_final = opt.param_groups[0]['lr']
        assert lr_final < 1e-5, (
            f"LR al final debería ser ~0, got {lr_final:.8f}"
        )

    def test_final_lr_ratio_respected(self):
        """Con final_lr_ratio > 0, el LR no debe caer por debajo de base_lr * ratio."""
        base_lr       = 0.01
        final_lr_ratio = 0.1
        w_steps       = 100
        t_steps       = 1000

        model = nn.Linear(5, 5)
        opt   = torch.optim.SGD(model.parameters(), lr=base_lr)
        sched = build_scheduler(opt, w_steps, t_steps, final_lr_ratio=final_lr_ratio)

        for _ in range(t_steps):
            sched.step()

        lr_final = opt.param_groups[0]['lr']
        expected_floor = base_lr * final_lr_ratio
        assert lr_final >= expected_floor - 1e-7, (
            f"LR final {lr_final:.8f} está por debajo del floor {expected_floor:.8f}"
        )

    def test_cosine_midpoint_is_half_lr(self, optimizer_and_scheduler):
        """A mitad del decaimiento cosenoidal, el LR debe ser ~50% del LR base."""
        opt, scheduler, base_lr, w_steps, t_steps = optimizer_and_scheduler
        mid_decay = w_steps + (t_steps - w_steps) // 2
        for _ in range(mid_decay):
            scheduler.step()

        lr_mid = opt.param_groups[0]['lr']
        expected_mid = base_lr * 0.5  # cos(π/2) × 0.5 + 0.5 = 0.5
        assert abs(lr_mid - expected_mid) < 0.005, (
            f"LR en mitad de decay debería ser ~{expected_mid:.4f}, got {lr_mid:.4f}"
        )


# ---------------------------------------------------------------------------
# Skip-warmup mode (post-rollback)
# ---------------------------------------------------------------------------

class TestSkipWarmup:
    def test_skip_mode_starts_at_full_lr(self):
        """Con skip=True, el LR empieza en 100% del base y decae directamente."""
        base_lr = 0.01
        t_steps = 1000
        c_step  = 0

        model = nn.Linear(5, 5)
        opt   = torch.optim.SGD(model.parameters(), lr=base_lr)
        sched = build_scheduler(opt, w_steps=200, t_steps=t_steps,
                                c_step=c_step, skip=True)

        lr_step0 = opt.param_groups[0]['lr']
        assert abs(lr_step0 - base_lr) < 1e-7, (
            f"En modo skip, LR inicial debería ser {base_lr}, got {lr_step0}"
        )

    def test_skip_mode_lr_is_decreasing(self):
        """Con skip=True, el LR debe decrecer monótonamente desde el inicio."""
        base_lr = 0.01
        t_steps = 500

        model = nn.Linear(5, 5)
        opt   = torch.optim.SGD(model.parameters(), lr=base_lr)
        sched = build_scheduler(opt, w_steps=100, t_steps=t_steps,
                                c_step=0, skip=True)

        lrs = [opt.param_groups[0]['lr']]
        for _ in range(t_steps):
            sched.step()
            lrs.append(opt.param_groups[0]['lr'])

        for i in range(len(lrs) - 1):
            assert lrs[i] >= lrs[i + 1] - 1e-9, (
                f"LR no monótono en modo skip: step {i}={lrs[i]:.6f} > step {i+1}={lrs[i+1]:.6f}"
            )


# ---------------------------------------------------------------------------
# momentum_update
# ---------------------------------------------------------------------------

class TestMomentumUpdate:
    def test_ema_update_interpolation(self):
        """momentum_update debe actualizar model_k con EMA: k = m*k + (1-m)*q."""
        model_q = nn.Linear(4, 4, bias=False)
        model_k = nn.Linear(4, 4, bias=False)

        # Inicializar con pesos conocidos
        nn.init.ones_(model_q.weight)
        nn.init.zeros_(model_k.weight)

        m = 0.9
        momentum_update(model_q, model_k, m)

        expected = m * 0.0 + (1.0 - m) * 1.0  # = 0.1
        actual   = model_k.weight.mean().item()
        assert abs(actual - expected) < 1e-6, (
            f"Después de EMA update: expected {expected:.4f}, got {actual:.4f}"
        )

    def test_momentum_1_freezes_key(self):
        """Con m=1.0, model_k no debe cambiar (EMA congelada)."""
        model_q = nn.Linear(4, 4, bias=False)
        model_k = nn.Linear(4, 4, bias=False)

        nn.init.ones_(model_q.weight)
        original_k = model_k.weight.clone()

        momentum_update(model_q, model_k, m=1.0)

        assert torch.allclose(model_k.weight, original_k), (
            "Con m=1.0, model_k no debería cambiar."
        )

    def test_momentum_0_copies_query(self):
        """Con m=0.0, model_k debe ser una copia exacta de model_q."""
        model_q = nn.Linear(4, 4, bias=False)
        model_k = nn.Linear(4, 4, bias=False)

        nn.init.constant_(model_q.weight, 3.14)
        nn.init.zeros_(model_k.weight)

        momentum_update(model_q, model_k, m=0.0)

        assert torch.allclose(model_k.weight, model_q.weight), (
            "Con m=0.0, model_k debería ser igual a model_q."
        )
