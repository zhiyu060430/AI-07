from types import SimpleNamespace

from firephantom.synthesis.diffusion_server import _device_and_dtype


def _torch_backends(*, mps: bool, cuda: bool):
    return SimpleNamespace(
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
        cuda=SimpleNamespace(is_available=lambda: cuda),
        float16=object(),
        float32=object(),
    )


def test_local_diffusion_uses_float32_on_mps():
    torch = _torch_backends(mps=True, cuda=False)

    assert _device_and_dtype(torch) == ("mps", torch.float32)


def test_local_diffusion_keeps_cuda_half_precision_and_cpu_float32():
    cuda_torch = _torch_backends(mps=False, cuda=True)
    cpu_torch = _torch_backends(mps=False, cuda=False)

    assert _device_and_dtype(cuda_torch) == ("cuda", cuda_torch.float16)
    assert _device_and_dtype(cpu_torch) == ("cpu", cpu_torch.float32)


def test_local_diffusion_memory_guard_blocks_low_headroom(monkeypatch):
    import firephantom.synthesis.diffusion_server as server

    monkeypatch.setattr(server, "memory_status", lambda: {
        "state": "critical", "available_gib": 1.2, "minimum_available_gib": 3.0,
    })
    try:
        server._check_memory_headroom()
    except RuntimeError as exc:
        assert "memory guard" in str(exc)
    else:
        raise AssertionError("low memory should block a cold diffusion load")


def test_local_diffusion_memory_guard_can_be_disabled(monkeypatch):
    import firephantom.synthesis.diffusion_server as server

    monkeypatch.setenv("FIREPHANTOM_MEMORY_GUARD", "0")
    monkeypatch.setattr(server, "memory_status", lambda: {
        "state": "critical", "available_gib": 1.2, "minimum_available_gib": 3.0,
    })
    server._check_memory_headroom()
