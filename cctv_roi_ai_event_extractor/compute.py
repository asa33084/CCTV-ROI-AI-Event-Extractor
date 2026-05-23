try:
    import torch
except Exception:
    torch = None


def get_auto_device_info() -> dict:
    """Prefer NVIDIA CUDA when available, otherwise fall back to CPU."""
    info = {
        "device": "cpu",
        "name": "CPU",
        "source": "cpu",
    }

    try:
        if torch is not None and torch.cuda.is_available():
            count = torch.cuda.device_count()

            if count > 0:
                for i in range(count):
                    name = torch.cuda.get_device_name(i)
                    if "nvidia" in name.lower():
                        return {
                            "device": f"cuda:{i}",
                            "name": name,
                            "source": "cuda",
                        }

                name = torch.cuda.get_device_name(0)
                return {
                    "device": "cuda:0",
                    "name": name,
                    "source": "cuda",
                }
    except Exception as e:
        info["name"] = f"CPU（CUDA偵測失敗：{e}）"

    return info


def list_available_compute_devices():
    """Return UI-friendly compute device choices."""
    devices = [{
        "value": "cpu",
        "label": "CPU",
        "name": "CPU",
        "kind": "cpu",
    }]

    try:
        if torch is not None and torch.cuda.is_available():
            count = torch.cuda.device_count()
            gpu_names = []
            for i in range(count):
                name = torch.cuda.get_device_name(i)
                gpu_names.append(name)
                devices.append({
                    "value": f"cuda:{i}",
                    "label": f"GPU {i}: {name}",
                    "name": name,
                    "kind": "gpu",
                })

            if count > 1:
                devices.append({
                    "value": ",".join(str(i) for i in range(count)),
                    "label": "Multi-GPU: " + " | ".join(f"{i}:{name}" for i, name in enumerate(gpu_names)),
                    "name": ", ".join(gpu_names),
                    "kind": "multi_gpu",
                })
    except Exception:
        pass

    return devices


def describe_available_compute_devices():
    lines = []
    for item in list_available_compute_devices():
        if item["kind"] == "cpu":
            lines.append("cpu | CPU")
        else:
            lines.append(f"{item['value']} | {item['label']}")
    return "\n".join(lines)
