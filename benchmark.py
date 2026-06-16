import argparse
import time
import torch
from model import End2Race, End2RaceWithDelay

NUM_WARMUP = 200
NUM_MEASURE = 2000
NUM_FEATURES = 360


def parse_arguments():
    parser = argparse.ArgumentParser(description="Measure model inference frequency (Hz)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model .pth file")
    parser.add_argument("--mode", type=str, choices=["origin", "lidar_delay"], required=True,
                        help="Model mode: 'origin' or 'lidar_delay'")
    parser.add_argument("--hidden_scale", type=int, default=4)
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda/cpu). Auto-detected if not specified.")
    return parser.parse_args()


import numpy as np

def run_inference(model, lidar, speed, delay, hidden, use_lidar_delay):
    if use_lidar_delay:
        _, hidden = model(lidar, speed, delay, hidden)
    else:
        _, hidden = model(lidar, speed, hidden)
    return hidden


def measure_gpu_only(model, device, use_lidar_delay):
    """Pure GPU compute: tensors already on device."""
    lidar = torch.randn(1, 1, NUM_FEATURES, device=device)
    speed = torch.randn(1, 1, 1, device=device)
    delay = torch.randn(1, 1, 1, device=device) if use_lidar_delay else None
    hidden_size = model.gru.hidden_size
    hidden = torch.zeros(1, 1, hidden_size, device=device)

    model.eval()
    with torch.no_grad():
        for _ in range(NUM_WARMUP):
            hidden = run_inference(model, lidar, speed, delay, hidden, use_lidar_delay)
        if device.type == "cuda":
            torch.cuda.synchronize()

        if device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            for _ in range(NUM_MEASURE):
                hidden = run_inference(model, lidar, speed, delay, hidden, use_lidar_delay)
            end_event.record()
            torch.cuda.synchronize()
            elapsed_s = start_event.elapsed_time(end_event) / 1000.0
        else:
            t0 = time.perf_counter()
            for _ in range(NUM_MEASURE):
                hidden = run_inference(model, lidar, speed, delay, hidden, use_lidar_delay)
            elapsed_s = time.perf_counter() - t0

    return NUM_MEASURE / elapsed_s, elapsed_s / NUM_MEASURE * 1000.0


def measure_end_to_end(model, device, use_lidar_delay):
    """Realistic: numpy → tensor → device → inference (matches eval_singleagent.py)."""
    lidar_np = np.random.randn(NUM_FEATURES).astype(np.float32)
    speed_val = 0.0
    delay_val = 0.0
    hidden_size = model.gru.hidden_size
    hidden = torch.zeros(1, 1, hidden_size, device=device)

    model.eval()
    with torch.no_grad():
        for _ in range(NUM_WARMUP):
            lidar = torch.tensor(lidar_np, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            speed = torch.tensor([[[speed_val]]], dtype=torch.float32, device=device)
            delay = torch.tensor([[[delay_val]]], dtype=torch.float32, device=device) if use_lidar_delay else None
            hidden = run_inference(model, lidar, speed, delay, hidden, use_lidar_delay)
        if device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(NUM_MEASURE):
            lidar = torch.tensor(lidar_np, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            speed = torch.tensor([[[speed_val]]], dtype=torch.float32, device=device)
            delay = torch.tensor([[[delay_val]]], dtype=torch.float32, device=device) if use_lidar_delay else None
            hidden = run_inference(model, lidar, speed, delay, hidden, use_lidar_delay)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_s = time.perf_counter() - t0

    return NUM_MEASURE / elapsed_s, elapsed_s / NUM_MEASURE * 1000.0


if __name__ == "__main__":
    args = parse_arguments()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_lidar_delay = (args.mode == "lidar_delay")

    if use_lidar_delay:
        model = End2RaceWithDelay(hidden_scale=args.hidden_scale).to(device)
    else:
        model = End2Race(hidden_scale=args.hidden_scale).to(device)

    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=False))

    print(f"Model : {args.model_path}")
    print(f"Mode  : {args.mode}")
    print(f"Device: {device}")
    print(f"Warmup: {NUM_WARMUP} / Measure: {NUM_MEASURE}")
    print("-" * 35)

    hz_gpu, lat_gpu = measure_gpu_only(model, device, use_lidar_delay)
    hz_e2e, lat_e2e = measure_end_to_end(model, device, use_lidar_delay)

    print(f"[GPU compute only]")
    print(f"  Frequency : {hz_gpu:.1f} Hz")
    print(f"  Latency   : {lat_gpu:.3f} ms")
    print(f"[End-to-end (numpy → tensor → device → infer)]")
    print(f"  Frequency : {hz_e2e:.1f} Hz")
    print(f"  Latency   : {lat_e2e:.3f} ms")
