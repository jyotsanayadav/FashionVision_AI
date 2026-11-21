import time
import threading
import psutil
from pathlib import Path

import numpy as np
import onnxruntime as ort


MODEL_PATH = "models/resnet18_fashion_mnist.onnx"
IMAGE_PATH = "dataset/fashion-mnist/t10k-images-idx3-ubyte"

NUM_IMAGES = 1000
WARMUP_IMAGES = 50


def load_images(path, count):
    with open(path, "rb") as file:
        magic, total, rows, cols = np.frombuffer(
            file.read(16), dtype=">i4"
        )

        if magic != 2051:
            raise ValueError("Invalid Fashion-MNIST image file.")

        count = min(count, total)

        images = np.frombuffer(
            file.read(count * rows * cols),
            dtype=np.uint8
        )

    images = images.reshape(count, 1, rows, cols)
    return images.astype(np.float32) / 255.0


def main():
    print("FASHIONVISION AI - PYTHON ONNX BENCHMARK")
    print("----------------------------------------")

    images = load_images(IMAGE_PATH, NUM_IMAGES)

    options = ort.SessionOptions()
    session = ort.InferenceSession(
        MODEL_PATH,
        sess_options=options,
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name

    print("Model loaded successfully!")
    print(f"Images available: {len(images)}")

    # Warm-up
    print("\nRunning warm-up...")
    for i in range(WARMUP_IMAGES):
        session.run(None, {input_name: images[i:i+1]})

    # RAM monitoring
    process = psutil.Process()
    baseline_ram = process.memory_info().rss / (1024 * 1024)
    peak_ram = [baseline_ram]
    stop_monitor = threading.Event()

    def monitor_ram():
        while not stop_monitor.is_set():
            current_ram = process.memory_info().rss / (1024 * 1024)
            peak_ram[0] = max(peak_ram[0], current_ram)
            stop_monitor.wait(0.01)

    monitor = threading.Thread(target=monitor_ram, daemon=True)
    monitor.start()    
# Single-image latency
    print("\nMeasuring single-image inference latency...")

    start = time.perf_counter()

    for image in images:
        session.run(None, {input_name: image[np.newaxis, ...]})

    elapsed = time.perf_counter() - start

    average_latency = elapsed * 1000 / len(images)
    throughput = len(images) / elapsed

    print("\n========== BENCHMARK RESULTS ==========")
    print(f"Images tested: {len(images)}")
    print(f"Total inference time: {elapsed * 1000:.2f} ms")
    print(f"Average latency: {average_latency:.2f} ms/image")
    print(f"Throughput: {throughput:.2f} images/second")

    # Batch inference
    print("\nMeasuring batch inference...")

    batch_size = 32
    start = time.perf_counter()

    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        session.run(None, {input_name: batch})

    batch_elapsed = time.perf_counter() - start

    print(f"Batch size: {batch_size}")
    print(f"Batch inference time: {batch_elapsed * 1000:.2f} ms")
    print(
        f"Batch throughput: "
        f"{len(images) / batch_elapsed:.2f} images/second"
    )

    # Stop RAM monitoring
    stop_monitor.set()
    monitor.join()

    final_ram = process.memory_info().rss / (1024 * 1024)
    peak_ram[0] = max(peak_ram[0], final_ram)

    print("\n========== RAM USAGE ==========")
    print(f"Baseline process RAM: {baseline_ram:.2f} MB")
    print(f"Peak process RAM: {peak_ram[0]:.2f} MB")
    print(f"Additional RAM used: {max(0, peak_ram[0] - baseline_ram):.2f} MB")    
    print("\nPYTHON ONNX BENCHMARK COMPLETED!")


if __name__ == "__main__":
    main()