#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdint>
#include <algorithm>
#include <stdexcept>
#include <string>

#include "onnxruntime_cxx_api.h"

using namespace std;
using namespace chrono;

const wstring MODEL_PATH =
    L"models/resnet18_fashion_mnist.onnx";

const string IMAGE_PATH =
    "dataset/fashion-mnist/t10k-images-idx3-ubyte";

const int NUM_IMAGES = 1000;
const int WARMUP_IMAGES = 50;
const int BATCH_SIZE = 32;
const int IMAGE_SIZE = 28 * 28;

uint32_t readBigEndian(ifstream& file) {
    unsigned char bytes[4];

    file.read(reinterpret_cast<char*>(bytes), 4);

    if (!file) {
        throw runtime_error("Failed to read IDX header.");
    }

    return (uint32_t(bytes[0]) << 24) |
           (uint32_t(bytes[1]) << 16) |
           (uint32_t(bytes[2]) << 8) |
           uint32_t(bytes[3]);
}

vector<float> loadImages() {
    ifstream file(IMAGE_PATH, ios::binary);

    if (!file) {
        throw runtime_error(
            "Could not open Fashion-MNIST images."
        );
    }

    uint32_t magic = readBigEndian(file);
    uint32_t total = readBigEndian(file);
    uint32_t rows = readBigEndian(file);
    uint32_t cols = readBigEndian(file);

    if (magic != 2051 || rows != 28 || cols != 28) {
        throw runtime_error(
            "Invalid Fashion-MNIST image file."
        );
    }

    if (total < NUM_IMAGES) {
        throw runtime_error("Not enough test images.");
    }

    vector<unsigned char> raw(
        NUM_IMAGES * IMAGE_SIZE
    );

    file.read(
        reinterpret_cast<char*>(raw.data()),
        static_cast<streamsize>(raw.size())
    );

    if (!file) {
        throw runtime_error("Could not read test images.");
    }

    vector<float> images(raw.size());

    for (size_t i = 0; i < raw.size(); i++) {
        images[i] =
            static_cast<float>(raw[i]) / 255.0f;
    }

    return images;
}

int main() {
    try {
        cout << "FASHIONVISION AI - C++ ONNX BENCHMARK\n";
        cout << "-------------------------------------\n";

        vector<float> images = loadImages();

        Ort::Env env(
            ORT_LOGGING_LEVEL_WARNING,
            "FashionVision"
        );

        Ort::SessionOptions options;

        Ort::Session session(
            env,
            MODEL_PATH.c_str(),
            options
        );

        Ort::AllocatorWithDefaultOptions allocator;

        auto inputName =
            session.GetInputNameAllocated(0, allocator);

        const char* inputNames[] = {
            inputName.get()
        };

        auto outputName =
            session.GetOutputNameAllocated(0, allocator);

        const char* outputNames[] = {
            outputName.get()
        };

        Ort::MemoryInfo memoryInfo =
            Ort::MemoryInfo::CreateCpu(
                OrtArenaAllocator,
                OrtMemTypeDefault
            );

        cout << "Model loaded successfully!\n";
        cout << "Images available: "
             << NUM_IMAGES << "\n";

        // WARM-UP

        cout << "\nRunning warm-up...\n";

        int64_t singleShape[] = {
            1, 1, 28, 28
        };

        for (int i = 0; i < WARMUP_IMAGES; i++) {

            Ort::Value inputTensor =
                Ort::Value::CreateTensor<float>(
                    memoryInfo,
                    images.data() + i * IMAGE_SIZE,
                    IMAGE_SIZE,
                    singleShape,
                    4
                );

            session.Run(
                Ort::RunOptions{ nullptr },
                inputNames,
                &inputTensor,
                1,
                outputNames,
                1
            );
        }

        // SINGLE-IMAGE BENCHMARK

        cout << "\nMeasuring single-image inference latency...\n";

        auto singleStart =
            high_resolution_clock::now();

        for (int i = 0; i < NUM_IMAGES; i++) {

            Ort::Value inputTensor =
                Ort::Value::CreateTensor<float>(
                    memoryInfo,
                    images.data() + i * IMAGE_SIZE,
                    IMAGE_SIZE,
                    singleShape,
                    4
                );

            session.Run(
                Ort::RunOptions{ nullptr },
                inputNames,
                &inputTensor,
                1,
                outputNames,
                1
            );
        }

        auto singleEnd =
            high_resolution_clock::now();

        double singleTime =
            duration<double>(
                singleEnd - singleStart
            ).count();

        double latencyMs =
            (singleTime * 1000.0) / NUM_IMAGES;

        double throughput =
            NUM_IMAGES / singleTime;

        // BATCH BENCHMARK

        cout << "\nMeasuring batch inference...\n";

        auto batchStart =
            high_resolution_clock::now();

        for (int i = 0; i < NUM_IMAGES; i += BATCH_SIZE) {

            int currentBatch =
                min(BATCH_SIZE, NUM_IMAGES - i);

            int64_t batchShape[] = {
                currentBatch, 1, 28, 28
            };

            Ort::Value inputTensor =
                Ort::Value::CreateTensor<float>(
                    memoryInfo,
                    images.data() + i * IMAGE_SIZE,
                    static_cast<size_t>(
                        currentBatch * IMAGE_SIZE
                    ),
                    batchShape,
                    4
                );

            session.Run(
                Ort::RunOptions{ nullptr },
                inputNames,
                &inputTensor,
                1,
                outputNames,
                1
            );
        }

        auto batchEnd =
            high_resolution_clock::now();

        double batchTime =
            duration<double>(
                batchEnd - batchStart
            ).count();

        // RESULTS

        cout << fixed << setprecision(2);

        cout << "\n========== BENCHMARK RESULTS ==========\n";

        cout << "Images tested: "
             << NUM_IMAGES << "\n";

        cout << "Single-image total time: "
             << singleTime * 1000.0 << " ms\n";

        cout << "Average latency: "
             << latencyMs << " ms/image\n";

        cout << "Throughput: "
             << throughput << " images/second\n";

        cout << "\nBatch size: "
             << BATCH_SIZE << "\n";

        cout << "Batch inference time: "
             << batchTime * 1000.0 << " ms\n";

        cout << "Batch throughput: "
             << NUM_IMAGES / batchTime
             << " images/second\n";

        cout << "\nC++ ONNX BENCHMARK COMPLETED!\n";
    }
    catch (const exception& error) {

        cerr << "\nERROR: "
             << error.what() << "\n";

        return 1;
    }

    return 0;
}