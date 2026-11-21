#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <chrono>
#include <array>
#include <iomanip>

#include <onnxruntime_cxx_api.h>

using namespace std;
using namespace chrono;

const string IMAGE_PATH =
    "dataset/fashion-mnist/t10k-images-idx3-ubyte";

const string LABEL_PATH =
    "dataset/fashion-mnist/t10k-labels-idx1-ubyte";

const string CONFUSION_PATH =
    "results/cpp_confusion_matrix.csv";

const string REPORT_PATH =
    "results/cpp_classification_report.csv";

const int NUM_IMAGES = 10000;
const int NUM_CLASSES = 10;

const vector<string> CLASSES = {
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
};

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

vector<float> loadImage(ifstream& file, int index) {
    file.clear();
    file.seekg(16 + static_cast<streamoff>(index) * 784);

    vector<unsigned char> pixels(784);

    file.read(reinterpret_cast<char*>(pixels.data()), 784);

    if (!file) {
        throw runtime_error("Failed to read image.");
    }

    vector<float> image(784);

    for (int i = 0; i < 784; i++) {
        image[i] = pixels[i] / 255.0f;
    }

    return image;
}

int loadLabel(ifstream& file, int index) {
    file.clear();
    file.seekg(8 + index);

    unsigned char label;
    file.read(reinterpret_cast<char*>(&label), 1);

    if (!file) {
        throw runtime_error("Failed to read label.");
    }

    return label;
}

int main() {
    try {
        cout << "FASHIONVISION AI - FULL TEST EVALUATION\n";
        cout << "---------------------------------------\n";

        ifstream imageFile(IMAGE_PATH, ios::binary);
        ifstream labelFile(LABEL_PATH, ios::binary);

        if (!imageFile || !labelFile) {
            throw runtime_error("Fashion-MNIST test files not found.");
        }

        uint32_t imageMagic = readBigEndian(imageFile);
        uint32_t imageCount = readBigEndian(imageFile);
        uint32_t rows = readBigEndian(imageFile);
        uint32_t cols = readBigEndian(imageFile);

        uint32_t labelMagic = readBigEndian(labelFile);
        uint32_t labelCount = readBigEndian(labelFile);

        if (imageMagic != 2051 || labelMagic != 2049 ||
            rows != 28 || cols != 28) {
            throw runtime_error("Invalid Fashion-MNIST dataset.");
        }

        if (imageCount != NUM_IMAGES || labelCount != NUM_IMAGES) {
            throw runtime_error("Expected exactly 10,000 test images.");
        }

        // Initialize ONNX Runtime.
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "FashionVisionAI");

        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(2);
        options.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_ALL
        );

        Ort::Session session(
            env,
            L"models/resnet18_fashion_mnist.onnx",
            options
        );

        cout << "\nModel loaded successfully!\n";
        cout << "Test images: " << NUM_IMAGES << "\n";

        Ort::MemoryInfo memoryInfo =
            Ort::MemoryInfo::CreateCpu(
                OrtArenaAllocator,
                OrtMemTypeDefault
            );

        vector<int64_t> inputShape = {1, 1, 28, 28};

        Ort::AllocatorWithDefaultOptions allocator;

        auto inputName = session.GetInputNameAllocated(0, allocator);
        auto outputName = session.GetOutputNameAllocated(0, allocator);

        const char* inputNames[] = {inputName.get()};
        const char* outputNames[] = {outputName.get()};

        // Confusion matrix: rows = actual, columns = predicted.
        array<array<int, NUM_CLASSES>, NUM_CLASSES> confusion = {};

        int totalCorrect = 0;
        double totalInferenceMs = 0.0;

        auto totalStart = high_resolution_clock::now();

        // Evaluate every test image.
        for (int i = 0; i < NUM_IMAGES; i++) {

            vector<float> image = loadImage(imageFile, i);
            int actualLabel = loadLabel(labelFile, i);

            if (actualLabel < 0 || actualLabel >= NUM_CLASSES) {
                throw runtime_error("Invalid class label.");
            }

            Ort::Value inputTensor =
                Ort::Value::CreateTensor<float>(
                    memoryInfo,
                    image.data(),
                    image.size(),
                    inputShape.data(),
                    inputShape.size()
                );

            auto inferenceStart = high_resolution_clock::now();

            auto outputs = session.Run(
                Ort::RunOptions{nullptr},
                inputNames,
                &inputTensor,
                1,
                outputNames,
                1
            );

            auto inferenceEnd = high_resolution_clock::now();

            totalInferenceMs +=
                duration<double, milli>(
                    inferenceEnd - inferenceStart
                ).count();

            float* logits =
                outputs[0].GetTensorMutableData<float>();

            int predictedLabel =
                max_element(logits, logits + NUM_CLASSES) - logits;

            confusion[actualLabel][predictedLabel]++;

            if (predictedLabel == actualLabel) {
                totalCorrect++;
            }

            if ((i + 1) % 1000 == 0) {
                cout << "Processed "
                     << i + 1 << "/" << NUM_IMAGES
                     << " images\n";
            }
        }

        auto totalEnd = high_resolution_clock::now();

        double totalExecutionMs =
            duration<double, milli>(
                totalEnd - totalStart
            ).count();

        // Calculate per-class metrics.
        array<int, NUM_CLASSES> truePositive = {};
        array<int, NUM_CLASSES> falsePositive = {};
        array<int, NUM_CLASSES> falseNegative = {};
        array<int, NUM_CLASSES> support = {};

        double macroPrecision = 0.0;
        double macroRecall = 0.0;
        double macroF1 = 0.0;

        cout << "\n========== CLASSIFICATION REPORT ==========\n";

        cout << left << setw(16) << "Class"
             << right << setw(10) << "Support"
             << setw(12) << "Precision"
             << setw(12) << "Recall"
             << setw(12) << "F1-score\n";

        cout << string(62, '-') << "\n";

        ofstream report(REPORT_PATH);

        if (!report) {
            throw runtime_error("Could not create classification report.");
        }

        report << fixed << setprecision(4);

        report << "Class,Support,True Positive,False Positive,"
                  "False Negative,Precision,Recall,F1 Score\n";

        for (int c = 0; c < NUM_CLASSES; c++) {

            for (int j = 0; j < NUM_CLASSES; j++) {
                support[c] += confusion[c][j];
                if(j!=c){
                falsePositive[c] += confusion[j][c];
}
            }

            truePositive[c] = confusion[c][c];

            falseNegative[c] = support[c] - truePositive[c];

            double precision = 0.0;
            double recall = 0.0;
            double f1 = 0.0;

            if (truePositive[c] + falsePositive[c] > 0) {
                precision =
                    static_cast<double>(truePositive[c]) /
                    (truePositive[c] + falsePositive[c]);
            }

            if (support[c] > 0) {
                recall =
                    static_cast<double>(truePositive[c]) /
                    support[c];
            }

            if (precision + recall > 0.0) {
                f1 = 2.0 * precision * recall /
                     (precision + recall);
            }

            macroPrecision += precision;
            macroRecall += recall;
            macroF1 += f1;

            cout << left << setw(16) << CLASSES[c]
                 << right << setw(10) << support[c]
                 << setw(12) << fixed << setprecision(4) << precision
                 << setw(12) << recall
                 << setw(12) << f1 << "\n";

            report << CLASSES[c] << ","
                   << support[c] << ","
                   << truePositive[c] << ","
                   << falsePositive[c] << ","
                   << falseNegative[c] << ","
                   << precision << ","
                   << recall << ","
                   << f1 << "\n";
        }

        macroPrecision /= NUM_CLASSES;
        macroRecall /= NUM_CLASSES;
        macroF1 /= NUM_CLASSES;

        double accuracy =
            static_cast<double>(totalCorrect) / NUM_IMAGES;

        report << "Macro Average," << NUM_IMAGES
               << ",,,,,"
               << macroPrecision << ","
               << macroRecall << ","
               << macroF1 << "\n";

        report << "Overall Accuracy," << NUM_IMAGES
               << ",,,,,,"
               << accuracy << "\n";

        report.close();

        // Save confusion matrix.
        ofstream confusionFile(CONFUSION_PATH);

        if (!confusionFile) {
            throw runtime_error("Could not create confusion matrix.");
        }

        confusionFile << "Actual/Predicted";

        for (const string& name : CLASSES) {
            confusionFile << "," << name;
        }

        confusionFile << "\n";

        for (int i = 0; i < NUM_CLASSES; i++) {
            confusionFile << CLASSES[i];

            for (int j = 0; j < NUM_CLASSES; j++) {
                confusionFile << "," << confusion[i][j];
            }

            confusionFile << "\n";
        }

        confusionFile.close();

        // Display final results.
        cout << "\n========== OVERALL RESULTS ==========\n";

        cout << "Images tested: " << NUM_IMAGES << "\n";
        cout << "Correct predictions: " << totalCorrect << "\n";
        cout << "Incorrect predictions: "
             << NUM_IMAGES - totalCorrect << "\n";

        cout << fixed << setprecision(2);

        cout << "Accuracy: " << accuracy * 100.0 << "%\n";
        cout << "Macro Precision: " << macroPrecision * 100.0 << "%\n";
        cout << "Macro Recall: " << macroRecall * 100.0 << "%\n";
        cout << "Macro F1-score: " << macroF1 * 100.0 << "%\n";

        cout << "Average inference time: "
             << totalInferenceMs / NUM_IMAGES
             << " ms/image\n";

        cout << "Total inference time: "
             << totalInferenceMs << " ms\n";

        cout << "Total execution time: "
             << totalExecutionMs << " ms\n";

        cout << "\nReports saved:\n";
        cout << CONFUSION_PATH << "\n";
        cout << REPORT_PATH << "\n";

        cout << "\nFULL TEST EVALUATION COMPLETED!\n";
    }
    catch (const exception& error) {
        cerr << "\nERROR: " << error.what() << endl;
        return 1;
    }

    return 0;
}