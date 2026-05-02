
#include "curve.hpp"
#include "model_weights.hpp"
#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <random>

static std::mt19937 global_rng(std::chrono::high_resolution_clock::now().time_since_epoch().count());
static std::uniform_real_distribution<double> global_dist(0.0, 1.0);

double easeOutSine(double x) {
    return std::sin((x * 3.14159265358979323846) / 2.0);
}

double easeInOutQuad(double x) {
    return x < 0.5 ? 2.0 * x * x : 1.0 - std::pow(-2.0 * x + 2.0, 2.0) / 2.0;
}

double linear(double x) {
    return x;
}

std::vector<std::pair<double, double>> tweenPoints(
    const std::vector<std::pair<double, double>>& points,
    std::function<double(double)> tween,
    int targetPoints
) {
    if (points.empty() || targetPoints <= 0) return {};
    if (points.size() == 1) return std::vector<std::pair<double, double>>(targetPoints, points[0]);

    std::vector<std::pair<double, double>> res;
    res.reserve(targetPoints);

    for (int i = 0; i < targetPoints; i++) {
        double t = static_cast<double>(i) / (targetPoints - 1);
        double tweened_t = tween(t);
        int index = static_cast<int>(tweened_t * (points.size() - 1));
        index = std::max(0, std::min(index, static_cast<int>(points.size() - 1)));
        res.push_back(points[index]);
    }
    return res;
}

DenseLayer::DenseLayer(int input_size, int output_size, const std::string& activation)
    : input_size(input_size), output_size(output_size), activation(activation) {
    weights = Eigen::MatrixXd::Zero(input_size, output_size);
    biases = Eigen::VectorXd::Zero(output_size);
}

Eigen::VectorXd DenseLayer::forward(const Eigen::VectorXd& input) {
    this->input = input;
    output = weights.transpose() * input + biases;

    if (activation == "relu") {
        output = output.unaryExpr([](double x) { return x > 0 ? x : 0; });
    }
    else if (activation == "tanh") {
        output = output.array().tanh();
    }

    return output;
}

void DenseLayer::load_weights(const std::vector<std::vector<double>>& weights_data,
    const std::vector<double>& biases_data) {
    for (int i = 0; i < input_size; ++i) {
        for (int j = 0; j < output_size; ++j) {
            weights(i, j) = weights_data[i][j];
        }
    }

    for (int i = 0; i < output_size; ++i) {
        biases(i) = biases_data[i];
    }
}

DropoutLayer::DropoutLayer(double dropout_rate)
    : dropout_rate(dropout_rate) {
}

Eigen::VectorXd DropoutLayer::forward(const Eigen::VectorXd& input) {
    Eigen::VectorXd output = input;

    for (int i = 0; i < input.size(); i++) {
        if (global_dist(global_rng) < dropout_rate) {
            output(i) = 0.0;
        }
    }

    return output;
}

NeuralNetwork::NeuralNetwork() {}

NeuralNetwork::~NeuralNetwork() {
    for (auto* layer : layers) {
        delete layer;
    }
    layers.clear();
}

Eigen::VectorXd NeuralNetwork::forward(const Eigen::VectorXd& input) {
    Eigen::VectorXd current_input = input;
    for (auto* layer : layers) {
        current_input = layer->forward(current_input);
    }
    return current_input;
}

template<typename ModelWeightsType>
void loadWeightsFromModel(std::vector<Layer*>& layers) {
    ModelWeightsType model_weights;
    int dense_layer_idx = 0;

    if (dense_layer_idx < model_weights.dense_layers.size()) {
        const auto& layer_data = model_weights.dense_layers[dense_layer_idx];
        DenseLayer* layer = new DenseLayer(layer_data.input_size, layer_data.output_size, layer_data.activation);
        layer->load_weights(layer_data.weights, layer_data.biases);
        layers.push_back(layer);
        dense_layer_idx++;
    }

    layers.push_back(new DropoutLayer(0.2));

    if (dense_layer_idx < model_weights.dense_layers.size()) {
        const auto& layer_data = model_weights.dense_layers[dense_layer_idx];
        DenseLayer* layer = new DenseLayer(layer_data.input_size, layer_data.output_size, layer_data.activation);
        layer->load_weights(layer_data.weights, layer_data.biases);
        layers.push_back(layer);
        dense_layer_idx++;
    }

    if (dense_layer_idx < model_weights.dense_layers.size()) {
        const auto& layer_data = model_weights.dense_layers[dense_layer_idx];
        DenseLayer* layer = new DenseLayer(layer_data.input_size, layer_data.output_size, layer_data.activation);
        layer->load_weights(layer_data.weights, layer_data.biases);
        layers.push_back(layer);
    }
}

void NeuralNetwork::load_embedded_weights() {
    for (auto* layer : layers) {
        delete layer;
    }
    layers.clear();

    loadWeightsFromModel<MousePredictor::ModelWeights>(layers);
}

MMousePredictor::MMousePredictor()
    : model(nullptr), targetPoints(50) {
}

MMousePredictor:: ~MMousePredictor() {
    delete model;
}

void MMousePredictor::init(
    int width,
    int height,
    int target_radius,
    double mouse_step_size,
    int points
) {
    this->width = width;
    this->height = height;
    this->target_radius = target_radius;
    this->mouse_step_size = mouse_step_size;


    targetPoints = points;

    delete model;
    model = new NeuralNetwork();
    model->load_embedded_weights();
}

std::vector<std::pair<double, double>> MMousePredictor::moveTo(double target_x, double target_y) {
    return moveTo(0.0, 0.0, target_x, target_y);
}

std::vector<std::pair<double, double>> MMousePredictor::moveTo(
    double from_x, double from_y,
    double target_x, double target_y
) {
    Eigen::Vector2d start_pos(from_x, from_y);
    Eigen::Vector2d target_pos(target_x, target_y);

    std::vector<Eigen::Vector2d> path_eigen = draw_predicted_path(start_pos, target_pos);

    double clampedTarget_x = target_x - from_x;
    double clampedTarget_y = target_y - from_y;

    return convertToRelativePathCombined(path_eigen, clampedTarget_x, clampedTarget_y);
}

std::vector<std::pair<double, double>> MMousePredictor::moveToAbsolute(double target_x, double target_y) {
    return moveToAbsolute(0.0, 0.0, target_x, target_y);
}

std::vector<std::pair<double, double>> MMousePredictor::moveToAbsolute(
    double from_x, double from_y,
    double target_x, double target_y
) {
    Eigen::Vector2d start_pos(from_x, from_y);
    Eigen::Vector2d target_pos(target_x, target_y);

    std::vector<Eigen::Vector2d> path_eigen = draw_predicted_path(start_pos, target_pos);

    std::vector<std::pair<double, double>> path;
    path.reserve(path_eigen.size());
    for (const auto& p : path_eigen) {
        path.emplace_back(p[0], p[1]);
    }

    if (path.empty()) {
        return { {target_x, target_y} };
    }

    if (targetPoints > (int)path.size()) {
        return tweenPoints(path, easeOutSine, targetPoints);
    }

    return path;
}

std::vector<std::pair<double, double>> MMousePredictor::convertToRelativePathCombined(
    const std::vector<Eigen::Vector2d>& absolute_path,
    double clampedTarget_x,
    double clampedTarget_y
) {
    if (absolute_path.empty()) {
        return {};
    }

    std::vector<std::pair<double, double>> path;
    path.reserve(absolute_path.size());
    for (const auto& p : absolute_path) {
        path.emplace_back(p[0], p[1]);
    }

    std::vector<std::pair<double, double>> tweenedCoords;
    tweenedCoords = tweenPoints(path, easeOutSine, targetPoints);

    std::vector<std::pair<double, double>> relativePath;
    std::pair<double, double> origin = { 0.0, 0.0 };
    std::pair<double, double> extraNumbers = { 0.0, 0.0 };
    std::pair<double, double> totalOffset = { 0.0, 0.0 };

    bool xReached = false;
    bool yReached = false;

    for (const auto& point : tweenedCoords) {
        std::pair<double, double> currentPoint = { point.first, point.second };
        std::pair<double, double> offset = {
            currentPoint.first - origin.first,
            currentPoint.second - origin.second
        };

        extraNumbers.first += offset.first;
        extraNumbers.second += offset.second;

        double outputX = 0.0;
        double outputY = 0.0;
        bool hasOutput = false;

        if (!xReached && std::abs(extraNumbers.first) >= 1.0) {
            double roundedValue = std::round(extraNumbers.first);

            if (std::abs(totalOffset.first + roundedValue) <= std::abs(clampedTarget_x)) {
                outputX = roundedValue;
                totalOffset.first += roundedValue;
                extraNumbers.first -= roundedValue;
                hasOutput = true;
            }
            else {
                double remaining = clampedTarget_x - totalOffset.first;
                if (std::abs(remaining) > 1e-6) {
                    outputX = remaining;
                    totalOffset.first = clampedTarget_x;
                    hasOutput = true;
                }
                extraNumbers.first = 0.0;
                xReached = true;
            }
        }

        if (!yReached && std::abs(extraNumbers.second) >= 1.0) {
            double roundedValue = std::round(extraNumbers.second);

            if (std::abs(totalOffset.second + roundedValue) <= std::abs(clampedTarget_y)) {
                outputY = roundedValue;
                totalOffset.second += roundedValue;
                extraNumbers.second -= roundedValue;
                hasOutput = true;
            }
            else {
                double remaining = clampedTarget_y - totalOffset.second;
                if (std::abs(remaining) > 1e-6) {
                    outputY = remaining;
                    totalOffset.second = clampedTarget_y;
                    hasOutput = true;
                }
                extraNumbers.second = 0.0;
                yReached = true;
            }
        }

        if (hasOutput) {
            relativePath.push_back({ outputX, outputY });
        }

        origin = currentPoint;

        if (xReached && yReached) {
            break;
        }
        if (std::abs(totalOffset.first - clampedTarget_x) < 1e-6 &&
            std::abs(totalOffset.second - clampedTarget_y) < 1e-6) {
            break;
        }
    }

    double xError = 0.0;
    double yError = 0.0;

    if (std::abs(totalOffset.first - clampedTarget_x) > 1e-6) {
        xError = clampedTarget_x - totalOffset.first;
    }

    if (std::abs(totalOffset.second - clampedTarget_y) > 1e-6) {
        yError = clampedTarget_y - totalOffset.second;
    }

    if (std::abs(xError) > 1e-6 || std::abs(yError) > 1e-6) {
        relativePath.push_back({ xError, yError });
    }

    return relativePath;
}

Eigen::Vector2d MMousePredictor::clip_to_bounds(const Eigen::Vector2d& pos) {
    return pos;
}

Eigen::Vector2d MMousePredictor::predict_next_point(
    const Eigen::Vector2d& curr_pos,
    const Eigen::Vector2d& target_pos
) {
    Eigen::Vector2d distance_to_target = target_pos - curr_pos;
    double distance_magnitude = distance_to_target.norm();

    Eigen::Vector2d normalized_direction = Eigen::Vector2d::Zero();
    if (distance_magnitude > 1e-9) {
        normalized_direction = distance_to_target / distance_magnitude;
    }

    Eigen::VectorXd features(4);
    features << distance_to_target[0] / width,
        distance_to_target[1] / height,
        normalized_direction[0],
        normalized_direction[1];

    Eigen::VectorXd prediction = model->forward(features);
    Eigen::Vector2d next_pos = curr_pos + (prediction * mouse_step_size);

    return clip_to_bounds(next_pos);
}

std::vector<Eigen::Vector2d> MMousePredictor::draw_predicted_path(
    const Eigen::Vector2d& start_pos,
    const Eigen::Vector2d& target_pos
) {
    std::vector<Eigen::Vector2d> path;
    Eigen::Vector2d current_pos = start_pos;
    path.push_back(current_pos);

    for (int i = 0; i < 100; ++i) {
        Eigen::Vector2d next_pos = predict_next_point(current_pos, target_pos);
        path.push_back(next_pos);

        if ((next_pos - target_pos).norm() < target_radius) {
            break;
        }

        current_pos = next_pos;
    }

    return path;
}
