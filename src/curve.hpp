#ifndef CURVE_HPP
#define CURVE_HPP

#include <iostream>
#include <string>
#include <vector>
#include <deque>
#include <random>
#include <cmath>
#include <functional>
#include <memory>
#include <Eigen/Dense>

class Layer {
public:
    virtual ~Layer() = default;
    virtual Eigen::VectorXd forward(const Eigen::VectorXd& input) = 0;
};

class DenseLayer : public Layer {
public:
    DenseLayer(int input_size, int output_size, const std::string& activation);
    Eigen::VectorXd forward(const Eigen::VectorXd& input) override;
    void load_weights(const std::vector<std::vector<double>>& weights_data,
        const std::vector<double>& biases_data);

private:
    int input_size;
    int output_size;
    std::string activation;
    Eigen::MatrixXd weights;
    Eigen::VectorXd biases;
    Eigen::VectorXd input;
    Eigen::VectorXd output;
};

class DropoutLayer : public Layer {
public:
    DropoutLayer(double dropout_rate);
    Eigen::VectorXd forward(const Eigen::VectorXd& input) override;

private:
    double dropout_rate;
};


class NeuralNetwork {
public:
    NeuralNetwork();
    ~NeuralNetwork();
    Eigen::VectorXd forward(const Eigen::VectorXd& input);
    void load_embedded_weights();

private:
    std::vector<Layer*> layers;
};

class MMousePredictor {
public:
    MMousePredictor();
    ~MMousePredictor();

    void init(int width = 800, int height = 600, int target_radius = 8, double mouse_step_size = 4.0, int points = 50);

    std::vector<std::pair<double, double>> moveTo(double target_x, double target_y);
    std::vector<std::pair<double, double>> moveTo(double from_x, double from_y, double target_x, double target_y);

    std::vector<std::pair<double, double>> moveToAbsolute(double target_x, double target_y);
    std::vector<std::pair<double, double>> moveToAbsolute(double from_x, double from_y, double target_x, double target_y);
private:
    Eigen::Vector2d clip_to_bounds(const Eigen::Vector2d& pos);
    Eigen::Vector2d predict_next_point(const Eigen::Vector2d& curr_pos, const Eigen::Vector2d& target_pos);
    std::vector<Eigen::Vector2d> draw_predicted_path(const Eigen::Vector2d& start_pos, const Eigen::Vector2d& target_pos);

    std::vector<std::pair<double, double>> convertToRelativePathCombined(const std::vector<Eigen::Vector2d>& absolute_path, double clampedTarget_x, double clampedTarget_y);

    NeuralNetwork* model;
    int targetPoints;

    int width;
    int height;
    int target_radius;
    double mouse_step_size;

};

double easeOutSine(double x);
double easeInOutQuad(double x);
double linear(double x);

std::vector<std::pair<double, double>> tweenPoints(
    const std::vector<std::pair<double, double>>& points,
    std::function<double(double)> tween,
    int targetPoints
);

#endif // CURVE_HPP
