#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>  

//c++ -Ofast -Wall -shared -std=c++20 -fPIC $(python3.12 -m pybind11 --includes) transformer.cpp -o transformer$(python3.12-config --extension-suffix)

namespace py = pybind11;

class TransformerEncoderLayer {
public:
    TransformerEncoderLayer(size_t d_model, size_t num_heads)
        : d_model(d_model), num_heads(num_heads) {
        // Inicializamos pesos (en la vida real, estos serían aleatorios o entrenados)
        query_weights.resize(d_model * d_model, 0.1); // Pesos de la capa de query
        key_weights.resize(d_model * d_model, 0.1);   // Pesos de la capa de key
        value_weights.resize(d_model * d_model, 0.1); // Pesos de la capa de value
    }

    // Self-Attention mechanism
    std::vector<std::vector<double>> self_attention(const std::vector<std::vector<double>>& x) {
        size_t seq_len = x.size();  // Longitud de la secuencia de entrada

        std::vector<std::vector<double>> attention_output(seq_len, std::vector<double>(d_model, 0.0));

        for (size_t i = 0; i < seq_len; ++i) {
            for (size_t j = 0; j < seq_len; ++j) {
                // Simulación de un cálculo básico de atención (dot product)
                double score = dot_product(x[i], x[j]) / std::sqrt(d_model);
                attention_output[i] = add_vectors(attention_output[i], scalar_multiply(x[j], score));
            }
        }
        return attention_output;
    }

    // Forward function combining self-attention with feedforward
    std::vector<std::vector<double>> forward(const std::vector<std::vector<double>>& x) {
        // Primero, aplicar self-attention
        auto attention_output = self_attention(x);

        // Aplicar una capa feedforward simple (en la vida real, esto sería más complejo)
        for (auto& vec : attention_output) {
            for (auto& elem : vec) {
                elem = std::tanh(elem);  // Función de activación simplificada
            }
        }

        return attention_output;
    }

private:
    size_t d_model;  // Dimensionalidad del modelo
    size_t num_heads;  // Número de cabezas de atención

    // Pesos (para simplificar, están inicializados en valores constantes)
    std::vector<double> query_weights;
    std::vector<double> key_weights;
    std::vector<double> value_weights;

    // Función de producto punto entre dos vectores
    double dot_product(const std::vector<double>& a, const std::vector<double>& b) {
        double result = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            result += a[i] * b[i];
        }
        return result;
    }

    // Multiplicar un vector por un escalar
    std::vector<double> scalar_multiply(const std::vector<double>& vec, double scalar) {
        std::vector<double> result(vec.size());
        for (size_t i = 0; i < vec.size(); ++i) {
            result[i] = vec[i] * scalar;
        }
        return result;
    }

    // Sumar dos vectores
    std::vector<double> add_vectors(const std::vector<double>& a, const std::vector<double>& b) {
        std::vector<double> result(a.size());
        for (size_t i = 0; i < a.size(); ++i) {
            result[i] = a[i] + b[i];
        }
        return result;
    }
};

// Código de binding con PyBind11
PYBIND11_MODULE(transformer, m) {
    py::class_<TransformerEncoderLayer>(m, "TransformerEncoderLayer")
        .def(py::init<size_t, size_t>())  // Constructor: d_model, num_heads
        .def("forward", &TransformerEncoderLayer::forward);  // Método forward
}
