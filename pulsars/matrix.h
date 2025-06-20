#ifndef MATRIX_H
#define MATRIX_H

#include <vector>

template <typename T>
struct Matrix {
    std::vector<std::vector<T>> data;

    size_t rows;
    size_t cols;

    Matrix(size_t r, size_t c, const T& value = T())
        : data(r, std::vector<T>(c, value)), rows(r), cols(c) {}

    std::vector<T>& operator[](size_t i) { return data[i]; }
    const std::vector<T>& operator[](size_t i) const { return data[i]; }
};

#endif // MATRIX_H