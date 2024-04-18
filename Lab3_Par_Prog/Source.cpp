#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <random>
#include <stdexcept>
#include "mpi.h"

std::vector<std::vector<int>> generateRandomMatrix(int rows, int cols) {
    std::vector<std::vector<int>> matrix(rows, std::vector<int>(cols));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 100);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = dis(gen);
        }
    }

    return matrix;
}

std::vector<int> multiplyBlock(const std::vector<std::vector<int>>& matrix1, int startRow, int endRow, const std::vector<std::vector<int>>& matrix2) {
    std::vector<int> result(matrix2[0].size(), 0);

    for (int i = startRow; i < endRow; ++i) {
        for (int j = 0; j < matrix2[0].size(); ++j) {
            for (int k = 0; k < matrix1[0].size(); ++k) {
                result[j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }

    return result;
}


std::vector<std::vector<int>> readMatrix(const std::string& filename, int& rows, int& cols) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Не удалось открыть файл.");
    }

    file >> rows >> cols;
    std::vector<std::vector<int>> matrix(rows, std::vector<int>(cols));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            file >> matrix[i][j];
        }
    }

    file.close();
    return matrix;
}

void writeMatrix(const std::vector<std::vector<int>>& matrix, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Не удалось открыть файл для записи.");
    }

    file << matrix.size() << " " << matrix[0].size() << std::endl;
    for (const auto& row : matrix) {
        for (const int& val : row) {
            file << val << " ";
        }
        file << std::endl;
    }

    file.close();
}

std::vector<std::vector<int>> parallelMultiplyMatricesFromFile(const std::string& file1, const std::string& file2, int myRank, int MPI_COMM_SIZE, int rows1, int rowsPerProcess) {
    int rowsA, colsA, rowsB, colsB;

    // Чтение первой матрицы из файла
    std::vector<std::vector<int>> matrix1 = readMatrix(file1, rowsA, colsA);

    // Чтение второй матрицы из файла
    std::vector<std::vector<int>> matrix2 = readMatrix(file2, rowsB, colsB);

    // Проверка совместимости размеров матриц
    if (colsA != rowsB) {
        throw std::runtime_error("Incorrect matrix sizes for multiplication.");
    }

    std::vector<std::vector<int>> result(matrix1.size(), std::vector<int>(colsB, 0));

    std::vector<int> blockResult(colsB, 0);
    int startRow = myRank * rowsPerProcess;
    int endRow = (myRank == MPI_COMM_SIZE - 1) ? rows1 : (myRank + 1) * rowsPerProcess;
    for (int i = startRow; i < endRow; ++i) {
        for (int j = 0; j < colsB; ++j) {
            for (int k = 0; k < colsA; ++k) {
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }

    return result;
}
// Функция для расчета доверительного интервала на основе времени выполнения
void calculateConfidenceInterval(const double& mean, const double& stdev, const int& numSamples) {
    double z = 1.96; // Значение z-критерия для 95% доверительного интервала
    double marginOfError = z * stdev / sqrt(numSamples);

    std::cout << "Confidence Interval for the Execution Time: [" << mean - marginOfError << ", " << mean + marginOfError << "]" << std::endl;
}


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int myRank, MPI_COMM_SIZE;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &MPI_COMM_SIZE);

    const int rows1 = 2000;
    const int cols1 = 2000;
    const int rows2 = 2000;
    const int cols2 = 2000;

    // Генерация случайных матриц и запись их в файлы
    std::vector<std::vector<int>> matrix1 = generateRandomMatrix(rows1, cols1);
    writeMatrix(matrix1, "matrix1.txt");
    std::vector<std::vector<int>> matrix2 = generateRandomMatrix(rows2, cols2);
    writeMatrix(matrix2, "matrix2.txt");

    // Чтение матриц из файлов
    int rows_read1, cols_read1, rows_read2, cols_read2;
    matrix1 = readMatrix("matrix1.txt", rows_read1, cols_read1);
    matrix2 = readMatrix("matrix2.txt", rows_read2, cols_read2);

    const int rowsPerProcess = rows1 / MPI_COMM_SIZE;
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<int>> result = parallelMultiplyMatricesFromFile("matrix1.txt", "matrix2.txt", myRank, MPI_COMM_SIZE, rows1, rowsPerProcess);
    auto end = std::chrono::high_resolution_clock::now();
    if (myRank == 0) {
        writeMatrix(result, "result_matrix.txt");
    }
    std::chrono::duration<double> duration = end - start;
    double meanTime = duration.count();

    // Вычисление стандартного отклонения для доверительного интервала
    double stdevTime = 0; // Для примера, здесь можно посчитать стандартное отклонение времени выполнения

    calculateConfidenceInterval(meanTime, stdevTime, 1); // 1 - количество измерений
    std::cout << "The scope of the task: " << rows1 * cols1 + rows2 * cols2 << " elements." << std::endl;
    std::cout << "Execution time: " << duration.count() << " seconds." << std::endl;

    MPI_Finalize();
    return 0;
}