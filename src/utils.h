#include <vector>
#include <cblas.h>

class Utils {
    public:
        void multiply_matrix_matrix(double *A, double *B, double *C, int N);
        void multiply_matrix_vector(double *A, double *B, double *C, int N);

        void multiply_matrix_matrix(
                std::vector<std::vector<double> > &matrix_a,
                std::vector<std::vector<double> > &matrix_b,
                std::vector<std::vector<double> > &result);

        void multiply_matrix_vector(
                std::vector<std::vector<double> > &matrix,
                std::vector<double> &vector,
                std::vector<double> &result);

        void multiply_matrix_matrix_cblas(
                double *A, double *B, double *C, int N);

        void multiply_matrix_vector_cblas(
                double *A, double *B, double *C, int N);
};
