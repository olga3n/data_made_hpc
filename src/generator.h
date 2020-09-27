#include <vector>
#include <random>
#include <math.h>

class Generator {
    public:
        void generate_vector(int N, std::vector<double> &vector);
        void generate_vector(int N, double *vector);

        void generate_matrix(int N, std::vector<std::vector<double> > &matrix);
        void generate_matrix(int N, double *matrix);
};
