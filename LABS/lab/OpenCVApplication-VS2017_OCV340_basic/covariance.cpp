#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
//#include "IPFunctions.cpp"
using namespace cv;
using namespace std;

void compute_mean_1(Mat_<float> I, float feature_mean[361], int n, int m) {
    //n = number of features
    //m = numeber of examples

    for (int i = 0; i < n; i++) {
        float sum = 0;
        for (int k = 0; k < m; k++) {
            sum += I(k, i);
        }
        float mean = (float) sum / m;

        feature_mean[i] = mean;
    }
}

Mat_<float> compute_covariance_1(float mean[], Mat_<float> I, int n, int m) {
    Mat_<float> covariance_matrix(n, n);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0;
            for (int k = 0; k < m; k++) {
                sum += (I(k, i) - mean[i]) * (I(k, j) - mean[j]);
            }

            float cij = 1. / m * sum;
            covariance_matrix(i, j) = cij;
        }
    }
    return covariance_matrix;
}

void compute_std_1(float mean[], Mat_<int> I, float st_dev[], int n, int m) {

    for (int i = 0; i < n; i++) {
        float sum = 0;
        for (int k = 0; k < m; k++) {
            float temp = I(k, i) - mean[i];
            sum += temp * temp;
        }

        st_dev[i] = sqrt(1.0 / m * sum);
    }
}

Mat_<float> compute_correlation_coefficient_1(float feature_std[], Mat_<float> covariance_matrix, int n) {
//    std::ofstream myfile;
//    myfile.open("correlation_coefficient.csv");
    Mat_<float> correlation_coefficient(361, 361);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float cij = covariance_matrix[i][j] / (feature_std[i] * feature_std[j]);
            correlation_coefficient(i, j) = cij;
        }
    }
    return correlation_coefficient;
}
int main() {
    float values[] = {20, 32, 15, 8, 17, 5, 41, 12, 20};
    Mat_<float> X(3, 3, values);

    //MEAN
    float mean_val[4];
    compute_mean_1(X, mean_val, 3, 3);

    cout << "Means: " << endl;

    for (int i = 0; i < 3; i++) {
        cout << mean_val[i] << " ";
    }

    //COVARIANCE
    Mat_<float> cov_mat, corell_mat;
    cov_mat = compute_covariance_1(mean_val, X, 3, 3);

    cout << "\n\nCovariances: \n";
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            cout << cov_mat(i, j) << " ";
        }

        cout << "\n";
    }



//
//    //CORELLATION COEFFICIENTS
//    float std[4];
//
//    compute_std_1(mean_val, X, std, 3, 4);
//    corell_mat = compute_correlation_coefficient_1(std, cov_mat, 3);
//
//    cout << "\nCorrelation coefficients:" << endl;
//
//    for (int i = 0; i < 3; i++) {
//        for (int j = 0; j < 3; j++) {
//            cout << corell_mat(i, j) << " ";
//        }
//
//        cout << "\n";
//    }

    return 0;
}

//void correlation_chart_1(int r1, int c1, int r2, int c2, uchar I[400][361], Mat_<float> correlation_coefficient) {
//    int i = r1 * 19 + c1;
//    int j = r1 * 19 + c2;
//
//    Mat_<uchar> chart(256, 256, 255);
//
//    for (int k = 0; k < 400; k++) {
//        chart(I[k][i], I[k][j]) = 0;
//    }
//
//    cout << correlation_coefficient[i][j] << " ";
//
//    imshow("correlation chart", chart);
//    waitKey(0);
//}