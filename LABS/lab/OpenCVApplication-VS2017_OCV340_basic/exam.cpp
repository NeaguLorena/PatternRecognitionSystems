#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include "IPFunctions.cpp"


void thetas(){

    int n = 4;
    Point points[n + 1];
    float th0, th1;
    float sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;

    points[0] = Point(60, 160);
    points[1] = Point(70, 160);
    points[2] = Point(80, 170);
    points[3] = Point(90, 180);


    for (int i = 0; i < n; i++) {
        sumX += points[i].x;
        sumXX += points[i].x * points[i].x;
        sumY += points[i].y;
        sumXY += points[i].x * points[i].y;
    }

    th1 = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    th0 = (sumY - th1 * sumX) / n;
    cout << endl << th1 << endl<< th0;

}

void compute_mean_1(Mat_<int> I, float feature_mean[361], int n, int m) {
    //n = number of features
    //m = numeber of examples

    for (int i = 0; i < n; i++) {
        int sum = 0;
        for (int k = 0; k < m; k++) {
            sum += I(k, i);
        }
        float mean = (float) sum / m;

        feature_mean[i] = mean;
    }
}

int main(){
    thetas();
    return 0;
}
