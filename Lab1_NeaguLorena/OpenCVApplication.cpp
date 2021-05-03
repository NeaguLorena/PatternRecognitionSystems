#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

using namespace cv;
using namespace std;

int main() {
    FILE *f = fopen("Points/points0.txt", "r");
    int n;
    fscanf(f, "%d", &n);
    Point points[n + 1];

    for (int i = 0; i < n; i++) {
        float x, y;
        fscanf(f, "%f%f", &x, &y);
        points[i] = Point(x, y);
    }
    Mat img(500, 500, CV_8UC3);

    for (int i = 0; i < 500; i++)
        for (int j = 0; j < 500; j++) {
            img.at<Vec3b>(i, j)[0] = 255;
            img.at<Vec3b>(i, j)[1] = 255;
            img.at<Vec3b>(i, j)[2] = 255;
        }

    for (int i = 0; i < n; i++) {
        if (points[i].x > 0 & points[i].y > 0) {
            circle(img, points[i], 1, Scalar(0, 0, 0), FILLED, 1, 0);
        }
    }

    float th0, th1;
    float sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
    int x1, y1, x2, y2;

    for (int i = 0; i < n; i++) {
        sumX += points[i].x;
        sumXX += points[i].x * points[i].x;
        sumY += points[i].y;
        sumXY += points[i].x * points[i].y;
    }

    th1 = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    th0 = (sumY - th1 * sumX) / n;

    if (fabs(th1) < 1) {
        x1 = 0;
        y1 = th0 + th1 * x1;
        x2 = img.cols - 1;
        y2 = th0 + th1 * x2;
    } else {
        y1 = 0;
        x1 = (y1 - th0) / th1;
        y2 = img.rows - 1;
        x2 = (y2 - th0) / th1;
    }
    line(img, Point(x1, y1), Point(x2, y2), Scalar(255, 0, 0));
    imshow("img using model 1", img);

    th0 = 1, th1 = 1;
    float dth1 = 0, dth0 = 0, costFunction = 999, learningRate = 0.0001;

    int it_nb = 0;
    while (costFunction > 0.0001) { //set threshold
        dth1 = 0, dth0 = 0;
        costFunction = 0;
        for (int i = 0; i < n; i++) {
            costFunction += (th0 + th1 * points[i].x - points[i].y) * (th0 + th1 * points[i].x - points[i].y);
            dth0 += th0 + th1 * points[i].x - points[i].y;
            dth1 += (th0 + th1 * points[i].x - points[i].y) * points[i].x;
        }
        costFunction /= 2;
        cout << "iteration :" << it_nb++ << " cost: " << costFunction << endl;

        //update rule
        th0 = th0 - learningRate * dth0;
        th1 = th1 - learningRate * dth1;
    }

    float beta, rho;
    float sumY_X = 0;

    for (int i = 0; i < n; i++) {
        sumY_X += points[i].y * points[i].y - points[i].x * points[i].x;
    }

    beta = -(atan2(2 * sumXY - (2 * sumX * sumY) / n, sumY_X + 1.0f / n * (sumX * sumX) - 1.0f / n * (sumY * sumY))) /
           2.0f;
    rho = (cos(beta) * sumX + sin(beta) * sumY) / n;

    if (fabs(beta) < M_PI / 4 || fabs(beta) > 3 * M_PI / 4) {
        x1 = 0;
        y1 = (rho - x1 * cos(beta)) / sin(beta);
        x2 = img.cols - 1;
        y2 = (rho - x2 * cos(beta)) / sin(beta);
    } else {
        y1 = 0;
        x1 = (rho - y1 * sin(beta)) / cos(beta);
        y2 = img.rows - 1;
        x2 = (rho - y2 * sin(beta)) / cos(beta);
    }
    line(img, Point(x1, y1), Point(x2, y2), Scalar(0, 0, 255));
    imshow("img using model 2", img);

    fclose(f);
    waitKey(0);
    return 0;
}