#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include "IPFunctions.cpp"

using namespace cv;
using namespace std;

void leastMeanSquaresLineFitting() {
    FILE *f = fopen("Points/points3.txt", "r");
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

    while (costFunction > 0.0001) { //set threshold
        dth1 = 0, dth0 = 0;
        costFunction = 0;
        for (int i = 0; i < n; i++) {
            costFunction += (th0 + th1 * points[i].x - points[i].y) * (th0 + th1 * points[i].x - points[i].y);
            dth0 += th0 + th1 * points[i].x - points[i].y;
            dth1 += (th0 + th1 * points[i].x - points[i].y) * points[i].x;
        }
        costFunction /= 2;

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

    if (fabs(beta) < M_PI / 4 || fabs(beta) > 3 * M_PI / 4) {//line closer to horizontal
        x1 = 0;
        y1 = (rho - x1 * cos(beta)) / sin(beta);
        x2 = img.cols - 1;
        y2 = (rho - x2 * cos(beta)) / sin(beta);
    } else { //closer to vertical
        y1 = 0;
        x1 = (rho - y1 * sin(beta)) / cos(beta);
        y2 = img.rows - 1;
        x2 = (rho - y2 * sin(beta)) / cos(beta);
    }
    line(img, Point(x1, y1), Point(x2, y2), Scalar(0, 0, 255));
    imshow("img using model 2", img);
    fclose(f);
}

double distanceFromPointToLine(Point p, int a, int b, int c) {
    return fabs(a * p.x + b * p.y + c) / sqrt(a * a + b * b);
}

void RANSACLineFitting() {
    Mat img = imread("points_RANSAC/points3.bmp", CV_LOAD_IMAGE_GRAYSCALE);
    imshow("img initial", img);

    vector<Point> black_points;

    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            if (img.at<uchar>(i, j) == 0) {
                black_points.push_back(Point(j, i));
            }
        }
    }

    int s = 2;
    int t = 10;
    int a, b, c;
    double q = 0.8;
    double p = 0.99;
    double N = log(1 - p) / log(1 - pow(q, s));
    double T = q * black_points.size();
    int p1, p2;
    time_t timp;
    srand((unsigned) time(&timp));

    int max_consensus_size = 0, iteration = 0;
    Point maxp1, maxp2;

    while (iteration < N) {
        p1 = rand() % black_points.size();
        p2 = rand() % black_points.size();
        while (p1 == p2) {
            p2 = rand() % black_points.size();
        }

        int consensus_set_size = 0;
        a = black_points.at(p1).y - black_points.at(p2).y;
        b = black_points.at(p2).x - black_points.at(p1).x;
        c = black_points.at(p1).x * black_points.at(p2).y - black_points.at(p2).x * black_points.at(p1).y;

        for (int i = 0; i < black_points.size(); i++) {
            double distance = distanceFromPointToLine(black_points[i], a, b, c);
            if (distance <= t) {
                consensus_set_size++;
            }
        }

        if (consensus_set_size > max_consensus_size) {
            max_consensus_size = consensus_set_size;
            maxp1 = black_points.at(p1);
            maxp2 = black_points.at(p2);
        }

        if (consensus_set_size > T) {
            break;
        }
        iteration++;
    }

    line(img, maxp1, maxp2, Scalar(0, 0, 255));
    imshow("img result", img);

}

struct peak {
    int theta, ro, hval;

    bool operator<(const peak &o) const {
        return hval > o.hval;
    }
};

void pca();

void HoughTransform4LineDetection() {
    Mat_<uchar> img = imread("images_Hough/edge_simple.bmp", CV_LOAD_IMAGE_GRAYSCALE);
    int diagD = sqrt(img.cols * img.cols + img.rows * img.rows);

    Mat hough(diagD + 1, 360, CV_32SC1); //matrix with int values
    hough.setTo(0);

    Mat_<Vec3b> dst(img.rows, img.cols);

    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            dst(i, j) = Vec3b(img(i, j), img(i, j), img(i, j));
        }
    }
    float theta_deg, rho;
    int theta;
    int maxHough = 0;

    for (int i = 0; i < img.rows; i++)
        for (int j = 0; j < img.cols; j++) {
            if (img(i, j) == 255) {
                for (theta = 0; theta < 360; theta++) {
                    theta_deg = (float) (theta * M_PI) / 180;
                    rho = j * cos(theta_deg) + i * sin(theta_deg);

                    if (rho >= 0 && rho <= diagD) {

                        int current_hough = ++hough.at<int>((int) rho, (int) theta);
                        if (current_hough > maxHough) {
                            maxHough = current_hough;
                        }
                    }
                }
            }
        }

    Mat hough_acc;
    hough.convertTo(hough_acc, CV_8UC1, 255.f / maxHough);
    imshow("Hough accumulator", hough_acc);

    vector<peak> peaks;

    int n = 3;
    for (int i = n / 2; i < diagD + 1 - n / 2; i++) {
        for (int j = n / 2; j < 360 - n / 2; j++) {

            int local_maxima = hough.at<int>(i, j);
            bool isLocalMaxima = true;

            for (int l = -n / 2; l <= n / 2; l++) {
                for (int m = -n / 2; m <= n / 2; m++) {
                    if (local_maxima < hough.at<int>(i + l, j + m)) {
                        isLocalMaxima = false;
                    }
                }
            }

            if (isLocalMaxima) {
                peak peak_point;
                peak_point.hval = local_maxima;
                peak_point.ro = i;
                peak_point.theta = j;
                peaks.push_back(peak_point);
            }
        }
    }
    //sort peaks array in desc order
    sort(peaks.begin(), peaks.end());
    //draw lines
    int nb_lines = 10, x1, x2, y1, y2;
    float beta;

    for (int i = 0; i < nb_lines; i++) {
        rho = peaks.at(i).ro;
        beta = (float) (peaks.at(i).theta * M_PI) / 180.0f;

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
        line(dst, Point(x1, y1), Point(x2, y2), Scalar(0, 255, 0));
    }
    imshow("Lined image", dst);

}

Mat_<uchar> distanceTransform(Mat_<uchar> img) {
    Mat_<uchar> dt = img.clone();
    int di[9] = {-1, -1, -1, 0, 0, 0, 1, 1, 1};
    int dj[9] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
    int weight[9] = {3, 2, 3, 2, 0, 2, 3, 2, 3};

    for (int i = 1; i < img.rows; i++) {
        for (int j = 1; j < img.cols - 1; j++) {
            int min = 255;
            for (int k = 0; k <= 4; k++) {
                int pixel = dt.at<uchar>(i + di[k], j + dj[k]) + weight[k];
                if (pixel < min)
                    min = pixel;
            }
            dt.at<uchar>(i, j) = min;
        }
    }

    for (int i = img.rows - 2; i >= 0; i--) {
        for (int j = img.cols - 2; j >= 1; j--) {
            int min = 255;
            for (int k = 4; k <= 8; k++) {
                int pixel = dt.at<uchar>(i + di[k], j + dj[k]) + weight[k];
                if (pixel < min)
                    min = pixel;
            }
            dt.at<uchar>(i, j) = min;
        }
    }
    imshow("DT", dt);
    return dt;
}

void patternMatching() {
    Mat_<uchar> templateImg = imread("images_DT_PM/PatternMatching/template.bmp", IMREAD_GRAYSCALE);
    Mat_<uchar> unknownImg = imread("images_DT_PM/PatternMatching/unknown_object1.bmp", IMREAD_GRAYSCALE);

    imshow("template", templateImg);
    Mat_<uchar> dt = distanceTransform(templateImg);
    imshow("unknown image", unknownImg);

    int sum = 0;
    int nbContourPix = 0;
    for (int i = 0; i < templateImg.rows; i++) {
        for (int j = 0; j < templateImg.cols; j++) {
            if (unknownImg(i, j) == 0) {
                nbContourPix++;
                sum += dt(i, j);
            }
        }
    }
    float score = (float) sum / nbContourPix;
    cout << score << endl;
}

#define PIXEL_DIM 361
#define NB_IMAGES 400
ofstream f;

float *computeMean4EachFeature(Mat_<uchar> featureMat, float featureMean[PIXEL_DIM]) {
    f.open("mean_feature_values.txt");

    for (int i = 0; i < PIXEL_DIM; i++) {
        int sum = 0;
        for (int k = 0; k < NB_IMAGES; k++)
            sum += featureMat(k, i);

        float mean_value = (float) sum / NB_IMAGES;
        featureMean[i] = mean_value;

        (i != PIXEL_DIM - 1) ? f << mean_value << " ," : f << mean_value;
    }
    return featureMean;
}

void computeCovarianceMatrix(Mat_<uchar> featureMat, float featureMean[PIXEL_DIM],
                             float covarianceMatrix[PIXEL_DIM][PIXEL_DIM]) {
    f.open("covariance_matrix_values.txt");

    for (int i = 0; i < PIXEL_DIM; i++) {
        for (int j = 0; j < PIXEL_DIM; j++) {
            int sum = 0;
            for (int k = 0; k < NB_IMAGES; k++) {
                sum += (featureMat(k, i) - featureMean[i]) * (featureMat(k, j) - featureMean[j]);
            }

            float covariance_value = (float) sum / NB_IMAGES;
            covarianceMatrix[i][j] = covariance_value;

            (i != PIXEL_DIM - 1 && j != PIXEL_DIM - 1) ? f << covariance_value << " ," : f << covariance_value;
        }
    }
}

void computeStandardDeviations(Mat_<uchar> featureMat, float featureMean[PIXEL_DIM], float stdevs[PIXEL_DIM]) {

    for (int i = 0; i < PIXEL_DIM; i++) {
        int sum = 0;
        for (int k = 0; k < NB_IMAGES; k++) {
            float val = featureMat(k, i) - featureMean[i];
            sum += val * val;
        }
        stdevs[i] = sqrt((float) sum / NB_IMAGES);
    }
}

void computeCorrelationCoefficients(Mat_<uchar> featureMat, float featureMean[PIXEL_DIM],
                                    float covarianceMatrix[PIXEL_DIM][PIXEL_DIM],
                                    float correlationCoefficients[PIXEL_DIM][PIXEL_DIM]) {
    f.open("correlation_coefficients_matrix_values.txt");
    float stdevs[PIXEL_DIM];
    computeStandardDeviations(featureMat, featureMean, stdevs);

    for (int i = 0; i < PIXEL_DIM; i++) {
        for (int j = 0; j < PIXEL_DIM; j++) {
            if (i == j)
                correlationCoefficients[i][i] = 1;
            else {
                correlationCoefficients[i][j] = covarianceMatrix[i][j] / (stdevs[i] * stdevs[j]);
            }
            f << correlationCoefficients[i][j] << " ,";
        }
    }
}

Mat drawCorrelationChart(int x1, int y1, int x2, int y2, Mat_<uchar> featureMat,
                         float correlationCoefficients[PIXEL_DIM][PIXEL_DIM]) {

    Mat corrChart(256, 256, CV_8UC1, Scalar(255));

    int featA = x1 * 19 + y1;
    int featB = x2 * 19 + y2;

    for (int k = 0; k < NB_IMAGES; k++) {
        corrChart.at<uchar>(featureMat(k, featA), featureMat(k, featB)) = 0;
    }
    cout << correlationCoefficients[featA][featB] << endl;
    return corrChart;
}

void statisticalDataAnalysis() {

    char folder[256] = "images_faces";
    char fname[256];

    Mat_<uchar> featureMat(NB_IMAGES, PIXEL_DIM);

    for (int k = 1; k <= NB_IMAGES; k++) {
        sprintf(fname, "%s/face%05d.bmp", folder, k);
        Mat_<uchar> img = imread(fname, IMREAD_GRAYSCALE);

        for (int i = 0; i < img.rows; i++) {
            for (int j = 0; j < img.cols; j++) {
                featureMat(k - 1, i * 19 + j) = img(i, j);
            }
        }
    }
    float featureMean[PIXEL_DIM];
    computeMean4EachFeature(featureMat, featureMean);
    imshow("features", featureMat);
    float covarianceMatrix[PIXEL_DIM][PIXEL_DIM];
    computeCovarianceMatrix(featureMat, featureMean, covarianceMatrix);
    float correlationCoefficients[PIXEL_DIM][PIXEL_DIM];
    computeCorrelationCoefficients(featureMat, featureMean, covarianceMatrix, correlationCoefficients);

    Mat a = drawCorrelationChart(5, 4, 5, 14, featureMat, correlationCoefficients);
    Mat b = drawCorrelationChart(10, 3, 9, 15, featureMat, correlationCoefficients);
    Mat c = drawCorrelationChart(5, 4, 18, 0, featureMat, correlationCoefficients);

    imshow("correlation a", a);
    imshow("correlation b", b);
    imshow("correlation c", c);
    f.close();
}

void pca() {
    FILE *f = fopen("data_PCA/pca2d.txt", "r");
    int n, d;
    fscanf(f, "%d %d", &n, &d);
    Mat feat(n, d, CV_64FC1);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) {
            double x;
            fscanf(f, "%lf", &x);
            feat.at<double>(i, j) = x;
        }
    }
    double miu[7];
    for (int j = 0; j < 7; j++) {
        double sum = 0;
        for (int i = 0; i < n; i++) {
            sum += feat.at<double>(i, j);
        }
        miu[j] = (1.0 / n) * sum;
    }

    Mat X(n, d, CV_64FC1);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) {
            X.at<double>(i, j) = feat.at<double>(i, j) - miu[j];
        }
    }

    Mat C = X.t() * X / (n - 1);
    Mat Lambda, Q;
    eigen(C, Lambda, Q);
    Q = Q.t();

    for (int i = 0; i < 7; i++) {
        cout << Lambda.at<double>(i, 0) << " ";
    }

    int k = 2;
    Mat Q1k(d, k, CV_64FC1);
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < k; j++) {
            Q1k.at<double>(i, j) = Q.at<double>(i, j);
        }
    }

    Mat Xcoef = X * Q1k;

    float minimums[k];
    minimums[0] = Xcoef.at<double>(0, 0);
    minimums[1] = Xcoef.at<double>(0, 1);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            if (Xcoef.at<double>(i, j) < minimums[j]) {
                minimums[j] = Xcoef.at<double>(i, j);
            }
        }
    }

    Mat imageRes(500, 500, CV_8UC1, Scalar(255));
    for (int i = 0; i < n; i++) {
        int pointX = Xcoef.at<double>(i, 0) - minimums[0];
        int pointY = Xcoef.at<double>(i, 1) - minimums[1];
        imageRes.at<uchar>(pointX, pointY) = 0;
    }

    imshow("Image", imageRes);

    Mat Xtilda = Xcoef * Q1k.t();

    double MAD = 0;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) {
            MAD += abs(X.at<double>(i, j) - Xtilda.at<double>(i, j));
        }
    }

    MAD /= (n * d * 1.0);

    cout << endl << MAD;
}

//Lab7
#include <random>

void kmeansclustering() {
    Mat img = imread("images_Kmeans/points4.bmp", IMREAD_GRAYSCALE);
    imshow("img initial", img);

    vector<Point> points;

    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            if (img.at<uchar>(i, j) == 0) {
                points.push_back(Point(j, i));
            }
        }
    }

    int K = 3;
    int n = points.size();
    int a = 0, b = n;
    int d = n;
    vector<Point> mk; //cluster centers
    default_random_engine gen;
    uniform_int_distribution<int> dist_img(a, b); //for random nbs
    int randint;// = dist_img(gen);

    //1 pick K random points as cluster centers
    for (int k = 0; k < K; k++) {
        randint = dist_img(gen);
        mk.push_back(points.at(randint));
    }

    float dist;

    int clusters[n];
    for (int i = 0; i < n; i++) {
        clusters[i] = 0;
    }
    int sumX[3];
    int sumY[3];
    int clusterSize[3];
    for (int i = 0; i < 3; i++) {
        sumX[i] = 0;
        sumY[i] = 0;
        clusterSize[i] = 0;
    }
    bool any_changes = false;

    do {
        //asign each Xi to the closest cluster center
        for (int i = 0; i < n; i++) {
            float min = INT_MAX;
            for (int k = 0; k < K; k++) {
                dist = sqrt((points[i].x - mk[k].x) * (points[i].x - mk[k].x) +
                            (points[i].y - mk[k].y) * (points[i].y - mk[k].y));
                if (dist < min) {
                    min = dist;
                    clusters[i] = k;
                }
            }
        }
        //update cluster centers
        for (int i = 0; i < n; i++) {
            sumX[clusters[i]] += points[i].x;
            sumY[clusters[i]] += points[i].y;
            clusterSize[clusters[i]]++;
        }
        any_changes = false;
        for (int k = 0; k < K; k++) {
            int pointx = (clusterSize[k] == 0) ? 0 : (sumX[k] / clusterSize[k]);
            int pointy = (clusterSize[k] == 0) ? 0 : (sumY[k] / clusterSize[k]);
            Point point = Point(pointx, pointy);
            if (point != mk[k]) {
                any_changes = true;
                mk[k] = point;
            }
        }
    } while (any_changes);


    vector<Vec3b> colors;
    uniform_int_distribution<int> dist_img1(0, 255);
    for (int i = 0; i < K; i++) {
        int randint1 = dist_img1(gen);
        int randint2 = dist_img1(gen);
        int randint3 = dist_img1(gen);
        colors.push_back(Vec3b(randint1, randint2, randint3));
    }

    Mat clusteringImg(img.rows, img.cols, CV_8UC3, Scalar(255, 255, 255));

    for (int i = 0; i < n; i++) {
        clusteringImg.at<Vec3b>(points[i].y, points[i].x) = colors[clusters[i]];
    }

    imshow("K means clustering", clusteringImg);

    Mat voronoiImg(img.rows, img.cols, CV_8UC3, Scalar(255, 255, 255));


    for (int i = 0; i < voronoiImg.rows; i++) {
        for (int j = 0; j < voronoiImg.cols; j++) {
            int min = INT_MAX;
            for (int k = 0; k < K; k++) {
                dist = sqrt((j - mk[k].x) * (j - mk[k].x) + (i - mk[k].y) * (i - mk[k].y));
                if (dist < min) {
                    min = dist;
                    voronoiImg.at<Vec3b>(i, j) = colors[k];
                }
            }
        }
    }
    Vec3b blackPoint = Vec3b(0, 0, 0);
    for (auto &point : points) {
        voronoiImg.at<Vec3b>(point.y, point.x) = blackPoint;
    }

    imshow("Voronoi image", voronoiImg);
}

//Lab8 - kNN

const int nrclassesKNN = 6; //or C
char classesKNN[nrclassesKNN][10] =
        {"beach", "city", "desert", "forest", "landscape", "snow"};

vector<int> calculateHistogramInBins(Mat img, int nr_bins) {
    //hist = feature vector size = 3*m ; m-nr_bins
    vector<int> hist;
    int binsize = 256 / nr_bins;
    vector<int> histogramB(nr_bins);
    vector<int> histogramG(nr_bins);
    vector<int> histogramR(nr_bins);

    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            histogramB[img.at<Vec3b>(i, j)[0] / binsize]++;
            histogramG[img.at<Vec3b>(i, j)[1] / binsize]++;
            histogramR[img.at<Vec3b>(i, j)[2] / binsize]++;
        }
    }
    hist.insert(hist.end(), histogramB.begin(), histogramB.end());
    hist.insert(hist.end(), histogramG.begin(), histogramG.end());
    hist.insert(hist.end(), histogramR.begin(), histogramR.end());
    return hist;
}

pair<Mat, Mat> computeFeatureMatrixKNN(int m) {
    //m-nr_bins
    int nrinst = 672; //number of forms-images in the training set
    int feature_dim_d = 3 * m;//features
    //Allocate the feature matrix and the label vector:
    Mat X(nrinst, feature_dim_d, CV_32FC1); //feat matrix
    Mat y(nrinst, 1, CV_8UC1); //class labels/forms
    char fname[256];
    int fileNr, rowX = 0;

    for (int c = 0; c < nrclassesKNN; c++) {
        fileNr = 0;
        while (1) {
            sprintf(fname, "images_KNN/train/%s/%06d.jpeg", classesKNN[c], fileNr++);
            Mat img = imread(fname, IMREAD_COLOR);
            if (img.cols == 0) break;
            vector<int> hist = calculateHistogramInBins(img, m);
            for (int d = 0; d < feature_dim_d; d++)
                X.at<float>(rowX, d) = hist[d];
            y.at<uchar>(rowX) = c;
            rowX++;
        }
    }
    return pair<Mat, Mat>(X, y);
}

bool compareDistances(const pair<float, int> &dist1, const pair<float, int> &dist2) {
    return (dist1.first < dist2.first);
}

int KNearestNeighborsTrain(Mat img, int m, int K, Mat X, Mat y) {
    int nrinst = 672;
    int feature_dim_d = 3 * m;//features
    vector<int> hist = calculateHistogramInBins(img, m);
    vector<pair<float, int>> distances;
    int closestClass = INT_MIN, maxVote = INT_MIN;
    vector<int> votes(K);

    for (int i = 0; i < nrinst; i++) {
        float distance = 0;
        for (int k = 0; k < feature_dim_d; k++) {
            distance += sqrt(pow(hist[k] - X.at<float>(i, k), 2));
        }
        distances.push_back(pair<float, int>(distance, y.at<uchar>(i)));
    }
    sort(distances.begin(), distances.end(), compareDistances);

    for (int i = 0; i < K; i++) {
        votes[distances[i].second]++;
    }

    for (int c = 0; c < K; c++) {
        if (votes[c] > maxVote) {
            maxVote = votes[c];
            closestClass = c;
        }
    }
    return closestClass;
}

void testKNN(int K, int m) {
    pair<Mat, Mat> Xy = computeFeatureMatrixKNN(m);

    Mat img = imread("images_KNN/test/beach/000002.jpeg", CV_LOAD_IMAGE_COLOR);
    int prediction = KNearestNeighborsTrain(img, m, K, Xy.first, Xy.second);
    cout << classesKNN[prediction];
}

void printMat(int rows, int cols, Mat mat) {

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            cout << mat.at<float>(i, j) << " ";
        }
        cout << endl;
    }
}

void KNearestNeighborsTest(int K, int m) {
    char fname[256];
    int fileNr, closestClass;
    pair<Mat, Mat> Xy = computeFeatureMatrixKNN(m);
    Mat C = Mat::zeros(nrclassesKNN, nrclassesKNN, CV_32FC1);
    int correct = 0, wrong = 0;

    for (int c = 0; c < nrclassesKNN; c++) {
        fileNr = 0;
        while (1) {
            sprintf(fname, "images_KNN/train/%s/%06d.jpeg", classesKNN[c], fileNr++);
            Mat img = imread(fname, IMREAD_COLOR);
            if (img.cols == 0) break;
            closestClass = KNearestNeighborsTrain(img, m, K, Xy.first, Xy.second);
            C.at<float>(c, closestClass)++;
            (c == closestClass) ? correct++ : wrong++;
        }
    }
    cout << "Confusion Matrix:" << endl;
    printMat(C.rows, C.cols, C);
    cout << "Accuracy of class prediction: " << (float) correct / (correct + wrong) << endl;
}
//Lab 9

Mat_<uchar> binarizeImage(Mat_<uchar> img) {
    Mat_<uchar> res(img.rows, img.cols);

    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            res(i, j) = img(i, j) > 128 ? 255 : 0;
        }
    }
    return res;
}

void train(Mat likelihoodsMatrix, Mat priorsMatrix, int C) {
    char fname[256];
    int index = 0, d = 28 * 28, rows = 0, instances_total = 0;
    Mat_<uchar> X(1000, d); //feature matrix
    Mat_<uchar> Y(1000, 1); //class labels
    vector<int> no_instances;

    for (int i = 0; i < C; i++) {
        no_instances.push_back(0);
    }

    for (int c = 0; c < C; c++) {
        index = 0;
        while (index < 100) {
            sprintf(fname, "images_Bayes/train/%d/%06d.png", c, index);
            Mat_<uchar> img = imread(fname, 0);
            if (img.cols == 0) break;
            //process img
            index++;

            Mat_<uchar> binarized = binarizeImage(img);

            d = 0;
            for (int i = 0; i < binarized.rows; i++) {
                for (int j = 0; j < binarized.cols; j++) {
                    X(rows, d) = binarized(i, j);
                    d++;
                }
            }
            //d = 784
            rows++;
            Y(rows, 0) = c;
            no_instances.at(c)++;
            instances_total++;
        }
    }
    cout << rows << endl;
    for (int c = 0; c < C; c++) {
        priorsMatrix.at<double>(c, 0) = (double) no_instances.at(c) / instances_total;
    }

    for (int c = 0; c < C; c++) {//for each class
        int cnt_class = 0;
        for (int k = 0; k < d; k++) {//for each feature
            int cnt = 0;
            for (int n = 0; n < rows; n++) {//for each instance
                if (Y(n, 0) == c) {
                    if (X(n, k) == 255) {
                        cnt++;
                    }
                    cnt_class++;
                }
            }
            likelihoodsMatrix.at<double>(k, c) = (double) ((double) cnt + 1) / ((double) cnt_class + C);
        }
    }
}

int classifyBayes(Mat img, Mat priors, Mat likelihood, int C) {
    Mat log_posterior_class(C, 1, CV_64FC1);
    double ll, log_posterior, sum;
    for (int c = 0; c < C; c++) {
        sum = 0;

        for (int i = 0; i < img.rows; i++) {
            for (int j = 0; j < img.cols; j++) {
                int index_of_feat = i * 28 + j;
                if (img.at<uchar>(i, j) == 255) {
                    ll = log(likelihood.at<double>(index_of_feat, c));
                } else {
                    ll = log(1 - likelihood.at<double>(index_of_feat, c));
                }
                sum += ll;
            }
        }
        log_posterior = log(priors.at<double>(c, 0)) + sum;
        log_posterior_class.at<double>(c, 0) = log_posterior;
    }

    float max_log_posterior = INT_MIN;
    int predicted_class = 0;

    for (int c = 0; c < C; c++) {
        cout << "Log posterior " << c << ": " << log_posterior_class.at<double>(c, 0) << endl;
        if (log_posterior_class.at<double>(c, 0) > max_log_posterior) {
            max_log_posterior = log_posterior_class.at<double>(c, 0);
            predicted_class = c;
        }
    }

    return predicted_class;
}

void naiveBayesClassifier() {
    int C = 10, d = 28 * 28;
    Mat priors(C, 1, CV_64FC1);
    Mat likelihood(d, C, CV_64FC1);

    train(likelihood, priors, C);

    Mat_<uchar> img = imread("images_Bayes/test/7/000007.png", 0);

    Mat_<uchar> binarized = binarizeImage(img);
    int predicted_class = classifyBayes(binarized, priors, likelihood, C);
    cout << "Predicted class: " << predicted_class << endl;
}

//Lab 10 Linear Classification

vector<float>
online_perceptron(vector<float> w, float lr, float E_limit, int max_iter, Mat_<int> X, Mat_<int> Y, int n, int d) {

    for (int iter = 0; iter < max_iter; iter++) {
        float E = 0;
        for (int i = 0; i < n; i++) {
            float z = 0;
            for (int j = 0; j < d; j++) {
                z += w[j] * X(i, j);
            }
            if (z * Y(i, 0) < 0) {
                for (int j = 0; j < d; j++) {
                    w[j] = w[j] + lr * X(i, j) * Y(i, 0);
                }
                E += 1;
            }
        }

        E /= n;
        if (E < E_limit)
            break;
    }
    return w;
}

void linearClassification() {
    Mat_<Vec3b> img = imread("images_Perceptron/test06.bmp", CV_LOAD_IMAGE_COLOR);
    Mat_<int> X(10000, 3);
    Mat_<int> Y(10000, 1);

    int n = 0;
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {

            if (img(i, j)[0] == 0 && img(i, j)[1] == 0 && img(i, j)[2] == 255) {
                X(n, 0) = 1;
                X(n, 1) = j;
                X(n, 2) = i;
                Y(n++, 0) = 1;
            } else if (img(i, j)[0] == 255 && img(i, j)[1] == 0 && img(i, j)[2] == 0) {
                X(n, 0) = 1;
                X(n, 1) = j;
                X(n, 2) = i;
                Y(n++, 0) = -1; // blue
            }

        }
    }

    vector<float> weights;
    weights.push_back(0.1);
    weights.push_back(0.1);
    weights.push_back(0.1);

    float lr = pow(10, -4);
    int max_iter = pow(10, 5);
    float E_limit = pow(10, -5);

    int d = 3;
    weights = online_perceptron(weights, lr, E_limit, max_iter, X, Y, n, d);

    Mat img_final = img;

    float x1 = 0;
    float x2 = img.rows - 1;
    float y1 = -(weights[1] * x1 + weights[0]) / weights[2];
    float y2 = -(weights[1] * x2 + weights[0]) / weights[2];

    line(img_final, Point(x1, y1), Point(x2, y2), Scalar(0, 255, 0));

    imshow("Linear_Classifier", img_final);

}
//Lab 11

//j care weak learner aplic pe trasatura
//in loc de img.size punem widthu pt j x1
// cand j repr trasat x2 punem weightu
//zi arata rezultatu clasificarii pt a ia forma utilizand decision stump
//daca zi coincide cu yi inseamna ca e clasificata corect
//in e eroarea adun ponderi suma = 1
//(w + h ) * 2
//ponderile formelor clasificate corect scad
//X-forma cu 2 trasaturi (din labwork)

struct weaklearner {
    int feature_i;
    int threshold;
    int class_label;
    float error;

    int classify(Mat X) {
        if (X.at<int>(feature_i) < threshold)
            return class_label;
        else
            return -class_label;
    }
};

struct classifier {
    int T;
    vector<float> alphas;
    vector<weaklearner> ht;

    int classify(Mat X) {
        float sum = 0;
        for (int i = 0; i < T; i++) {
            sum += alphas[i] * ht[i].classify(X);
        }

        if (sum < 0) {
            return -1;
        } else {
            return 1;
        }
    }
};

weaklearner findWeakLearner(Mat_<Vec3b> img, Mat_<int> X, Mat_<int> y, Mat_<float> w) {
    weaklearner best_h = {};
    float best_err = INFINITY;
    float e;
    int z = 0;
    for (int j = 0; j < X.cols; j++) {
        int size = (j == 0) ? img.cols : img.rows;
        for (int threshold = 0; threshold < size; threshold++) {
            vector<int> class_label{-1, 1};
            for (int &label_index : class_label) {
                e = 0;
                for (int i = 0; i < X.rows; i++) {
                    (X(i, j) < threshold) ? z = label_index : z = -label_index;
                    if (z * y(i, 0) < 0)
                        e += w(i, 0);
                }
                if (e < best_err) {
                    best_err = e;
                    best_h = {j, threshold, label_index, e};
                }
            }
        }
    }
    return best_h;
}

void adaboost(int T, Mat_<Vec3b> img, Mat_<int> X, Mat_<int> y) {
    int n = y.rows;
    Mat_<float> w(n, 1);

    for (int i = 0; i < n; i++)
        w(i, 0) = (float) 1 / n;


    classifier clf;
    clf.T = T;
    for (int t = 0; t < T; t++) {
        weaklearner weakLearner = findWeakLearner(img, X, y, w);
        clf.alphas.push_back(0.5 * log((float) (1 - weakLearner.error) / weakLearner.error));
        clf.ht.push_back(weakLearner);

        float s = 0;
        for (int i = 0; i < n; i++) {
            w(i, 0) = w(i, 0) * exp(-clf.alphas[t] * y(i, 0) * weakLearner.classify(X.row(i)));
            s += w(i, 0);
        }

        for (int i = 0; i < n; i++) { //weight normalization
            w(i, 0) = w(i, 0) / s;
        }
    }

    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            if (img(i, j) != Vec3b(255, 0, 0) && img(i, j) != Vec3b(0, 0, 255)) {
                Mat_<int> row(1, 2);
                row(0, 0) = i;
                row(0, 1) = j;

                (clf.classify(row) > 0) ? img(i, j) = Vec3b(0, 255, 255) : img(i, j) = Vec3b(255, 255, 0);
            }
        }
    }
    imshow("Adaboosted image", img);
    waitKey();
}

void adaboostMethod() {
    Mat_<Vec3b> img;
    Mat_<float> w;
    classifier classifier;

    img = imread("images_AdaBoost/points0.bmp", IMREAD_COLOR);

    vector<Point2i> pointsVector;

    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            if (img(i, j) != Vec3b(255, 255, 255)) {
                pointsVector.push_back(Point2i(i, j));
            }
        }
    }

    Mat_<int> X(pointsVector.size(), 2); //feature matrix
    Mat_<int> Y(pointsVector.size(), 1); //class labels

    for (int i = 0; i < pointsVector.size(); i++) {
        int x = pointsVector.at(i).x;
        int y = pointsVector.at(i).y;
        X(i, 0) = x;
        X(i, 1) = y;

        (img(x, y) == Vec3b(255, 0, 0)) ? Y(i, 0) = -1 : Y(i, 0) = 1; //if blue
    }

    adaboost(13, img, X, Y);
    waitKey(0);

}

int main() {
//Lab1
//    leastMeanSquaresLineFitting();
//Lab2
//    RANSACLineFitting();
//Lab3
//    HoughTransform4LineDetection();
//Lab4
//    patternMatching();
//Lab 5
//    statisticalDataAnalysis();
//Lab6
//    pca();
//Lab 7
    kmeansclustering();

//Lab 8
    testKNN(5, 8);
    cout << endl;
    KNearestNeighborsTest(5, 8);
    waitKey();
//Lab 9
//    naiveBayesClassifier();
//Lab 10
//    linearClassification();
//Lab 11
//    adaboostMethod();
    return 0;
}

