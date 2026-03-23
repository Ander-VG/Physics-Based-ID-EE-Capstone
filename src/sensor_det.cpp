#include "sensor_det.h"
#include <cmath>

#include <string>
#include <cmath>
#include <iomanip>
#include <string>
#include <vector>

using std::string;
using std::vector;
using std::endl;
using std::fstream;
using std::istringstream;
using std::getline;

static int n = 0;
static float mu[15] = {0};
static float C[15][15] = {0};
static float S_inv[15][15] = {0};
static const int N_cal = 200;
static const int N_cap = 1000;
static const int P = 15;
static const float F_crit = 24.5; 

//Invert Matrix function for later
void invert_matrix(float S[15][15], float S_inv[15][15]) {

    float aug[15][30];
    for (int i = 0; i < P; i++){
        for(int j = 0; j < P; j++){
            aug[i][j] = S[i][j];
            aug[i][j + P] = (i == j) ? 1.0f : 0.0f;
        }
    }
    for (int col = 0; col < P; col++) {
        int max_row = col;
        float max_val = fabsf(aug[col][col]);
        for (int row = col + 1; row < P; row++){
            if (fabsf(aug[row][col]) > max_val) {
                max_val = fabsf(aug[row][col]);
                max_row = row;
            }
        }
        
        //swap rows
        if (max_row != col) {
            for (int j =0; j < 2 * P; j++) {
                float tmp = aug[col][j];
                aug[col][j] = aug[max_row][j];
                aug[max_row][j] = tmp;
            }
        }

        //scale pivot row so pivot becomes 1
        float pivot = aug[col][col];
        for (int j = 0; j < 2 * P; j++){
            aug[col][j] /= pivot;
        }

        // eliminate all other rows
        for (int row = 0; row < P; row++) {
            if (row == col) continue;
            float factor = aug[row][col];
            for (int j = 0; j < 2*P; j++){
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }
    // extract inverse
    for (int i = 0; i < P; i++) {
        for (int j = 0; j < P; j++) {
            S_inv[i][j] = aug[i][j+P];
        }
    }
}


//Welford Calibration to prepare for detection
void welford_calibration(vector<float> samples) {
    if (n < N_cap) {
        n++;
    }
    else {
        float scale = static_cast<float>(N_cap-1) / (N_cap);
        for (int i = 0; i < P; i++) {
            for (int j = 0; j < P; j++) {
                C[i][j] *= scale;
            }
        }
    }
    float delta[15], delta2[15];

    for (int i = 0; i < P; i++) {
        delta[i] = samples[i] - mu[i];
    }
    for (int i = 0; i < P; i++) {
        mu[i] = mu[i] + delta[i]/n;
    }
    for (int i = 0; i < P; i++) {
        delta2[i] = samples[i] - mu[i];
    }
    for (int i = 0; i < P; i++) {
        for (int j = 0; j < P; j++) {
            C[i][j] += delta[i] * delta2[j];
        }
    }
}

//Welford Detector that uses Hotelling T^2
welford_results detector(vector<float> samples) {
    welford_results result;
    result.calibrated = (n >= N_cal);
    result.t2_score = 0;
    result.threshold = 0;
    result.anomaly = false;

    if (!result.calibrated) {
        return result;
    }

    // Making S from C
    float S[15][15];
    for (int i = 0; i < P; i++) {
        for (int j =0; j < P; j++) {
            S[i][j] = C[i][j] / (n-1);
        }
    }

    // Regularize
    for (int i = 0; i < P; i++) {
        S[i][i] += 1e-6f;
    }

    // Invert S
    invert_matrix(S, S_inv);

    float d[15];
    for (int i = 0; i < P; i++) {
        d[i] = samples[i] - mu[i];
    }
    float t2 = 0;
    for (int i = 0; i < P; i++) {
        float row_sum = 0;
        for (int j = 0; j < P; j++) {
            row_sum += S_inv[i][j] * d[j];
        }
        t2 += d[i] * row_sum;
    }

    float threshold = (float)(P * (N_cal - 1)) / (N_cal - P) * F_crit;

    result.t2_score = t2;
    result.threshold = threshold;
    result.anomaly = (t2 > threshold);

    return result;
}