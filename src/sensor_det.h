#ifndef SENSOR_DET_H
#define SENSOR_DET_H
 
#include <vector>
 
using std::vector;
 
// ── Structs ──────────────────────────────────────────────────────────────────
 
struct welford_results {
    float t2_score   = 0;
    float threshold  = 0;
    bool  anomaly    = false;
    bool  calibrated = false;
};
 
// ── Function Prototypes ──────────────────────────────────────────────────────
bool           preprocess_sample(vector<float>& samples);
void            invert_matrix(float S[15][15], float S_inv[15][15]);
void            welford_calibration(vector<float> samples);
welford_results detector(vector<float> samples);
 
#endif