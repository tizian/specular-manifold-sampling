#include <iostream>
#include <cmath>
#include <cstdio>
#include <queue>
#include <cstdlib>
// #include <OpenEXR/OpenEXRConfig.h>
#include <ImfRgbaFile.h>
#include <ImfArray.h>
#include "../../include/mitsuba/render/flake.h"

using namespace std;
using namespace OPENEXR_IMF_NAMESPACE;

/**
 * Code for "Position-Normal Distributions for Efficient Rendering of Specular Microstructure"
 * by Ling-Qi Yan, Miloš Hašan, Steve Marschner, and Ravi Ramamoorthi.
 * ACM Transactions on Graphics (Proceedings of SIGGRAPH 2016)
 *
 * Released on:
 * https://sites.cs.ucsb.edu/~lingqi/
 *
 * With minor adapations to connect it with the Mitsuba 2 infrastructure.
 */

// X in UV coordinate: [0, 1]
// Y in normalized coordinate: [-1, 1]
// const float x1 = 0.5f;
// const float x2 = 0.5f;
// float sigma_p1;
// float sigma_p2;
// const float sigma_r1 = 0.001f;
// const float sigma_r2 = 0.001f;
// const float eta = 1.55f;

float delta = 0.5f;

// const float eps = 1e-4f;

int crop_x1, crop_x2, crop_y1, crop_y2;

Rgba *pixels;
float *ndf_sample;
Array2D<Rgba> normal_map;
int width, height;

float A_inv[16][16] = {{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                       {0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                       {-3, 3, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                       {2, -2, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                       {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
                       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
                       {0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, -2, -1, 0, 0},
                       {0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 1, 1, 0, 0},
                       {-3, 0, 3, 0, 0, 0, 0, 0, -2, 0, -1, 0, 0, 0, 0, 0},
                       {0, 0, 0, 0, -3, 0, 3, 0, 0, 0, 0, 0, -2, 0, -1, 0},
                       {9, -9, -9, 9, 6, 3, -6, -3, 6, -6, 3, -3, 4, 2, 2, 1},
                       {-6, 6, 6, -6, -3, -3, 3, 3, -4, 4, -2, 2, -2, -2, -1, -1},
                       {2, 0, -2, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0},
                       {0, 0, 0, 0, 2, 0, -2, 0, 0, 0, 0, 0, 1, 0, 1, 0},
                       {-6, 6, 6, -6, -4, -2, 4, 2, -3, 3, -3, 3, -2, -1, -2, -1},
                       {4, -4, -4, 4, 2, 2, -2, -2, 2, -2, 2, -2, 1, 1, 1, 1}};

void read_normal_map(const char *file_name, int &width, int &height) {
    RgbaInputFile file(file_name);
    Imath::Box2i dw = file.dataWindow();
    width = dw.max.x - dw.min.x + 1;
    height = dw.max.y - dw.min.y + 1;
    printf("%d %d\n", width, height);
    normal_map.resizeErase(height, width);
    file.setFrameBuffer(&normal_map[0][0] - dw.min.x - dw.min.y * width, 1, width);
    file.readPixels(dw.min.y, dw.max.y);
}

float dist(float x1, float y1, float x2, float y2) {
    return sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
}

float clamp(float x, float a, float b) {
    if (x < a)
        return a;
    else if (x > b)
        return b;
    return x;
}

inline int mod(int x, int y) {
    return ((x % y) + y) % y;
}

inline float p(int x, int y) {
    return normal_map[mod(x, width)][mod(y, height)].r;
}

inline float px(int x, int y) {
    return (normal_map[mod(x + 1, width)][mod(y, height)].r - normal_map[mod(x - 1, width)][mod(y, height)].r) / 2;
}

inline float py(int x, int y) {
    return (normal_map[mod(x, width)][mod(y + 1, height)].r - normal_map[mod(x, width)][mod(y - 1, height)].r) / 2;
}

inline float pxy(int x, int y) {
    return (p(x + 1, y + 1) - p(x + 1, y) - p(x, y + 1) + 2 * p(x, y) - p(x - 1, y) - p(x, y - 1) + p(x - 1, y - 1)) / 2;
}

inline float q(int x, int y) {
    return normal_map[mod(x, width)][mod(y, height)].g;
}

inline float qx(int x, int y) {
    return (normal_map[mod(x + 1, width)][mod(y, height)].g - normal_map[mod(x - 1, width)][mod(y, height)].g) / 2;
}

inline float qy(int x, int y) {
    return (normal_map[mod(x, width)][mod(y + 1, height)].g - normal_map[mod(x, width)][mod(y - 1, height)].g) / 2;
}

inline float qxy(int x, int y) {
    return (q(x + 1, y + 1) - q(x + 1, y) - q(x, y + 1) + 2 * q(x, y) - q(x - 1, y) - q(x, y - 1) + q(x - 1, y - 1)) / 2;
}

void computeCoeff(float *alpha, const float *x) {
    memset(alpha, 0, sizeof(float) * 16);
    for (int i = 0; i < 16; i++)
        for (int j = 0; j < 16; j++)
            alpha[i] += A_inv[i][j] * x[j];
}

Vector2f getNormal(float u, float v) {
    // Bicubic interpolation
    float x = u * width;
    float y = v * height;
    int x1 = (int) x;
    int y1 = (int) y;
    int x2 = x1 + 1;
    int y2 = y1 + 1;

    x1 = mod(x1, width);
    x2 = mod(x2, width);
    y1 = mod(y1, height);
    y2 = mod(y2, height);

    float a[16], b[16];
    float xp[16] = {p(x1, y1), p(x2, y1), p(x1, y2), p(x2, y2),
                    px(x1, y1), px(x2, y1), px(x1, y2), px(x2, y2),
                    py(x1, y1), py(x2, y1), py(x1, y2), py(x2, y2),
                    pxy(x1, y1), pxy(x2, y1), pxy(x1, y2), pxy(x2, y2)};
    float xq[16] = {q(x1, y1), q(x2, y1), q(x1, y2), q(x2, y2),
                    qx(x1, y1), qx(x2, y1), qx(x1, y2), qx(x2, y2),
                    qy(x1, y1), qy(x2, y1), qy(x1, y2), qy(x2, y2),
                    qxy(x1, y1), qxy(x2, y1), qxy(x1, y2), qxy(x2, y2)};

    computeCoeff(a, xp);
    computeCoeff(b, xq);

    float coeffA[4][4] = {{a[0], a[4], a[8], a[12]},
                          {a[1], a[5], a[9], a[13]},
                          {a[2], a[6], a[10], a[14]},
                          {a[3], a[7], a[11], a[15]}};
    float coeffB[4][4] = {{b[0], b[4], b[8], b[12]},
                          {b[1], b[5], b[9], b[13]},
                          {b[2], b[6], b[10], b[14]},
                          {b[3], b[7], b[11], b[15]}};

    float n1 = 0.0f, n2 = 0.0f;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            n1 += coeffA[i][j] * pow(x - x1, (float)(i)) * pow(y - y1, (float)(j));
            n2 += coeffB[i][j] * pow(x - x1, (float)(i)) * pow(y - y1, (float)(j));
        }


    return Vector2f(n1, n2);
}

Vector2f getNormalInt(int u1, int u2) {
    return Vector2f(p(u1, u2), q(u1, u2));
}

inline float getNx(float u, float v) {
    u /= width;
    v /= height;

    // Bicubic interpolation
    float x = u * width;
    float y = v * height;
    int x1 = (int) x;
    int y1 = (int) y;
    int x2 = x1 + 1;
    int y2 = y1 + 1;

    x1 = mod(x1, width);
    x2 = mod(x2, width);
    y1 = mod(y1, height);
    y2 = mod(y2, height);

    float a[16], b[16];
    float xp[16] = {p(x1, y1), p(x2, y1), p(x1, y2), p(x2, y2),
                    px(x1, y1), px(x2, y1), px(x1, y2), px(x2, y2),
                    py(x1, y1), py(x2, y1), py(x1, y2), py(x2, y2),
                    pxy(x1, y1), pxy(x2, y1), pxy(x1, y2), pxy(x2, y2)};
    float xq[16] = {q(x1, y1), q(x2, y1), q(x1, y2), q(x2, y2),
                    qx(x1, y1), qx(x2, y1), qx(x1, y2), qx(x2, y2),
                    qy(x1, y1), qy(x2, y1), qy(x1, y2), qy(x2, y2),
                    qxy(x1, y1), qxy(x2, y1), qxy(x1, y2), qxy(x2, y2)};

    computeCoeff(a, xp);
    computeCoeff(b, xq);

    float coeffA[4][4] = {{a[0], a[4], a[8], a[12]},
                          {a[1], a[5], a[9], a[13]},
                          {a[2], a[6], a[10], a[14]},
                          {a[3], a[7], a[11], a[15]}};
    // float coeffB[4][4] = {{b[0], b[4], b[8], b[12]},
    //                       {b[1], b[5], b[9], b[13]},
    //                       {b[2], b[6], b[10], b[14]},
    //                       {b[3], b[7], b[11], b[15]}};

    float n1 = 0.0f;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            n1 += coeffA[i][j] * pow(x - x1, (float)(i)) * pow(y - y1, (float)(j));

    return n1;
}

inline float getNy(float u, float v) {
    u /= width;
    v /= height;

    // Bicubic interpolation
    float x = u * width;
    float y = v * height;
    int x1 = (int) x;
    int y1 = (int) y;
    int x2 = x1 + 1;
    int y2 = y1 + 1;

    x1 = mod(x1, width);
    x2 = mod(x2, width);
    y1 = mod(y1, height);
    y2 = mod(y2, height);

    float a[16], b[16];
    float xp[16] = {p(x1, y1), p(x2, y1), p(x1, y2), p(x2, y2),
                    px(x1, y1), px(x2, y1), px(x1, y2), px(x2, y2),
                    py(x1, y1), py(x2, y1), py(x1, y2), py(x2, y2),
                    pxy(x1, y1), pxy(x2, y1), pxy(x1, y2), pxy(x2, y2)};
    float xq[16] = {q(x1, y1), q(x2, y1), q(x1, y2), q(x2, y2),
                    qx(x1, y1), qx(x2, y1), qx(x1, y2), qx(x2, y2),
                    qy(x1, y1), qy(x2, y1), qy(x1, y2), qy(x2, y2),
                    qxy(x1, y1), qxy(x2, y1), qxy(x1, y2), qxy(x2, y2)};

    computeCoeff(a, xp);
    computeCoeff(b, xq);

    // float coeffA[4][4] = {{a[0], a[4], a[8], a[12]},
    //                       {a[1], a[5], a[9], a[13]},
    //                       {a[2], a[6], a[10], a[14]},
    //                       {a[3], a[7], a[11], a[15]}};
    float coeffB[4][4] = {{b[0], b[4], b[8], b[12]},
                          {b[1], b[5], b[9], b[13]},
                          {b[2], b[6], b[10], b[14]},
                          {b[3], b[7], b[11], b[15]}};

    float n2 = 0.0f;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            n2 += coeffB[i][j] * pow(x - x1, (float)(i)) * pow(y - y1, (float)(j));

    return n2;
}

inline float getNxDx(float x, float y) {
    return (getNx(x + delta, y) - getNx(x - delta, y)) / (2.0f * delta);
}

inline float getNxDy(float x, float y) {
    return (getNx(x, y + delta) - getNx(x, y - delta)) / (2.0f * delta);
}

inline float getNxDxDx(float x, float y) {
    return (getNxDx(x + delta, y) - getNxDx(x - delta, y)) / (2.0f * delta);
}

inline float getNxDxDy(float x, float y) {
    return (getNxDy(x + delta, y) - getNxDy(x - delta, y)) / (2.0f * delta);
}

inline float getNxDyDx(float x, float y) {
    return (getNxDx(x, y + delta) - getNxDx(x, y - delta)) / (2.0f * delta);
}

inline float getNxDyDy(float x, float y) {
    return (getNxDy(x, y + delta) - getNxDy(x, y - delta)) / (2.0f * delta);
}

inline float getNyDx(float x, float y) {
    return (getNy(x + delta, y) - getNy(x - delta, y)) / (2.0f * delta);
}

inline float getNyDy(float x, float y) {
    return (getNy(x, y + delta) - getNy(x, y - delta)) / (2.0f * delta);
}

inline float getNyDxDx(float x, float y) {
    return (getNyDx(x + delta, y) - getNyDx(x - delta, y)) / (2.0f * delta);
}

inline float getNyDxDy(float x, float y) {
    return (getNyDy(x + delta, y) - getNyDy(x - delta, y)) / (2.0f * delta);
}

inline float getNyDyDx(float x, float y) {
    return (getNyDx(x, y + delta) - getNyDx(x, y - delta)) / (2.0f * delta);
}

inline float getNyDyDy(float x, float y) {
    return (getNyDy(x, y + delta) - getNyDy(x, y - delta)) / (2.0f * delta);
}

// Get normal
inline Vector2f getN(float x, float y) {
    return Vector2f(getNx(x, y), getNy(x, y));
}

// Get Jacobian matrix
inline Matrix2f getJ(float x, float y) {
    Matrix2f J(getNxDx(x, y), getNxDy(x, y), getNyDx(x, y), getNyDy(x, y));
    return J;
}

void sampleForFlatFlakes(const char*/* exrFilename */, const char */*flakesFilename */) {
    // vector<FlatFlake> flatFlakes;

    // read_normal_map(exrFilename, width, height);

    // for (int i = 0; i < height; i++) {
    //     for (int j = 0; j < width; j++) {
    //         Vector2f u0(i, j);
    //         Vector2f n0 = getNormal(i * 1.0f / width, j * 1.0f / height);
    //         Vector2f shape(1.0f / sqrt(12.0f), 1.0f / sqrt(12.0f));

    //         flatFlakes.push_back(FlatFlake(u0, n0, shape));
    //     }
    // }

    // FILE *fp = fopen(flakesFilename, "w");
    // fprintf(fp, "%d\n", 0);
    // fprintf(fp, "%d %d\n", width, height);
    // for (int i = 0; i < flatFlakes.size(); i++) {
    //     fprintf(fp, "%f %f %f %f %f %f\n", flatFlakes[i].u0[0], flatFlakes[i].u0[1], flatFlakes[i].n0[0], flatFlakes[i].n0[1],flatFlakes[i].shape[0], flatFlakes[i].shape[1]);
    // }
    // fclose(fp);
}

// Binary version
void sampleForLinearFlakes(const char* exrFilename, const char *flakesFilename, float sampling_rate) {
    read_normal_map(exrFilename, width, height);

    FILE *fp = fopen(flakesFilename, "wb");

    int type = 1;
    fwrite(&type, sizeof(int), 1, fp);
    fwrite(&width, sizeof(int), 1, fp);
    fwrite(&height, sizeof(int), 1, fp);

    int numFlakes = height * width * sampling_rate * sampling_rate;
    fwrite(&numFlakes, sizeof(int), 1, fp);

    delta /= sampling_rate;

    float step = 1.0f / sampling_rate;
    for (float i = 0.0f; i < width; i += step) {
        for (float j = 0.0f; j < height; j += step) {
            Vector2f u0(i, j);
            Vector2f n0 = getNormal(i * 1.0f / width, j * 1.0f / height);

            Vector2f shape(step * 1.5f / sqrt(12.0f), step * 1.5f / sqrt(12.0f));
            Matrix2f J = getJ(i, j);

            fwrite(&u0[0], sizeof(float), 1, fp);
            fwrite(&u0[1], sizeof(float), 1, fp);
            fwrite(&n0[0], sizeof(float), 1, fp);
            fwrite(&n0[1], sizeof(float), 1, fp);
            fwrite(&shape[0], sizeof(float), 1, fp);
            fwrite(&shape[1], sizeof(float), 1, fp);
            fwrite(&J(0, 0), sizeof(float), 1, fp);
            fwrite(&J(0, 1), sizeof(float), 1, fp);
            fwrite(&J(1, 0), sizeof(float), 1, fp);
            fwrite(&J(1, 1), sizeof(float), 1, fp);

            float area = step * step;
            fwrite(&area, sizeof(float), 1, fp);
        }
    }
    fclose(fp);
}

int main(int argc, char **argv) {
    if (argc != 4) {
        cout << "Usage: ./normalmap_to_flakes exr_filename flakes_filename sampling_rate" << endl;
        return 0;
    }

    char EXR_FILENAME[100];
    char FLAKES_FILENAME[100];

    strcpy(EXR_FILENAME, argv[1]);
    strcpy(FLAKES_FILENAME, argv[2]);

    // sampleForFlatFlakes(EXR_FILENAME, FLAKES_FILENAME);
    sampleForLinearFlakes(EXR_FILENAME, FLAKES_FILENAME, atof(argv[3]));

    return 0;
}
