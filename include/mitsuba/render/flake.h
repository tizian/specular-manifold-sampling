#pragma once

#include <iostream>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <set>
#include <string>
#include <mitsuba/render/fwd.h>
#include <mitsuba/core/fwd.h>
#include <mitsuba/core/math.h>
#include <mitsuba/core/vector.h>
#include <mitsuba/core/transform.h>

using namespace mitsuba;

NAMESPACE_BEGIN(mitsuba)

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

using Float = float;
using Vector2f = Vector<Float, 2>;
using Vector2i = Vector<int, 2>;
using Vector4f = Vector<Float, 4>;
using Matrix2f = Matrix<Float, 2>;

#define FLAKE_SHAPE_SIGMAS 3.0f
#define FLAKE_NORMAL_SIGMAS 4.0f
#define FLAKE_PIXEL_SIGMAS 3.0f

inline float g(const float x, const float sigma) {
    return exp(-0.5f * x * x / (sigma * sigma)) / (sigma * math::SqrtTwoPi<float>);
}

inline float g(const float x, const float mu, const float sigma) {
    return g(x - mu, sigma);
}

class Flake {
public:
    Vector2f u0;
    Vector2f n0;
    Vector2f shape;         // Standard deviation.
    Vector4f aa, bb;        // Bounding box.
    float area;             // Actual area it covers.
    int index;              // For internal use.
public:
    Flake() {}
    Flake(Vector2f u0, Vector2f n0, Vector2f shape, float area): u0(u0), n0(n0), shape(shape), area(area) {}
    virtual ~Flake() {}

    virtual Vector2f getNormal(const Vector2f &u) const = 0;
    virtual void getBoundingBox(float intrinsicRoughness, Vector4f &aa, Vector4f &bb) const = 0;
    virtual float contributionToNdf(Vector2f uQuery, Vector2f sigma_p, Vector2f nQuery, float sigma_r) const = 0;
};


class FlatFlake: public Flake {
public:
    FlatFlake() {}
    FlatFlake(Vector2f u0, Vector2f n0, Vector2f shape, float area): Flake(u0, n0, shape, area) {}
    ~FlatFlake() {}

    virtual Vector2f getNormal(const Vector2f &/* u */) const {
        return n0;
    }

    virtual void getBoundingBox(float intrinsicRoughness, Vector4f &aa, Vector4f &bb) const {
        aa = Vector4f(u0[0] - FLAKE_SHAPE_SIGMAS * shape[0], u0[1] - FLAKE_SHAPE_SIGMAS * shape[1], n0[0] - FLAKE_NORMAL_SIGMAS * intrinsicRoughness, n0[1] - FLAKE_NORMAL_SIGMAS * intrinsicRoughness);
        bb = Vector4f(u0[0] + FLAKE_SHAPE_SIGMAS * shape[0], u0[1] + FLAKE_SHAPE_SIGMAS * shape[1], n0[0] + FLAKE_NORMAL_SIGMAS * intrinsicRoughness, n0[1] + FLAKE_NORMAL_SIGMAS * intrinsicRoughness);
    }

    virtual float contributionToNdf(Vector2f uQuery, Vector2f sigma_p, Vector2f nQuery, float sigma_r) const {
        float gy = g(n0[0], nQuery[0], sigma_r) * g(n0[1], nQuery[1], sigma_r);
        float gx1 = g(u0[0], uQuery[0], sqrt(sigma_p[0] * sigma_p[0] + shape[0] * shape[0]));
        float gx2 = g(u0[1], uQuery[1], sqrt(sigma_p[1] * sigma_p[1] + shape[1] * shape[1]));
        return area * gy * gx1 * gx2;
    }
};


class LinearFlake: public Flake {
public:
    Matrix2f J;
private:
    inline double intExp(double a, double b, double c, double d, double e, double f) const {
        double t1 = 4.0 * a * c - b * b;
        if (t1 <= 0.0) return 0.0;
        double t2 = f + (b * d * e - c * d * d - a * e * e) / t1;
        return 2 * math::Pi<double> / sqrt(t1) * exp(t2);
    }

    inline float intExp(float *coeffs) const {
        return intExp(double(coeffs[0]),
                      double(coeffs[1]),
                      double(coeffs[2]),
                      double(coeffs[3]),
                      double(coeffs[4]),
                      double(coeffs[5]));
    }

    inline float getC(float sigma) const {
        return 1.0f / (math::SqrtTwoPi<float> * sigma);
    }

    inline void addCoeffs(float k1, float k2, float b, float sigma, float *coeffs) const {
        float denominatorInv = 1.0f / (-2.0f * sigma * sigma);
        coeffs[0] += k1 * k1 * denominatorInv;
        coeffs[1] += 2.0f * k1 * k2 * denominatorInv;
        coeffs[2] += k2 * k2 * denominatorInv;
        coeffs[3] += 2.0f * b * k1 * denominatorInv;
        coeffs[4] += 2.0f * b * k2 * denominatorInv;
        coeffs[5] += b * b * denominatorInv;
    }

public:
    LinearFlake() {}
    LinearFlake(Vector2f u0, Vector2f n0, Vector2f shape, float area, Matrix2f J): Flake(u0, n0, shape, area), J(J) {}
    ~LinearFlake() {}

    virtual Vector2f getNormal(const Vector2f &u) const {
        float nx = n0[0] + J(0, 0) * (u[0] - u0[0]) + J(0, 1) * (u[1] - u0[1]);
        float ny = n0[1] + J(1, 0) * (u[0] - u0[0]) + J(1, 1) * (u[1] - u0[1]);
        return Vector2f(nx, ny);
    }

    virtual void getBoundingBox(float intrinsicRoughness, Vector4f &aa, Vector4f &bb) const {
        float nxMin = n0[0] - fabsf(J(0, 0)) * FLAKE_SHAPE_SIGMAS * shape[0] - fabsf(J(0, 1)) * FLAKE_SHAPE_SIGMAS * shape[1];
        float nxMax = n0[0] + fabsf(J(0, 0)) * FLAKE_SHAPE_SIGMAS * shape[0] + fabsf(J(0, 1)) * FLAKE_SHAPE_SIGMAS * shape[1];
        float nyMin = n0[1] - fabsf(J(1, 0)) * FLAKE_SHAPE_SIGMAS * shape[0] - fabsf(J(1, 1)) * FLAKE_SHAPE_SIGMAS * shape[1];
        float nyMax = n0[1] + fabsf(J(1, 0)) * FLAKE_SHAPE_SIGMAS * shape[0] + fabsf(J(1, 1)) * FLAKE_SHAPE_SIGMAS * shape[1];

        aa = Vector4f(u0[0] - FLAKE_SHAPE_SIGMAS * shape[0], u0[1] - FLAKE_SHAPE_SIGMAS * shape[1], nxMin - FLAKE_NORMAL_SIGMAS * intrinsicRoughness, nyMin - FLAKE_NORMAL_SIGMAS * intrinsicRoughness);
        bb = Vector4f(u0[0] + FLAKE_SHAPE_SIGMAS * shape[0], u0[1] + FLAKE_SHAPE_SIGMAS * shape[1], nxMax + FLAKE_NORMAL_SIGMAS * intrinsicRoughness, nyMax + FLAKE_NORMAL_SIGMAS * intrinsicRoughness);
    }

    virtual float contributionToNdf(Vector2f uQuery, Vector2f sigma_p, Vector2f nQuery, float sigma_r) const {
        float c = area;
        c *= getC(shape[0]);
        c *= getC(shape[1]);
        c *= getC(sigma_p[0]);
        c *= getC(sigma_p[1]);
        c *= getC(sigma_r);
        c *= getC(sigma_r);

        float coeffs[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        addCoeffs(1.0f, 0.0f, 0.0f, shape[0], coeffs);
        addCoeffs(0.0f, 1.0f, 0.0f, shape[1], coeffs);
        addCoeffs(1.0f, 0.0f, u0[0] - uQuery[0], sigma_p[0], coeffs);
        addCoeffs(0.0f, 1.0f, u0[1] - uQuery[1], sigma_p[1], coeffs);
        addCoeffs(J(0, 0), J(0, 1), n0[0] - nQuery[0], sigma_r, coeffs);
        addCoeffs(J(1, 0), J(1, 1), n0[1] - nQuery[1], sigma_r, coeffs);

        return c * intExp(coeffs);
    }
};

#define GRID_RES 32
#define SAMPLING_RATE 15

class FlakesTree {
private:
    struct Node {
        std::vector<Flake*> flakes;
        Node *left, *right;
        Vector4f aa, bb;
    };

    Vector4f min4(const Vector4f &a, const Vector4f &b) {
        return Vector4f(std::min(a[0], b[0]), std::min(a[1], b[1]), std::min(a[2], b[2]), std::min(a[3], b[3]));
    }

    Vector4f max4(const Vector4f &a, const Vector4f &b) {
        return Vector4f(std::max(a[0], b[0]), std::max(a[1], b[1]), std::max(a[2], b[2]), std::max(a[3], b[3]));
    }

    bool intersectAABB(const Vector4f &aa1, const Vector4f &bb1, const Vector4f &aa2, const Vector4f &bb2) const {
        if ((aa1[0] > bb2[0] || bb1[0] < aa2[0]) ||
            (aa1[1] > bb2[1] || bb1[1] < aa2[1]) ||
            (aa1[2] > bb2[2] || bb1[2] < aa2[2]) ||
            (aa1[3] > bb2[3] || bb1[3] < aa2[3]))
            return false;
        return true;
    }

    void queryFlakes(Node *node, const Vector4f &aa, const Vector4f &bb, std::vector<Flake*> &flakes) const {
        if (node == NULL)
            return;

        if (!intersectAABB(node->aa, node->bb, aa, bb))
            return;

        if (node->left == NULL && node->right == NULL) {
            for (size_t i = 0; i < node->flakes.size(); i++) {
                if (!intersectAABB(node->flakes[i]->aa, node->flakes[i]->bb, aa, bb))
                    continue;
                flakes.push_back(node->flakes[i]);
            }
            return;
        }

        queryFlakes(node->left, aa, bb, flakes);
        queryFlakes(node->right, aa, bb, flakes);
    }

    void buildSTGrid(const char *gridFilename) {
        FILE *fp = fopen(gridFilename, "rb");
        if (!fp) {
            printf("Initializing grid...");
            std::set<int> setGrid[GRID_RES][GRID_RES];
            #pragma omp parallel for schedule(dynamic)
            for (size_t k = 0; k < m_flakes.size(); k++) {
                if (k % 1000 == 0)
                    printf("Processing %dK th flake!\n", int(k / 1000));
                Vector2i sRate(SAMPLING_RATE, SAMPLING_RATE);
                int pPrev = -1, qPrev = -1;
                for (int i = 0; i < sRate[0]; i++) {
                    for (int j = 0; j < sRate[1]; j++) {
                        float s = i / (sRate[0] - 1.0f) * 2.0f - 1.0f;
                        float t = j / (sRate[1] - 1.0f) * 2.0f - 1.0f;
                        if (s * s + t * t > 1.0f)
                            continue;
                        Vector2f du(s * FLAKE_SHAPE_SIGMAS * m_flakes[k]->shape[0], t * FLAKE_SHAPE_SIGMAS * m_flakes[k]->shape[1]);
                        Vector2f n = m_flakes[k]->getNormal(m_flakes[k]->u0 + du);

                        int r = 0;
                        if (i == 0 || j == 0 || i == sRate[0] - 1 || j == sRate[1] - 1)
                            r = 1;

                        for (int ii = -r; ii <= r; ii++) {
                            for (int jj = -r; jj <= r; jj++) {
                                float n0 = n[0] + ii * FLAKE_NORMAL_SIGMAS * m_intrinsicRoughness;
                                float n1 = n[1] + jj * FLAKE_NORMAL_SIGMAS * m_intrinsicRoughness;

                                int p = (n0 + 1.0f) / 2.0f * GRID_RES;
                                int q = (n1 + 1.0f) / 2.0f * GRID_RES;

                                if (p < 0 || p >= GRID_RES || q < 0 || q >= GRID_RES)
                                    continue;

                                if (p == pPrev && q == qPrev)
                                    continue;
                                pPrev = p;
                                qPrev = q;

                                #pragma omp critical
                                {
                                    setGrid[p][q].insert(k);
                                }
                            }
                        }
                    }
                }
            }
            printf("OK!\n");

            #pragma omp parallel for schedule(dynamic)
            for (int t = 0; t < GRID_RES * GRID_RES; t++) {
                int i = t / GRID_RES;
                int j = t % GRID_RES;
                stGrid[i][j] = new Node;

                for (std::set<int>::iterator it = setGrid[i][j].begin(); it != setGrid[i][j].end(); it++)
                    stGrid[i][j]->flakes.push_back(m_flakes[*it]);
            }

            printf("Writing grid data...");
            fp = fopen(gridFilename, "wb");
            for (int i = 0; i < GRID_RES; i++) {
                for (int j = 0; j < GRID_RES; j++) {
                    int size = stGrid[i][j]->flakes.size();
                    fwrite(&size, sizeof(int), 1, fp);
                    for (int k = 0; k < size; k++) {
                        int index = stGrid[i][j]->flakes[k]->index;
                        fwrite(&index, sizeof(int), 1, fp);
                    }
                }
            }
            fclose(fp);
            printf("OK!\n");

        } else {
            // Read grid data from file
            printf("Reading grid data...");
            for (int i = 0; i < GRID_RES; i++) {
                for (int j = 0; j < GRID_RES; j++) {
                    stGrid[i][j] = new Node;
                    int size;
                    fread(&size, sizeof(int), 1, fp);
                    for (int k = 0; k < size; k++) {
                        int index;
                        fread(&index, sizeof(int), 1, fp);
                        stGrid[i][j]->flakes.push_back(m_flakes[index]);
                    }
                }
            }
            fclose(fp);
            printf("OK!\n");
        }

        int totalSize = 0;
        #pragma omp parallel for schedule(dynamic)
        for (int t = 0; t < GRID_RES * GRID_RES; t++) {
            int i = t / GRID_RES;
            int j = t % GRID_RES;
            printf("Building hierarchy on st grid: (%d, %d)\n", i, j);
            if (stGrid[i][j]->flakes.size() != 0) {
                buildFlakesTree(stGrid[i][j], m_intrinsicRoughness, 0);
                #pragma omp atomic
                totalSize += stGrid[i][j]->flakes.size();
            } else {
                delete stGrid[i][j];
                stGrid[i][j] = NULL;
            }
        }

        printf("Original size: %lu, Hierarchy size: %d\n", m_flakes.size(), totalSize);
        printf("Flake size: %lu\n", sizeof(Flake));
    }

    void buildUVGrid() {
        for (int i = 0; i < GRID_RES; i++)
            for (int j = 0; j < GRID_RES; j++)
                uvGrid[i][j] = new Node;

        #pragma omp parallel for schedule(dynamic)
        for (size_t k = 0; k < m_flakes.size(); k++) {
            if (k % 1000 == 0)
                printf("Processing %dK th flake!\n", int(k / 1000));

            int p1 = max(0, (int) (m_flakes[k]->aa[0] / m_resolutionU * GRID_RES));
            int p2 = min(GRID_RES - 1, (int) (m_flakes[k]->bb[0] / m_resolutionU * GRID_RES));
            int q1 = max(0, (int) (m_flakes[k]->aa[1] / m_resolutionV * GRID_RES));
            int q2 = min(GRID_RES - 1, (int) (m_flakes[k]->bb[1] / m_resolutionV * GRID_RES));

            for (int p = p1; p <= p2; p++) {
                for (int q = q1; q <= q2; q++) {
                    #pragma omp critical
                    {
                        uvGrid[p][q]->flakes.push_back(m_flakes[k]);
                    }
                }
            }
        }

        int totalSize = 0;
        #pragma omp parallel for schedule(dynamic)
        for (int t = 0; t < GRID_RES * GRID_RES; t++) {
            int i = t / GRID_RES;
            int j = t % GRID_RES;
            printf("Building hierarchy on uv grid: (%d, %d)\n", i, j);

            if (uvGrid[i][j]->flakes.size() != 0) {
                buildFlakesTree(uvGrid[i][j], m_intrinsicRoughness, 0);
                #pragma omp atomic
                totalSize += uvGrid[i][j]->flakes.size();
            } else {
                delete uvGrid[i][j];
                uvGrid[i][j] = NULL;
            }
        }
        printf("Original size: %lu, Hierarchy size: %d\n", m_flakes.size(), totalSize);
        printf("Flake size: %lu\n", sizeof(Flake));
    }

    void buildFlakesTree(Node *root, int currentLevel, int badTimes = 0) {
        root->left = NULL;
        root->right = NULL;

        root->aa = Vector4f(1e10f, 1e10f, 1e10f, 1e10f);
        root->bb = Vector4f(-1e10f, -1e10f, -1e10f, -1e10f);
        for (size_t i = 0; i < root->flakes.size(); i++) {
            root->aa = min4(root->aa, root->flakes[i]->aa);
            root->bb = max4(root->bb, root->flakes[i]->bb);
        }

        if (root->flakes.size() <= 5)
            return;

        Node *left = new Node, *right = new Node;

        float median;
        switch (currentLevel % 2) {
        case 0:
            // Split along u0[0]
            median = (root->aa[0] + root->bb[0]) / 2.0f;
            for (size_t i = 0; i < root->flakes.size(); i++) {
                if (root->flakes[i]->u0[0] < median)
                    left->flakes.push_back(root->flakes[i]);
                else
                    right->flakes.push_back(root->flakes[i]);
            }
            break;
        case 1:
            // Split along u0[1]
            median = (root->aa[1] + root->bb[1]) / 2.0f;
            for (size_t i = 0; i < root->flakes.size(); i++) {
                if (root->flakes[i]->u0[1] < median)
                    left->flakes.push_back(root->flakes[i]);
                else
                    right->flakes.push_back(root->flakes[i]);
            }
            break;
        }

        if (left->flakes.size() > 0) {
            if (left->flakes.size() != root->flakes.size()) {
                root->left = left;
                buildFlakesTree(root->left, currentLevel + 1);
            } else if (badTimes < 2) {
                delete left;
                buildFlakesTree(root, currentLevel + 1, badTimes + 1);
            }
        } else {
            delete left;
        }

        if (right->flakes.size() > 0) {
            if (right->flakes.size() != root->flakes.size()) {
                root->right = right;
                buildFlakesTree(root->right, currentLevel + 1);
            } else if (badTimes < 2) {
                delete right;
                buildFlakesTree(root, currentLevel + 1, badTimes + 1);
            }
        } else {
            delete right;
        }
    }

    void readFlakesFile(const char *flakesFilename) {
        FILE *fp = fopen(flakesFilename, "rb");
        int type;
        int numFlakes;
        fread(&type, sizeof(int), 1, fp);
        fread(&m_resolutionU, sizeof(int), 1, fp);
        fread(&m_resolutionV, sizeof(int), 1, fp);
        fread(&numFlakes, sizeof(int), 1, fp);

        if (type == 1) {
            printf("Reading flakes file...");
            float *buffer = new float[11 * numFlakes];
            fread(buffer, sizeof(float), 11 * numFlakes, fp);
            printf("OK!\n");
            printf("Generating flakes...");
            for (int i = 0; i < numFlakes; i++) {
                Vector2f u0(buffer[i * 11 + 0], buffer[i * 11 + 1]);
                Vector2f n0(buffer[i * 11 + 2], buffer[i * 11 + 3]);
                Vector2f shape(buffer[i * 11 + 4], buffer[i * 11 + 5]);
                Matrix2f J;
                J(0, 0) = buffer[i * 11 + 6];
                J(0, 1) = buffer[i * 11 + 7];
                J(1, 0) = buffer[i * 11 + 8];
                J(1, 1) = buffer[i * 11 + 9];
//                J << buffer[i * 11 + 6], buffer[i * 11 + 7], buffer[i * 11 + 8], buffer[i * 11 + 9];
                float area = buffer[i * 11 + 10];

                Flake *currentFlake = new LinearFlake(u0, n0, shape, area, J);
                m_flakes.push_back(currentFlake);
            }
            delete[] buffer;
            printf("OK!\n");
        }
    }

    void configure(float intrinsicRoughness, const char *gridFilename) {
        m_intrinsicRoughness = intrinsicRoughness;

        printf("Building bounding boxes for flakes...");
        for (size_t i = 0; i < m_flakes.size(); i++) {
            m_flakes[i]->getBoundingBox(intrinsicRoughness, m_flakes[i]->aa, m_flakes[i]->bb);
            m_flakes[i]->index = i;
        }
        printf("OK!\n");

        buildSTGrid(gridFilename);
        buildUVGrid();
    }

    Node *stGrid[GRID_RES][GRID_RES];
    Node *uvGrid[GRID_RES][GRID_RES];
public:
    std::vector<Flake*> m_flakes;
    float m_intrinsicRoughness;
    int m_resolutionU, m_resolutionV;

    FlakesTree() {}

    void initialize(const char *flakesFilename, float intrinsicRoughness) {
        readFlakesFile(flakesFilename);

        char t[64];
        sprintf(t, "%.04f", double(intrinsicRoughness));
        configure(intrinsicRoughness, (std::string(flakesFilename) + std::string(t) + std::string(".grid")).c_str());
    }

    void deleteFlakesTree(Node *root) {
        if (!root)
            return;

        deleteFlakesTree(root->left);
        deleteFlakesTree(root->right);
        delete root;
    }

    ~FlakesTree() {
        for (int i = 0; i < GRID_RES; i++) {
            for (int j = 0; j < GRID_RES; j++) {
                deleteFlakesTree(stGrid[i][j]);
                deleteFlakesTree(uvGrid[i][j]);
            }
        }
    }

    void queryFlakesEval(const Vector4f &uvstMin, const Vector4f &uvstMax, std::vector<Flake*> &flakes) const {
//        assert(uvstMin[2] == uvstMax[2] && uvstMin[3] == uvstMax[3]);
        if (!(uvstMin[2] == uvstMax[2] && uvstMin[3] == uvstMax[3]))
            return;
        int p = (uvstMin[2] + 1.0f) / 2.0f * GRID_RES;
        int q = (uvstMin[3] + 1.0f) / 2.0f * GRID_RES;
        queryFlakes(stGrid[p][q], uvstMin, uvstMax, flakes);
    }

    void queryFlakesSample(const Vector4f &uvstMin, const Vector4f &uvstMax, std::vector<Flake*> &flakes) const {
//        assert(uvstMin[0] == uvstMax[0] && uvstMin[1] == uvstMax[1]);
        if (!(uvstMin[0] == uvstMax[0] && uvstMin[1] == uvstMax[1]))
            return;
        int p = uvstMin[0] / m_resolutionU * GRID_RES;
        int q = uvstMin[1] / m_resolutionV * GRID_RES;
        if (p < 0 || p >= GRID_RES || q < 0 || q >= GRID_RES)
            return;
        queryFlakes(uvGrid[p][q], uvstMin, uvstMax, flakes);
    }
};

NAMESPACE_END(mitsuba)