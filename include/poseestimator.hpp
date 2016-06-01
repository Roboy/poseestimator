#pragma once
// Include GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
// opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
// Eigen
#include <Eigen/Core>
#include <Eigen/Dense>
#include <string>
// mesh
#include "mesh.hpp"
// cuda
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

//#define VISUALIZE

using namespace std;
using namespace Eigen;
using cv::Mat;

#define WIDTH 640
#define HEIGHT 480

#define STR1(x)  #x
#define STR(x)  STR1(x)

//! cuda error checking
void cuda_check(string file, int line);

#define CUDA_CHECK cuda_check(__FILE__,__LINE__)

__global__ void costFcn(Vertex *vertices, float3 *vertices_out, float3 *normals_out, float3 *tangents_out,
                        uchar *border, uchar *image, float mu_in, float mu_out, float sigma_in, float sigma_out,
                        uchar *img_out, int numberOfVertices, float3 *gradTrans, float3 *gradRot);


__global__ void deviceParSum(float3 *grad, int numberOfVertices, float* gradSum);

struct ModelData{
    Matrix4f *ModelMatrix;
    vector<struct cudaGraphicsResource *> cuda_vbo_resource;
    vector<size_t> numberOfVertices;
    vector<float3*> vertices_out;
    vector<float3*> normals_out;
    vector<float3*> tangents_out;
    vector<float3*> d_vertices_out;
    vector<float3*> d_normals_out;
    vector<float3*> d_tangents_out;
    vector<float3*> gradRot;
    vector<float3*> gradTrans;
    vector<float3*> d_gradRot;
    vector<float3*> d_gradTrans;
    ~ModelData(){
        for(auto v:vertices_out)
            delete[] v;
        for(auto n:normals_out)
            delete[] n;
        for(auto t:tangents_out)
            delete[] t;
        for(auto g:gradRot)
            delete[] g;
        for(auto g:gradTrans)
            delete[] g;
        for(auto v:d_vertices_out) {
            cudaFree(v);
            CUDA_CHECK;
        }
        for(auto n:d_normals_out) {
            cudaFree(n);
            CUDA_CHECK;
        }
        for(auto t:d_tangents_out) {
            cudaFree(t);
            CUDA_CHECK;
        }
        for(auto g:d_gradRot) {
            cudaFree(g);
            CUDA_CHECK;
        }
        for(auto g:d_gradTrans) {
            cudaFree(g);
            CUDA_CHECK;
        }
    }
};

class Poseestimator {
public:
    Poseestimator(vector<Mesh*> meshes, Matrix3f &K);

    ~Poseestimator();

    double iterateOnce(const Mat &img_camera, Mat &img_artificial, VectorXd &pose, VectorXd &grad);
    vector<ModelData*> modelData;
    vector<double> cost;
private:
    float *d_gradient = NULL;
    uchar *d_image = NULL, *d_border = NULL, *d_img_out = NULL, *res;
};
