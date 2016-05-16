// Include GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
// opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
// cuda
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
// Eigen
#include <Eigen/Core>
#include <Eigen/Dense>
#include <string>
// timer
#include "timer.hpp"

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

__global__ void costFcn(float3 *vertices_in, float3 *normals_in, float3 *vertices_out, float3 *normals_out,
                        float3 *tangents_out, uchar *border, uchar *image, float mu_in, float mu_out, float sigma_in,
                        float sigma_out, uchar *img_out, int numberOfVertices, float3 *gradTrans, float3 *gradRot);

class Poseestimator {
public:
    Poseestimator(uint numberOfVertices, Matrix3f &K);

    ~Poseestimator();

    double iterateOnce(Mat img_camera, Mat img_artificial, VectorXd &pose, VectorXd &grad);
    float3 *vertices_out, *normals_out, *tangents_out;
    int m_numberOfVertices = 0;
private:
    float3 *d_vertices = NULL, *d_normals = NULL, *d_vertices_out = NULL, *d_normals_out = NULL, *d_tangents_out = NULL,
            *d_gradTrans = NULL, *d_gradRot = NULL;
    uchar *d_image = NULL, *d_border = NULL, *d_img_out = NULL, *res;
    float3 *gradTrans, *gradRot;
    Timer timer;
};
