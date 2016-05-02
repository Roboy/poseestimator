#include <GL/glew.h>
#include <SFML/OpenGL.hpp>
#include <cstring>
#include <map>
#include <fstream>
// Include GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
// Eigen
#include <Eigen/Core>
#include <Eigen/Dense>
// opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
// cuda
#include <cuda_runtime.h>

#define WIDTH 640
#define HEIGHT 480

using namespace std;
using namespace Eigen;
using namespace cv;

#define STR1(x)  #x
#define STR(x)  STR1(x)

//! cuda error checking
void cuda_check(string file, int line);
#define CUDA_CHECK cuda_check(__FILE__,__LINE__)

struct PackedVertex {
    glm::vec3 position;
    glm::vec2 uv;
    glm::vec3 normal;

    bool operator<(const PackedVertex that) const {
        return memcmp((void *) this, (void *) &that, sizeof(PackedVertex)) > 0;
    };
};

__constant__ float c_K[9];
__constant__ float c_Kinv[9];
__constant__ float c_cameraPose[16];

__global__ void costFcn(float3 *vertices_in, float3 *normals_in, float3 *positions_out, float3 *normals_out,
                        uchar *border, uchar *image, float mu_in, float mu_out, uchar *img, int numberOfVertices,
                        float3* gradTrans, float3* gradRot);

class PoseEstimator {
public:
    PoseEstimator(const char *file_path, string &object);

    ~PoseEstimator();

    bool loadShaderCodeFromFile(const char *file_path, string &src);

    void compileShader(string src, GLenum type, GLuint &shader);

    void createTransformProgram(GLuint &shader, const GLchar *feedbackVaryings[], uint numberOfVaryings,
                                GLuint &program);

    GLint createRenderProgram(GLuint &vertex_shader, GLuint &fragment_shader, GLuint &program,
                              GLuint &texture, GLint &textureSampler, Mat image, GLint &MatrixID,
                              GLint &ViewMatrixID, GLint &ModelMatrixID, GLint &LightPositionID);

    bool loadObjFile(const char *path, GLuint &vertexbuffer, vector<glm::vec3> &indexed_vertices,
                     GLuint &uvbuffer, vector<glm::vec2> &indexed_uvs,
                     GLuint &normalbuffer, vector<glm::vec3> &indexed_normals,
                     GLuint &elementbuffer, vector<unsigned short> &indices,
                     GLuint &tbo, GLsizei &numberOfVertices, GLsizei &numberOfIndices) ;

    Mat renderColor(string &object, VectorXd &pose);

    void getPose(string &object, Mat img_camera);

private:
    bool getSimilarVertexIndex_fast(PackedVertex &packed, map<PackedVertex, unsigned short> &VertexToOutIndex,
                                    unsigned short &result);

    void indexVBO(vector<glm::vec3> &in_vertices, vector<glm::vec2> &in_uvs, vector<glm::vec3> &in_normals,
                  vector<unsigned short> &out_indices, vector<glm::vec3> &out_vertices, vector<glm::vec2> &out_uvs,
                  vector<glm::vec3> &out_normals);

    map<string, GLuint> shader, program, vertexbuffer, uvbuffer, normalbuffer, elementbuffer, texture;
    map<string, vector<glm::vec3>> vertices, normals;
    map<string, vector<glm::vec2>> uvs;
    map<string, vector<unsigned short>> indices;
    map<string, GLuint>  tbo;
    map<string, GLint> ViewMatrixID, MatrixID, ModelMatrixID, KID, LightPositionID, textureSampler;
    map<string, GLsizei> numberOfVertices, numberOfIndices;
    map<string, Mat> textureImage;
    map<string, Matrix4f> ViewMatrix, ModelMatrix;
    Matrix4f ProjectionMatrix;
    Matrix3f K, Kinv; // intrinsics matrix
};
