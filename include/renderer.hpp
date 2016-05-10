// glew
#include <GL/glew.h>
// sfml
#include <SFML/OpenGL.hpp>
// glm
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>
// Eigen
#include <Eigen/Core>
#include <Eigen/Dense>
// opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
// pcl
#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
// cuda types
#include <vector_types.h>
// std
#include <cstring>
#include <map>
#include <fstream>

// Converts degrees to radians.
#define degreesToRadians(angleDegrees) (angleDegrees * (float)M_PI / 180.0f)
// Converts radians to degrees.
#define radiansToDegrees(angleRadians) (angleRadians * 180.0f / (float)M_PI)

#define WIDTH 640
#define HEIGHT 480

using namespace std;
using namespace Eigen;
using cv::Mat;

struct PackedVertex {
    glm::vec3 position;
    glm::vec2 uv;
    glm::vec3 normal;

    bool operator<(const PackedVertex that) const {
        return memcmp((void *) this, (void *) &that, sizeof(PackedVertex)) > 0;
    };
};

class Renderer {
public:
    Renderer(const char *file_path, string &object);

    ~Renderer();

    Mat renderColor(string &object, VectorXd &pose);

    void visualize(float3 *vertices, float3 *vectors, int numberOfVertices);

    map<string, vector<glm::vec3>> vertices, normals;
    Matrix3f K, Kinv; // intrinsics matrix
private:
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
                     GLuint &tbo, GLsizei &numberOfVertices, GLsizei &numberOfIndices);

    // Returns true iif v1 can be considered equal to v2
    bool is_near(float v1, float v2) {
        return fabs(v1 - v2) < 0.001f;
    }

    bool getSimilarVertexIndex(
            glm::vec3 &in_vertex,
            glm::vec2 &in_uv,
            glm::vec3 &in_normal,
            std::vector<glm::vec3> &out_vertices,
            std::vector<glm::vec2> &out_uvs,
            std::vector<glm::vec3> &out_normals,
            unsigned short &result
    );

    void indexVBO_slow(vector<glm::vec3> &in_vertices, vector<glm::vec2> &in_uvs, vector<glm::vec3> &in_normals,
                  vector<unsigned short> &out_indices, vector<glm::vec3> &out_vertices, vector<glm::vec2> &out_uvs,
                  vector<glm::vec3> &out_normals);

    map<string, GLuint> shader, program, vertexbuffer, uvbuffer, normalbuffer, elementbuffer, texture;
    map<string, vector<glm::vec2>> uvs;
    map<string, vector<unsigned short>> indices;
    map<string, GLuint> tbo;
    map<string, GLint> ViewMatrixID, MatrixID, ModelMatrixID, KID, LightPositionID, textureSampler;
    map<string, GLsizei> numberOfVertices, numberOfIndices;
    map<string, Mat> textureImage;
    map<string, Matrix4f> ViewMatrix, ModelMatrix;
    Matrix4f ProjectionMatrix;
};
