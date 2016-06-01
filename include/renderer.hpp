// glew
#include <GL/glew.h>
// sfml
#include <SFML/Window.hpp>
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
// std
#include <cstring>
#include <map>
#include <fstream>
// cuda types
#include <vector_types.h>

#include "mesh.hpp"

// Converts degrees to radians.
#define degreesToRadians(angleDegrees) (angleDegrees * (float)M_PI / 180.0f)
// Converts radians to degrees.
#define radiansToDegrees(angleRadians) (angleRadians * 180.0f / (float)M_PI)

#define WIDTH 640
#define HEIGHT 480

using namespace std;
using namespace Eigen;
using cv::Mat;

class Renderer {
public:
    Renderer(const char *rootDirectory);

    ~Renderer();

    void renderColor(Mesh *mesh, VectorXd &pose, string program = "color");

    void renderColor(Mesh *mesh, string program = "color");

    void getImage(Mat &img);

    map<string, vector<glm::vec3>> vertices, normals;
    Matrix3f K, Kinv; // intrinsics matrix
private:
    bool loadShaderCodeFromFile(const char *file_path, string &src);

    void compileShader(string src, GLenum type, GLuint &shader);

    void createTransformProgram(GLuint &shader, const GLchar *feedbackVaryings[], uint numberOfVaryings,
                                GLuint &program);

    GLint createRenderProgram(GLuint &vertex_shader, GLuint &fragment_shader, GLuint &program);

    map<string, GLuint> shader, program;
    map<string, vector<unsigned short>> indices;
    map<string, GLuint> tbo;
    map<string, GLint> ViewMatrixID, MatrixID, ModelMatrixID, LightPositionID;
public:
    Matrix4f ProjectionMatrix, ViewMatrix;
};
