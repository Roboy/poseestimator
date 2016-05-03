#include <device_launch_parameters.h>
#include "poseestimator.hpp"

// cuda error checking
string prev_file = "";
int prev_line = 0;

void cuda_check(string file, int line) {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        cout << endl << file << ", line " << line << ": " << cudaGetErrorString(e) << " (" << e << ")" << endl;
        if (prev_line > 0) cout << "Previous CUDA call:" << endl << prev_file << ", line " << prev_line << endl;
        exit(1);
    }
    prev_file = file;
    prev_line = line;
}

PoseEstimator::PoseEstimator(const char *file_path, string &object) {
    char file[200];
    // load a model
    sprintf(file, "%s/images/%s.png", file_path, object.c_str());
    textureImage[object] = imread(file);

    sprintf(file, "%s/models/%s.obj", file_path, object.c_str());
    loadObjFile(file, vertexbuffer[object], vertices[object], uvbuffer[object], uvs[object], normalbuffer[object],
                normals[object], elementbuffer[object], indices[object],
                tbo[object], numberOfVertices[object], numberOfIndices[object]);

    // create color program
    string src;
    sprintf(file, "%s/shader/color.vertexshader", file_path);
    loadShaderCodeFromFile(file, src);
    compileShader(src, GL_VERTEX_SHADER, shader["color_vertex"]);

    sprintf(file, "%s/shader/color.fragmentshader", file_path);
    loadShaderCodeFromFile(file, src);
    compileShader(src, GL_FRAGMENT_SHADER, shader["color_fragment"]);


    if (createRenderProgram(shader["color_vertex"], shader["color_fragment"], program[object], texture[object],
                            textureSampler[object], textureImage[object], MatrixID[object], ViewMatrixID[object],
                            ModelMatrixID[object], LightPositionID[object]) == GL_FALSE)
        return;

    ModelMatrix[object] = Matrix4f::Identity();
    ViewMatrix[object] = Matrix4f::Identity();
    ViewMatrix[object].topRightCorner(3, 1) << 0, 0, -0.5;

    Mat cameraMatrix, distCoeffs;
    sprintf(file, "%s/intrinsics.xml", file_path);
    FileStorage fs(file, FileStorage::READ);
    fs["camera_matrix"] >> cameraMatrix;
    fs["distortion_coefficients"] >> distCoeffs;
    fs.release();

    // calculate undistortion mapping
    Mat img_rectified, map1, map2;
    initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(),
                            getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, Size(WIDTH, HEIGHT), 1,
                                                      Size(WIDTH, HEIGHT), 0),
                            Size(WIDTH, HEIGHT), CV_16SC2, map1, map2);

    float n = 0.01; // near field
    float f = 100; // far field
    ProjectionMatrix << cameraMatrix.at<double>(0, 0) / cameraMatrix.at<double>(0, 2), 0.0, 0.0, 0.0,
            0.0, cameraMatrix.at<double>(1, 1) / cameraMatrix.at<double>(1, 2), 0.0, 0.0,
            0.0, 0.0, -(f + n) / (f - n), (-2.0f * f * n) / (f - n),
            0.0, 0.0, -1.0, 0.0;
    K << cameraMatrix.at<double>(0, 0), cameraMatrix.at<double>(0, 1), cameraMatrix.at<double>(0, 2),
            cameraMatrix.at<double>(1, 0), cameraMatrix.at<double>(1, 1), cameraMatrix.at<double>(1, 2),
            cameraMatrix.at<double>(2, 0), cameraMatrix.at<double>(2, 1), cameraMatrix.at<double>(2, 2);
    cout << "K\n" << K << endl;
    Kinv = K.inverse();

    cout << "ProjectionMatrix\n" << ProjectionMatrix << endl;
    cout << "ModelMatrix\n" << ModelMatrix[object] << endl;
    cout << "ViewMatrix\n" << ViewMatrix[object] << endl;

    // background ccolor
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    // Enable depth test
    glEnable(GL_DEPTH_TEST);
    // Accept fragment if it closer to the camera than the former one
    glDepthFunc(GL_LESS);
    // Cull triangles which normal is not towards the camera
    glEnable(GL_CULL_FACE);

    // initialize cuda
    cudaDeviceSynchronize();
    CUDA_CHECK;
}

PoseEstimator::~PoseEstimator() {
    for (auto p:program)
        glDeleteProgram(p.second);
    for (auto t:tbo)
        glDeleteBuffers(1, &t.second);
}

bool PoseEstimator::loadShaderCodeFromFile(const char *file_path, string &src) {
    src.clear();
    ifstream VertexShaderStream(file_path, ios::in);
    if (VertexShaderStream.is_open()) {
        string Line = "";
        while (getline(VertexShaderStream, Line))
            src += "\n" + Line;
        VertexShaderStream.close();
        return true;
    } else {
        printf("Cannot read %s\n", file_path);
        getchar();
        return false;
    }
}

void PoseEstimator::compileShader(string src, GLenum type, GLuint &shader) {
    shader = glCreateShader(type);
    const char *c_str = src.c_str();
    glShaderSource(shader, 1, &c_str, nullptr);
    glCompileShader(shader);
}

GLint PoseEstimator::createRenderProgram(GLuint &vertex_shader, GLuint &fragment_shader, GLuint &program,
                                         GLuint &texture, GLint &textureSampler, Mat image, GLint &MatrixID,
                                         GLint &ViewMatrixID, GLint &ModelMatrixID, GLint &LightPositionID) {
    // Link the program
    program = glCreateProgram();
    glAttachShader(program, vertex_shader);
    glAttachShader(program, fragment_shader);
    glLinkProgram(program);

    int params = -1;
    glGetProgramiv(program, GL_LINK_STATUS, &params);
    printf("GL_LINK_STATUS = %i\n", params);

    glGetProgramiv(program, GL_ATTACHED_SHADERS, &params);
    printf("GL_ATTACHED_SHADERS = %i\n", params);

    glGetProgramiv(program, GL_ACTIVE_ATTRIBUTES, &params);
    printf("GL_ACTIVE_ATTRIBUTES = %i\n", params);

    GLint Result = GL_FALSE;
    int InfoLogLength;
    glGetShaderiv(vertex_shader, GL_COMPILE_STATUS, &Result);
    glGetShaderiv(vertex_shader, GL_INFO_LOG_LENGTH, &InfoLogLength);
    if (InfoLogLength > 0) {
        vector<char> VertexShaderErrorMessage(InfoLogLength + 1);
        glGetShaderInfoLog(vertex_shader, InfoLogLength, NULL, &VertexShaderErrorMessage[0]);
        printf("%s\n", &VertexShaderErrorMessage[0]);
        return Result;
    }

    glGetShaderiv(fragment_shader, GL_COMPILE_STATUS, &Result);
    glGetShaderiv(fragment_shader, GL_INFO_LOG_LENGTH, &InfoLogLength);
    if (InfoLogLength > 0) {
        vector<char> VertexShaderErrorMessage(InfoLogLength + 1);
        glGetShaderInfoLog(fragment_shader, InfoLogLength, NULL, &VertexShaderErrorMessage[0]);
        printf("%s\n", &VertexShaderErrorMessage[0]);
        return Result;
    }

    glDetachShader(program, vertex_shader);
    glDetachShader(program, fragment_shader);

    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);

    MatrixID = glGetUniformLocation(program, "MVP");
    ViewMatrixID = glGetUniformLocation(program, "ViewMatrix");
    ModelMatrixID = glGetUniformLocation(program, "ModelMatrix");
    LightPositionID = glGetUniformLocation(program, "LightPosition_worldspace");

    // Create one OpenGL texture
    glGenTextures(1, &texture);

    // "Bind" the newly created texture : all future texture functions will modify this texture
    glBindTexture(GL_TEXTURE_2D, texture);

    // Give the image to OpenGL
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.cols, image.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, image.data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glGenerateMipmap(GL_TEXTURE_2D);

    // filtering
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    // Set our "tex" sampler to user Texture Unit 0
    textureSampler = glGetUniformLocation(program, "myTextureSampler");
    glUniform1i(textureSampler, 0);
    return Result;
}

bool PoseEstimator::loadObjFile(const char *path, GLuint &vertexbuffer, vector<glm::vec3> &indexed_vertices,
                                GLuint &uvbuffer, vector<glm::vec2> &indexed_uvs,
                                GLuint &normalbuffer, vector<glm::vec3> &indexed_normals,
                                GLuint &elementbuffer, vector<unsigned short> &indices,
                                GLuint &tbo, GLsizei &numberOfVertices, GLsizei &numberOfIndices) {
    printf("Loading OBJ file %s...\n", path);

    vector<unsigned int> vertexIndices, uvIndices, normalIndices;
    vector<glm::vec3> temp_vertices, vertices;
    vector<glm::vec2> temp_uvs, uvs;
    vector<glm::vec3> temp_normals, normals;


    FILE *file = fopen(path, "r");
    if (file == NULL) {
        printf("Impossible to open the file ! Are you in the right path ? See Tutorial 1 for details\n");
        getchar();
        return false;
    }

    while (1) {

        char lineHeader[128];
        // read the first word of the line
        int res = fscanf(file, "%s", lineHeader);
        if (res == EOF)
            break; // EOF = End Of File. Quit the loop.

        // else : parse lineHeader

        if (strcmp(lineHeader, "v") == 0) {
            glm::vec3 vertex;
            fscanf(file, "%f %f %f\n", &vertex.x, &vertex.y, &vertex.z);
            temp_vertices.push_back(vertex);
        } else if (strcmp(lineHeader, "vt") == 0) {
            glm::vec2 uv;
            fscanf(file, "%f %f\n", &uv.x, &uv.y);
            uv.y = -uv.y; // Invert V coordinate since we will only use DDS texture, which are inverted. Remove if you want to use TGA or BMP loaders.
            temp_uvs.push_back(uv);
        } else if (strcmp(lineHeader, "vn") == 0) {
            glm::vec3 normal;
            fscanf(file, "%f %f %f\n", &normal.x, &normal.y, &normal.z);
            temp_normals.push_back(glm::normalize(normal));
        } else if (strcmp(lineHeader, "f") == 0) {
            string vertex1, vertex2, vertex3;
            unsigned int vertexIndex[3], uvIndex[3], normalIndex[3];
            int matches = fscanf(file, "%d/%d/%d %d/%d/%d %d/%d/%d\n", &vertexIndex[0], &uvIndex[0],
                                 &normalIndex[0], &vertexIndex[1], &uvIndex[1], &normalIndex[1], &vertexIndex[2],
                                 &uvIndex[2], &normalIndex[2]);
            if (matches != 9) {
                printf("File can't be read by our simple parser :-( Try exporting with other options\n");
                return false;
            }
            vertexIndices.push_back(vertexIndex[0]);
            vertexIndices.push_back(vertexIndex[1]);
            vertexIndices.push_back(vertexIndex[2]);
            uvIndices.push_back(uvIndex[0]);
            uvIndices.push_back(uvIndex[1]);
            uvIndices.push_back(uvIndex[2]);
            normalIndices.push_back(normalIndex[0]);
            normalIndices.push_back(normalIndex[1]);
            normalIndices.push_back(normalIndex[2]);
        } else {
            // Probably a comment, eat up the rest of the line
            char stupidBuffer[1000];
            fgets(stupidBuffer, 1000, file);
        }
    }

    // For each vertex of each triangle
    for (unsigned int i = 0; i < vertexIndices.size(); i++) {

        // Get the indices of its attributes
        unsigned int vertexIndex = vertexIndices[i];
        unsigned int uvIndex = uvIndices[i];
        unsigned int normalIndex = normalIndices[i];

        // Get the attributes thanks to the index
        glm::vec3 vertex = temp_vertices[vertexIndex - 1];
        glm::vec2 uv = temp_uvs[uvIndex - 1];
        glm::vec3 normal = temp_normals[normalIndex - 1];

        // Put the attributes in buffers
        vertices.push_back(vertex);
        uvs.push_back(uv);
        normals.push_back(normal);
    }

    indexVBO(vertices, uvs, normals, indices, indexed_vertices, indexed_uvs, indexed_normals);
    numberOfVertices = indexed_vertices.size();
    numberOfIndices = indices.size();

    // Load it into a VBO
    glGenBuffers(1, &vertexbuffer);
    glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
    glBufferData(GL_ARRAY_BUFFER, numberOfVertices * sizeof(glm::vec3), &indexed_vertices[0], GL_STATIC_DRAW);

    glGenBuffers(1, &uvbuffer);
    glBindBuffer(GL_ARRAY_BUFFER, uvbuffer);
    glBufferData(GL_ARRAY_BUFFER, numberOfVertices * sizeof(glm::vec2), &indexed_uvs[0], GL_STATIC_DRAW);

    glGenBuffers(1, &normalbuffer);
    glBindBuffer(GL_ARRAY_BUFFER, normalbuffer);
    glBufferData(GL_ARRAY_BUFFER, numberOfVertices * sizeof(glm::vec3), &indexed_normals[0], GL_STATIC_DRAW);

    // Generate a buffer for the indices as well
    glGenBuffers(1, &elementbuffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementbuffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, numberOfIndices * sizeof(unsigned short), &indices[0], GL_STATIC_DRAW);

    // Generate transform buffers for positions and normals
    glGenBuffers(1, &tbo);
    glBindBuffer(GL_ARRAY_BUFFER, tbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * numberOfIndices * 3, nullptr, GL_STATIC_READ);

    return true;
}

Mat PoseEstimator::renderColor(string &object, VectorXd &pose) {
    Matrix3f rot = Matrix3f::Identity();
    Vector3f p(pose(3), pose(4), pose(5));
    float angle = p.norm();
    if (abs(angle) > 0.0000001) {
        p.normalize();
        Matrix3f skew;
        skew << 0, -p(2), p(1),
                p(2), 0, -p(0),
                -p(1), p(0), 0;
        rot = rot + sin(angle) * skew;
        rot = rot + (1.0 - cos(angle)) * skew * skew;
    }

    ViewMatrix[object] = Matrix4f::Identity();
    ViewMatrix[object].topLeftCorner(3, 3) = rot;
    ViewMatrix[object].topRightCorner(3, 1) << pose(0), pose(1), pose(2);

    Eigen::Matrix4f MVP = ProjectionMatrix * ViewMatrix[object] * ModelMatrix[object];
    glUseProgram(program[object]);
    glUniformMatrix4fv(ViewMatrixID[object], 1, GL_FALSE, &ViewMatrix[object](0, 0));
    glUniformMatrix4fv(MatrixID[object], 1, GL_FALSE, &MVP(0, 0));
    glUniformMatrix4fv(ModelMatrixID[object], 1, GL_FALSE, &ModelMatrix[object](0, 0));
    Vector3f lightPosition(0, 1, 1);
    glUniform3fv(LightPositionID[object], 1, &lightPosition(0));

    // Bind our texture in Unit 0
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture[object]);
    // Set our "myTextureSampler" sampler to user Texture Unit 0
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, textureImage[object].cols, textureImage[object].rows, 0, GL_BGR,
                 GL_UNSIGNED_BYTE, textureImage[object].data);
    glUniform1i(textureSampler[object], 0);

    // Clear the screen
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // 1rst attribute buffer : vertices
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer[object]);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void *) 0);

    // 2nd attribute buffer : UVs
    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, uvbuffer[object]);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, (void *) 0);

    // 3rd attribute buffer : normals
    glEnableVertexAttribArray(2);
    glBindBuffer(GL_ARRAY_BUFFER, normalbuffer[object]);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, (void *) 0);

    // Index buffer
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementbuffer[object]);

    // Draw the triangles !
    glDrawElements(GL_TRIANGLES, numberOfIndices[object], GL_UNSIGNED_SHORT, (void *) 0);

    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(2);

    // get the image from opengl buffer
    GLubyte *data = new GLubyte[3 * WIDTH * HEIGHT];
    glReadPixels(0, 0, WIDTH, HEIGHT, GL_BGR, GL_UNSIGNED_BYTE, data);
    Mat img = cv::Mat(HEIGHT, WIDTH, CV_8UC3, data);
    flip(img, img, -1);
    return img;
}

void PoseEstimator::getPose(string &object, Mat img_camera, VectorXd &pose, float lambda_trans, float lambda_rot) {
    Mat img_camera_gray;
    VectorXd grad(6);
    VectorXd initial_pose = pose;

    float3 *d_vertices = NULL, *d_normals = NULL, *d_vertices_out = NULL, *d_normals_out = NULL,
            *d_gradTrans = NULL, *d_gradRot = NULL;
    uchar *d_image = NULL, *d_border = NULL, *d_img_out = NULL;

    // allocate memory on gpu
    cudaMalloc(&d_vertices, vertices[object].size() * sizeof(float3));
    CUDA_CHECK;
    cudaMalloc(&d_vertices_out, vertices[object].size() * sizeof(float3));
    CUDA_CHECK;
    cudaMalloc(&d_normals, normals[object].size() * sizeof(float3));
    CUDA_CHECK;
    cudaMalloc(&d_normals_out, normals[object].size() * sizeof(float3));
    CUDA_CHECK;
    cudaMalloc(&d_gradTrans, vertices[object].size() * sizeof(float3));
    CUDA_CHECK;
    cudaMalloc(&d_gradRot, vertices[object].size() * sizeof(float3));
    CUDA_CHECK;
    cudaMalloc(&d_border, WIDTH * HEIGHT * sizeof(uchar));
    CUDA_CHECK;
    cudaMalloc(&d_img_out, WIDTH * HEIGHT * sizeof(uchar));
    CUDA_CHECK;
    cudaMalloc(&d_image, WIDTH * HEIGHT * sizeof(uchar));
    CUDA_CHECK;

    // copy camera matrices to gpu
    cudaMemcpyToSymbol(c_K, &K(0, 0), 9 * sizeof(float));
    cudaMemcpyToSymbol(c_Kinv, &Kinv(0, 0), 9 * sizeof(float));

    uchar *res = new uchar[WIDTH * HEIGHT];
    float3 *vertices_out = new float3[vertices[object].size()];
    float3 *normals_out = new float3[normals[object].size()];
    float3 *gradTrans = new float3[normals[object].size()];
    float3 *gradRot = new float3[normals[object].size()];

    for (uint iter = 0; iter < 10000; iter++) {
        Mat img_camera_copy;
        img_camera.copyTo(img_camera_copy);
        cvtColor(img_camera, img_camera_gray, CV_BGR2GRAY);
        Mat img_estimator = renderColor(object, pose), img_estimator_gray, img_estimator_gray2;
        cvtColor(img_estimator, img_estimator_gray, CV_BGR2GRAY);
        // make a copy
        img_estimator_gray.copyTo(img_estimator_gray2);

        vector<vector<cv::Point> > contours;
        vector<cv::Vec4i> hierarchy;
        findContours(img_estimator_gray, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_TC89_L1,
                     cv::Point(0, 0));
        double min_contour_area = 40;
        for (auto it = contours.begin(); it != contours.end();) {
            if (contourArea(*it) < min_contour_area)
                it = contours.erase(it);
            else
                ++it;
        }
        if (contours.size() > 0) {
            Mat border = Mat::zeros(HEIGHT, WIDTH, CV_8UC1);
            for (int idx = 0; idx < contours.size(); idx++) {
                drawContours(border, contours, idx, 255, 2, 8, hierarchy, 0, cv::Point());
                drawContours(img_camera_copy, contours, idx, Scalar(0,255,0), 1, 8, hierarchy, 0, cv::Point());
            }
            imshow("camera image", img_camera_copy);

            Mat R_mask = Mat::zeros(HEIGHT, WIDTH, CV_8UC1), Rc_mask,
                    R = Mat::zeros(HEIGHT, WIDTH, CV_8UC1),
                    Rc = Mat::zeros(HEIGHT, WIDTH, CV_8UC1);
            fillPoly(R_mask, contours, 255);
            bitwise_not(R_mask, Rc_mask);

            // this will mask out the respective part of the webcam image
            bitwise_and(img_estimator_gray2, R_mask, R);
            bitwise_and(img_estimator_gray2, Rc_mask, Rc);

            // convert camera image to float
            R.convertTo(R, CV_32FC1);
            Rc.convertTo(Rc, CV_32FC1);
            double A_in = contourArea(contours[0]);
            double A_out = WIDTH * HEIGHT - A_in;
            double mu_in = sum(R).val[0] / A_in;
            double mu_out = sum(Rc).val[0] / A_out;
            R = R - mu_in;
            Rc = Rc - mu_out;

            // copy only the respective areas
            Mat Rpow = Mat::zeros(HEIGHT, WIDTH, CV_32FC1), Rcpow = Mat::zeros(HEIGHT, WIDTH, CV_32FC1);
            R.copyTo(Rpow, R_mask);
            Rc.copyTo(Rcpow, Rc_mask);

            pow(Rpow, 2.0, Rpow);
            pow(Rcpow, 2.0, Rcpow);

            double sigma_in = sum(Rpow).val[0] / A_in;
            double sigma_out = sum(Rcpow).val[0] / A_out;

            double energy = -sum(Rpow).val[0] - sum(Rcpow).val[0];
            cout << "cost: " << energy << endl;

            Matrix3f rot = Matrix3f::Identity();
            Matrix3f skew;
            Vector3f p(pose(3), pose(4), pose(5));
            float angle = p.norm();
            if (abs(angle) > 0.0000001) {
                p.normalize();
                skew << 0, -p(2), p(1),
                        p(2), 0, -p(0),
                        -p(1), p(0), 0;
                rot = rot + sin(angle) * skew;
                rot = rot + (1.0 - cos(angle)) * skew * skew;
            }

            ViewMatrix[object] = Matrix4f::Identity();
            ViewMatrix[object].topLeftCorner(3, 3) = rot;
            ViewMatrix[object].topRightCorner(3, 1) << pose(0), pose(1), pose(2);

            Eigen::Matrix4f cameraPose = ViewMatrix[object] * ModelMatrix[object];

            // copy data to gpu
            cudaMemcpy(d_vertices, vertices[object].data(), vertices[object].size() * sizeof(float3),
                       cudaMemcpyHostToDevice);
            CUDA_CHECK;
            cudaMemcpy(d_normals, normals[object].data(), normals[object].size() * sizeof(float3),
                       cudaMemcpyHostToDevice);
            CUDA_CHECK;
            cudaMemcpy(d_border, border.data, WIDTH * HEIGHT * sizeof(uchar), cudaMemcpyHostToDevice);
            CUDA_CHECK;
            cudaMemcpy(d_image, img_camera_gray.data, WIDTH * HEIGHT * sizeof(uchar), cudaMemcpyHostToDevice);
            CUDA_CHECK;
            // set result image an the gradients to zero
            cudaMemset(d_img_out, 0, WIDTH * HEIGHT * sizeof(uchar));
            CUDA_CHECK;
            // set constants on gpu
            cudaMemcpyToSymbol(c_cameraPose, &cameraPose(0, 0), 16 * sizeof(float));

            dim3 block = dim3(1, 1, 1);
            dim3 grid = dim3(vertices[object].size() / block.x, 1, 1);

            costFcn << < grid, block >> >
                               (d_vertices, d_normals, d_vertices_out, d_normals_out, d_border, d_image, mu_in, mu_out,
                                       sigma_in, sigma_out, d_img_out, vertices[object].size(), d_gradTrans, d_gradRot);
            CUDA_CHECK;

            // copy data from gpu to cpu
            cudaMemcpy(res, d_img_out, WIDTH * HEIGHT * sizeof(uchar), cudaMemcpyDeviceToHost);
            CUDA_CHECK;
            cudaMemcpy(vertices_out, d_vertices_out, vertices[object].size() * sizeof(float3), cudaMemcpyDeviceToHost);
            CUDA_CHECK;
            cudaMemcpy(normals_out, d_normals_out, normals[object].size() * sizeof(float3), cudaMemcpyDeviceToHost);
            CUDA_CHECK;
            cudaMemcpy(gradTrans, d_gradTrans, normals[object].size() * sizeof(float3), cudaMemcpyDeviceToHost);
            CUDA_CHECK;
            cudaMemcpy(gradRot, d_gradRot, normals[object].size() * sizeof(float3), cudaMemcpyDeviceToHost);
            CUDA_CHECK;

            grad << 0, 0, 0, 0, 0, 0;
            for (uint i = 0; i < vertices[object].size(); i++) {
//                cout << "v: " << vertices_out[i].x << " " << vertices_out[i].y << " " << vertices_out[i].z << endl;
//                cout << "n: " << normals_out[i].x << " " << normals_out[i].y << " " << normals_out[i].z << endl;
//                cout << "g: " << gradTrans[i].x << " " << gradTrans[i].y << " " << gradTrans[i].z << endl;
//                cout << "g: " << gradRot[i].x << " " << gradRot[i].y << " " << gradRot[i].z << endl;
//                Vector3f n(normals_out[i].x, normals_out[i].y, normals_out[i].z);
//                cout << n.norm() << endl;
                grad(0) += gradTrans[i].x;
                grad(1) += gradTrans[i].y;
                grad(2) += gradTrans[i].z;
                grad(3) += gradRot[i].x;
                grad(4) += gradRot[i].y;
                grad(5) += gradRot[i].z;
            }

            pose(0) += lambda_trans*grad(0);
            pose(1) += lambda_trans*grad(1);
            pose(2) += lambda_trans*grad(2);
            pose(3) += lambda_rot*grad(3);
            pose(4) += lambda_rot*grad(4);
            pose(5) += lambda_rot*grad(5);
            cout << "pose:\n" << pose << endl;

            Mat img(HEIGHT, WIDTH, CV_8UC1, res);
            imshow("result", img);

            for (int idx = 0; idx < contours.size(); idx++) {
                drawContours(img_camera_gray, contours, idx, 255, 2, 8, hierarchy, 0, cv::Point());
            }
            imshow("img_camera_gray", img_camera_gray);
            waitKey(1);
        } else {
            cout << "cannot find any contour" << endl;
            return;
        }
    }
    cudaFree(d_vertices);
    CUDA_CHECK;
    cudaFree(d_vertices_out);
    CUDA_CHECK;
    cudaFree(d_normals);
    CUDA_CHECK;
    cudaFree(d_normals_out);
    CUDA_CHECK;
    cudaFree(d_gradTrans);
    CUDA_CHECK;
    cudaFree(d_gradRot);
    CUDA_CHECK;
    cudaFree(d_border);
    CUDA_CHECK;
    cudaFree(d_img_out);
    CUDA_CHECK;
    cudaFree(d_image);
    CUDA_CHECK;

    delete[] res;
    delete[] vertices_out;
    delete[] normals_out;
    delete[] gradTrans;
    delete[] gradRot;
}

__global__ void costFcn(float3 *vertices_in, float3 *normals_in, float3 *positions_out, float3 *normals_out,
                        uchar *border, uchar *image, float mu_in, float mu_out, float sigma_in, float sigma_out,
                        uchar *img_out, int numberOfVertices, float3 *gradTrans, float3 *gradRot) {
    // iteration over image is parallelized
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < numberOfVertices) {
        // set gradients to zero
        gradTrans[idx].x = 0;
        gradTrans[idx].y = 0;
        gradTrans[idx].z = 0;
        gradRot[idx].x = 0;
        gradRot[idx].y = 0;
        gradRot[idx].z = 0;

        float3 v = vertices_in[idx];
        float3 n = normals_in[idx];

        // calculate position of vertex in camera coordinate system
        float3 pos;
        pos.x = 0.0f;
        pos.y = 0.0f;
        pos.z = 0.0f;

        // x
        pos.x += c_cameraPose[0 + 4 * 0] * v.x;
        pos.x += c_cameraPose[0 + 4 * 1] * v.y;
        pos.x += c_cameraPose[0 + 4 * 2] * v.z;
        pos.x += c_cameraPose[0 + 4 * 3];

        // y
        pos.y += c_cameraPose[1 + 4 * 0] * v.x;
        pos.y += c_cameraPose[1 + 4 * 1] * v.y;
        pos.y += c_cameraPose[1 + 4 * 2] * v.z;
        pos.y += c_cameraPose[1 + 4 * 3];

        // z
        pos.z += c_cameraPose[2 + 4 * 0] * v.x;
        pos.z += c_cameraPose[2 + 4 * 1] * v.y;
        pos.z += c_cameraPose[2 + 4 * 2] * v.z;
        pos.z += c_cameraPose[2 + 4 * 3];

        positions_out[idx] = pos;

        float posNorm = sqrtf(pos.x * pos.x + pos.y * pos.y + pos.z * pos.z);

        // calculate orientation of normal in camera coordinate system
        float3 normal;
        normal.x = 0.0f;
        normal.y = 0.0f;
        normal.z = 0.0f;

        // x
        normal.x += c_cameraPose[0 + 4 * 0] * n.x;
        normal.x += c_cameraPose[0 + 4 * 1] * n.y;
        normal.x += c_cameraPose[0 + 4 * 2] * n.z;

        // y
        normal.y += c_cameraPose[1 + 4 * 0] * n.x;
        normal.y += c_cameraPose[1 + 4 * 1] * n.y;
        normal.y += c_cameraPose[1 + 4 * 2] * n.z;

        // z
        normal.z += c_cameraPose[2 + 4 * 0] * n.x;
        normal.z += c_cameraPose[2 + 4 * 1] * n.y;
        normal.z += c_cameraPose[2 + 4 * 2] * n.z;

        normals_out[idx] = normal;

        // calculate dot product position and normal
        float dot = normal.x * pos.x / posNorm + normal.y * pos.y / posNorm + normal.z * pos.z / posNorm;

        // calculate gradient of silhuette
        float3 cross = {pos.y * normal.z - pos.z * normal.y,
                        pos.z * normal.x - pos.x * normal.z,
                        pos.x * normal.y - pos.y * normal.x};
        float dCnorm = sqrtf(cross.x * cross.x + cross.y * cross.y + cross.z * cross.z);

        // calculate pixel location with intrinsic matrix K
        float3 pixel;
        pixel.x = 0.0f;
        pixel.y = 0.0f;
        pixel.z = 0.0f;

        // x
        pixel.x += c_K[0 + 3 * 0] * pos.x;
        pixel.x += c_K[0 + 3 * 1] * pos.y;
        pixel.x += c_K[0 + 3 * 2] * pos.z;

        // y
        pixel.y += c_K[1 + 3 * 0] * pos.x;
        pixel.y += c_K[1 + 3 * 1] * pos.y;
        pixel.y += c_K[1 + 3 * 2] * pos.z;

        // z
        pixel.z += c_K[2 + 3 * 0] * pos.x;
        pixel.z += c_K[2 + 3 * 1] * pos.y;
        pixel.z += c_K[2 + 3 * 2] * pos.z;

        int2 pixelCoord;
        pixelCoord.x = (int) pixel.x / pixel.z;
        pixelCoord.y = (int) pixel.y / pixel.z;
        // if its a border pixel and the dot product small enough
        if (pixelCoord.x >= 0 && pixelCoord.x < WIDTH && pixelCoord.y >= 0 && pixelCoord.y < HEIGHT &&
            (dot < 0.1f && dot > -0.1f) && border[pixelCoord.y * WIDTH + pixelCoord.x] == 255) {
            img_out[pixelCoord.y * WIDTH + pixelCoord.x] = 255;
            float Rc = (((float) image[pixelCoord.y * WIDTH + pixelCoord.x] - mu_out) *
                       ((float) image[pixelCoord.y * WIDTH + pixelCoord.x] - mu_out))/sigma_out;
            float R = (((float) image[pixelCoord.y * WIDTH + pixelCoord.x] - mu_in) *
                      ((float) image[pixelCoord.y * WIDTH + pixelCoord.x] - mu_in))/sigma_in;
            float statistics = (logf(sigma_out/sigma_in)+ Rc - R) * dCnorm;//
            gradTrans[idx].x = statistics * normal.x;
            gradTrans[idx].y = statistics * normal.y;
            gradTrans[idx].z = statistics * normal.z;

            float Om[9] = {0, v.z, -v.y,
                           -v.z, 0, v.x,
                           v.y, -v.x, 0};
            float M[9] = {0, 0, 0,
                          0, 0, 0,
                          0, 0, 0};

            for (uint i = 0; i < 3; i++)
                for (uint j = 0; j < 3; j++)
                    for (uint k = 0; k < 3; k++)
                        M[i + 3 * j] += c_cameraPose[i + 4 * k] * Om[k + 3 * j];
            statistics *= posNorm/(pos.z*pos.z*pos.z);
            gradRot[idx].x = statistics * (M[0 + 3 * 0] * normal.x + M[1 + 3 * 0] * normal.y + M[2 + 3 * 0] * normal.z);
            gradRot[idx].y = statistics * (M[0 + 3 * 1] * normal.x + M[1 + 3 * 1] * normal.y + M[2 + 3 * 1] * normal.z);
            gradRot[idx].z = statistics * (M[0 + 3 * 2] * normal.x + M[1 + 3 * 2] * normal.y + M[2 + 3 * 2] * normal.z);
        }
    }
}

bool PoseEstimator::getSimilarVertexIndex_fast(PackedVertex &packed,
                                               map<PackedVertex, unsigned short> &VertexToOutIndex,
                                               unsigned short &result) {
    map<PackedVertex, unsigned short>::iterator it = VertexToOutIndex.find(packed);
    if (it == VertexToOutIndex.end()) {
        return false;
    } else {
        result = it->second;
        return true;
    }
}

void PoseEstimator::indexVBO(vector<glm::vec3> &in_vertices, vector<glm::vec2> &in_uvs,
                             vector<glm::vec3> &in_normals,
                             vector<unsigned short> &out_indices, vector<glm::vec3> &out_vertices,
                             vector<glm::vec2> &out_uvs,
                             vector<glm::vec3> &out_normals) {
    map<PackedVertex, unsigned short> VertexToOutIndex;

    // For each input vertex
    for (unsigned int i = 0; i < in_vertices.size(); i++) {
        PackedVertex packed = {in_vertices[i], in_uvs[i], in_normals[i]};
        // Try to find a similar vertex in out_XXXX
        unsigned short index;
        bool found = getSimilarVertexIndex_fast(packed, VertexToOutIndex, index);
        if (found) { // A similar vertex is already in the VBO, use it instead !
            out_indices.push_back(index);
        } else { // If not, it needs to be added in the output data.
            out_vertices.push_back(in_vertices[i]);
            out_uvs.push_back(in_uvs[i]);
            out_normals.push_back(in_normals[i]);
            unsigned short newindex = (unsigned short) out_vertices.size() - 1;
            out_indices.push_back(newindex);
            VertexToOutIndex[packed] = newindex;
        }
    }
};
