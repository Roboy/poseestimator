#include "renderer.hpp"

Renderer::Renderer(const char *file_path, string &object) {
    char file[200];
    // load a model
    sprintf(file, "%s/images/%s.png", file_path, object.c_str());
    textureImage[object] = cv::imread(file);

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
    cv::FileStorage fs(file, cv::FileStorage::READ);
    fs["camera_matrix"] >> cameraMatrix;
    fs["distortion_coefficients"] >> distCoeffs;
    fs.release();

    // calculate undistortion mapping
    Mat img_rectified, map1, map2;
    initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(),
                            getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, cv::Size(WIDTH, HEIGHT), 1,
                                                      cv::Size(WIDTH, HEIGHT), 0),
                            cv::Size(WIDTH, HEIGHT), CV_16SC2, map1, map2);

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
}

Renderer::~Renderer() {
    for (auto p:program)
        glDeleteProgram(p.second);
    for (auto t:tbo)
        glDeleteBuffers(1, &t.second);
}

Mat Renderer::renderColor(string &object, VectorXd &pose) {
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

bool Renderer::loadShaderCodeFromFile(const char *file_path, string &src) {
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

void Renderer::visualize(float3 *vertices, float3 *vectors, int numberOfVertices) {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::Normal>::Ptr cloud_vectors_ptr(new pcl::PointCloud<pcl::Normal>);
    uint8_t r(255), g(255), b(255);
    for (uint i = 0; i < numberOfVertices; i++) {
        if(vertices[i].x != 0) {
            pcl::PointXYZRGB point(255, 255, 255);
            point.x = vertices[i].x;
            point.y = vertices[i].y;
            point.z = vertices[i].z;
            point_cloud_ptr->points.push_back(point);
            pcl::Normal n(vectors[i].x, vectors[i].y, vectors[i].z);
            cloud_vectors_ptr->push_back(n);
        }
    }
    point_cloud_ptr->width = (int) point_cloud_ptr->points.size();
    point_cloud_ptr->height = 1;
    cloud_vectors_ptr->width = (int) cloud_vectors_ptr->points.size();
    cloud_vectors_ptr->height = 1;

    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D viewer"));
    viewer->setBackgroundColor(0, 0, 0);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(point_cloud_ptr);
    viewer->addPointCloud<pcl::PointXYZRGB>(point_cloud_ptr, rgb, "sample cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
    viewer->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal>(point_cloud_ptr, cloud_vectors_ptr, 1, 0.05, "normals");
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();

    while (!viewer->wasStopped() ) {
        viewer->spinOnce(100);
        boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    }
}

void Renderer::compileShader(string src, GLenum type, GLuint &shader) {
    shader = glCreateShader(type);
    const char *c_str = src.c_str();
    glShaderSource(shader, 1, &c_str, nullptr);
    glCompileShader(shader);
}

GLint Renderer::createRenderProgram(GLuint &vertex_shader, GLuint &fragment_shader, GLuint &program,
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

bool Renderer::loadObjFile(const char *path, GLuint &vertexbuffer, vector<glm::vec3> &indexed_vertices,
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
        }
        else if (strcmp(lineHeader, "vt") == 0) {
            glm::vec2 uv;
            fscanf(file, "%f %f\n", &uv.x, &uv.y);
            uv.y = -uv.y; // Invert V coordinate since we will only use DDS texture, which are inverted. Remove if you want to use TGA or BMP loaders.
            temp_uvs.push_back(uv);
        }
        else if (strcmp(lineHeader, "vn") == 0) {
            glm::vec3 normal;
            fscanf(file, "%f %f %f\n", &normal.x, &normal.y, &normal.z);
            temp_normals.push_back(glm::normalize(normal));
        }
        else if (strcmp(lineHeader, "f") == 0) {
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

//    // calculate normals
//    pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
//    for (uint i = 0; i < temp_vertices.size(); i++) {
//        pcl::PointXYZ point(temp_vertices[i].x, temp_vertices[i].y, temp_vertices[i].z);
//        point_cloud_ptr->points.push_back(point);
//    }
//    point_cloud_ptr->width = (int) point_cloud_ptr->points.size();
//    point_cloud_ptr->height = 1;
//
//    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
//    ne.setInputCloud (point_cloud_ptr);
//    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
//    ne.setSearchMethod (tree);
//    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
//    ne.setRadiusSearch (0.005);
//    ne.compute (*cloud_normals);
//
//    for (uint i = 0; i < temp_vertices.size(); i++) {
//        glm::vec3 n(cloud_normals->at(i).normal_x,cloud_normals->at(i).normal_y,cloud_normals->at(i).normal_z);
//        temp_normals.push_back(n);
////        cout << glm::to_string(n) << endl;
//    }

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

    indexVBO_slow(vertices, uvs, normals, indices, indexed_vertices, indexed_uvs, indexed_normals);
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

bool Renderer::getSimilarVertexIndex(
        glm::vec3 & in_vertex,
        glm::vec2 & in_uv,
        glm::vec3 & in_normal,
        std::vector<glm::vec3> & out_vertices,
        std::vector<glm::vec2> & out_uvs,
        std::vector<glm::vec3> & out_normals,
        unsigned short & result
){
    // Lame linear search
    for ( unsigned int i=0; i<out_vertices.size(); i++ ){
        if (
                is_near( in_vertex.x , out_vertices[i].x ) &&
                is_near( in_vertex.y , out_vertices[i].y ) &&
                is_near( in_vertex.z , out_vertices[i].z ) //&&
//                is_near( in_uv.x     , out_uvs     [i].x ) &&
//                is_near( in_uv.y     , out_uvs     [i].y ) &&
//                is_near( in_normal.x , out_normals [i].x ) &&
//                is_near( in_normal.y , out_normals [i].y ) &&
//                is_near( in_normal.z , out_normals [i].z )
                ){
            result = i;
            return true;
        }
    }
    // No other vertex could be used instead.
    // Looks like we'll have to add it to the VBO.
    return false;
}

void Renderer::indexVBO_slow(
        std::vector<glm::vec3> & in_vertices,
        std::vector<glm::vec2> & in_uvs,
        std::vector<glm::vec3> & in_normals,

        std::vector<unsigned short> & out_indices,
        std::vector<glm::vec3> & out_vertices,
        std::vector<glm::vec2> & out_uvs,
        std::vector<glm::vec3> & out_normals
){
    // For each input vertex
    for ( unsigned int i=0; i<in_vertices.size(); i++ ){

        // Try to find a similar vertex in out_XXXX
        unsigned short index;
        bool found = getSimilarVertexIndex(in_vertices[i], in_uvs[i], in_normals[i],     out_vertices, out_uvs, out_normals, index);

        if ( found ){ // A similar vertex is already in the VBO, use it instead !
            out_indices.push_back( index );
        }else{ // If not, it needs to be added in the output data.
            out_vertices.push_back( in_vertices[i]);
            out_uvs     .push_back( in_uvs[i]);
            out_normals .push_back( in_normals[i]);
            out_indices .push_back( (unsigned short)out_vertices.size() - 1 );
        }
    }
}
