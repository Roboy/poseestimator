#include "poseestimator.hpp"

// cuda error checking
string prev_file = "";
int prev_line = 0;

__constant__ float c_K[9];
__constant__ float c_modelPose[16];
__constant__ float c_cameraPose[16];

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

Poseestimator::Poseestimator(vector<Mesh*> meshes, Matrix3f &K){
    // initialize cuda
    cudaDeviceSynchronize();
    CUDA_CHECK;

    // register the VBOs in cuda
    for(uint i=0;i<meshes.size();i++){
        ModelData *m = new ModelData;
        // pass a pointer to the ModelMatrix
        m->ModelMatrix = &meshes[i]->ModelMatrix;

        m->cuda_vbo_resource.resize(meshes[i]->m_Entries.size());
        m->numberOfVertices.resize(meshes[i]->m_Entries.size());

        // host
        m->vertices_out.resize(meshes[i]->m_Entries.size());
        m->normals_out.resize(meshes[i]->m_Entries.size());
        m->tangents_out.resize(meshes[i]->m_Entries.size());
        m->gradTrans.resize(meshes[i]->m_Entries.size());
        m->gradRot.resize(meshes[i]->m_Entries.size());
        // device
        m->d_vertices_out.resize(meshes[i]->m_Entries.size());
        m->d_normals_out.resize(meshes[i]->m_Entries.size());
        m->d_tangents_out.resize(meshes[i]->m_Entries.size());
        m->d_gradTrans.resize(meshes[i]->m_Entries.size());
        m->d_gradRot.resize(meshes[i]->m_Entries.size());

        for(uint j=0;j<meshes[i]->m_Entries.size();j++) {
            // Register the OpenGL buffer objects in cuda
            cudaGraphicsGLRegisterBuffer(&m->cuda_vbo_resource[j], meshes[i]->m_Entries[j].VB, cudaGraphicsMapFlagsReadOnly );
            CUDA_CHECK;
            // how many vertices
            m->numberOfVertices[j] = meshes[i]->m_Entries[j].NumVertices;
            // allocate memory on host
            m->vertices_out[j] = new float3[m->numberOfVertices[j]];
            m->normals_out[j] = new float3[m->numberOfVertices[j]];
            m->tangents_out[j] = new float3[m->numberOfVertices[j]];
            m->gradTrans[j] = new float3[m->numberOfVertices[j]];
            m->gradRot[j] = new float3[m->numberOfVertices[j]];
            // allocate memory on gpu
            cudaMalloc(&m->d_vertices_out[j], m->numberOfVertices[j] * sizeof(float3));
            CUDA_CHECK;
            cudaMalloc(&m->d_normals_out[j], m->numberOfVertices[j] * sizeof(float3));
            CUDA_CHECK;
            cudaMalloc(&m->d_tangents_out[j], m->numberOfVertices[j] * sizeof(float3));
            CUDA_CHECK;
            cudaMalloc(&m->d_gradTrans[j], m->numberOfVertices[j] * sizeof(float3));
            CUDA_CHECK;
            cudaMalloc(&m->d_gradRot[j], m->numberOfVertices[j] * sizeof(float3));
            CUDA_CHECK;
            cout << "number of vertices: " << m->numberOfVertices[j] << endl;
        }
        modelData.push_back(m);
    }

    cudaMalloc(&d_border, WIDTH * HEIGHT * sizeof(uchar));
    CUDA_CHECK;
    cudaMalloc(&d_img_out, WIDTH * HEIGHT * sizeof(uchar));
    CUDA_CHECK;
    cudaMalloc(&d_image, WIDTH * HEIGHT * sizeof(uchar));
    CUDA_CHECK;
    res = new uchar[WIDTH * HEIGHT];

    // copy camera matrices to gpu
    cudaMemcpyToSymbol(c_K, &K(0, 0), 9 * sizeof(float));
}

Poseestimator::~Poseestimator() {
    cudaFree(d_border);
    CUDA_CHECK;
    cudaFree(d_img_out);
    CUDA_CHECK;
    cudaFree(d_image);
    CUDA_CHECK;

    delete[] res;

    for(auto m:modelData)
        delete m;
}

__global__ void costFcn(Vertex *vertices, float3 *vertices_out, float3 *normals_out,
                        float3 *tangents_out, uchar *border, uchar *image, float mu_in, float mu_out, float sigma_in,
                        float sigma_out, uchar *img_out, int numberOfVertices, float3 *gradTrans, float3 *gradRot) {
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

        float3 v = vertices[idx].m_pos;
        float3 n = vertices[idx].m_normal;

        // calculate position of vertex in camera coordinate system
        float3 pos, posModel;
        posModel.x = 0.0f;
        posModel.y = 0.0f;
        posModel.z = 0.0f;

        // modelPose
        // x
        posModel.x += c_modelPose[0 + 4 * 0] * v.x;
        posModel.x += c_modelPose[0 + 4 * 1] * v.y;
        posModel.x += c_modelPose[0 + 4 * 2] * v.z;
        posModel.x += c_modelPose[0 + 4 * 3];

        // y
        posModel.y += c_modelPose[1 + 4 * 0] * v.x;
        posModel.y += c_modelPose[1 + 4 * 1] * v.y;
        posModel.y += c_modelPose[1 + 4 * 2] * v.z;
        posModel.y += c_modelPose[1 + 4 * 3];

        // z
        posModel.z += c_modelPose[2 + 4 * 0] * v.x;
        posModel.z += c_modelPose[2 + 4 * 1] * v.y;
        posModel.z += c_modelPose[2 + 4 * 2] * v.z;
        posModel.z += c_modelPose[2 + 4 * 3];

        // cameraPose
        pos.x = 0.0f;
        pos.y = 0.0f;
        pos.z = 0.0f;
        // x
        pos.x += c_cameraPose[0 + 4 * 0] * posModel.x;
        pos.x += c_cameraPose[0 + 4 * 1] * posModel.y;
        pos.x += c_cameraPose[0 + 4 * 2] * posModel.z;
        pos.x += c_cameraPose[0 + 4 * 3];

        // y
        pos.y += c_cameraPose[1 + 4 * 0] * posModel.x;
        pos.y += c_cameraPose[1 + 4 * 1] * posModel.y;
        pos.y += c_cameraPose[1 + 4 * 2] * posModel.z;
        pos.y += c_cameraPose[1 + 4 * 3];

        // z
        pos.z += c_cameraPose[2 + 4 * 0] * posModel.x;
        pos.z += c_cameraPose[2 + 4 * 1] * posModel.y;
        pos.z += c_cameraPose[2 + 4 * 2] * posModel.z;
        pos.z += c_cameraPose[2 + 4 * 3];

        float posNorm = sqrtf(pos.x * pos.x + pos.y * pos.y + pos.z * pos.z);

        vertices_out[idx] = pos;

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

        tangents_out[idx] = cross;

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
                        ((float) image[pixelCoord.y * WIDTH + pixelCoord.x] - mu_out)) / sigma_out;
            float R = (((float) image[pixelCoord.y * WIDTH + pixelCoord.x] - mu_in) *
                       ((float) image[pixelCoord.y * WIDTH + pixelCoord.x] - mu_in)) / sigma_in;
            float statistics = (logf(sigma_out / sigma_in) + Rc - R) * dCnorm;//
            gradTrans[idx].x = statistics * normal.x;
            gradTrans[idx].y = statistics * normal.y;
            gradTrans[idx].z = statistics * normal.z;

            float Om[9] = {0, pos.z, -pos.y,
                           -pos.z, 0, pos.x,
                           pos.y, -pos.x, 0};
            float M[9] = {0, 0, 0,
                          0, 0, 0,
                          0, 0, 0};

            for (uint i = 0; i < 3; i++)
                for (uint j = 0; j < 3; j++)
                    for (uint k = 0; k < 3; k++)
                        M[i + 3 * j] += c_cameraPose[i + 4 * k] * Om[k + 3 * j];
            statistics *= posNorm / (pos.z * pos.z * pos.z);
            gradRot[idx].x = statistics * (M[0 + 3 * 0] * normal.x + M[1 + 3 * 0] * normal.y + M[2 + 3 * 0] * normal.z);
            gradRot[idx].y = statistics * (M[0 + 3 * 1] * normal.x + M[1 + 3 * 1] * normal.y + M[2 + 3 * 1] * normal.z);
            gradRot[idx].z = statistics * (M[0 + 3 * 2] * normal.x + M[1 + 3 * 2] * normal.y + M[2 + 3 * 2] * normal.z);
        }
        else {
            vertices_out[idx].x = 0;
            vertices_out[idx].y = 0;
            vertices_out[idx].z = 0;
        }
    }
}

double Poseestimator::iterateOnce(Mat img_camera, Mat img_artificial, VectorXd &pose, VectorXd &grad) {
    Mat img_camera_gray, img_camera_copy, img_artificial_gray, img_artificial_gray2;
    VectorXd initial_pose = pose;

    img_camera.copyTo(img_camera_copy);
    cvtColor(img_camera, img_camera_gray, CV_BGR2GRAY);
    cvtColor(img_artificial, img_artificial_gray, CV_BGR2GRAY);
    // make a copy
    img_artificial_gray.copyTo(img_artificial_gray2);

    vector<vector<cv::Point> > contours;
    vector<cv::Vec4i> hierarchy;
    findContours(img_artificial_gray, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_TC89_L1,
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
            drawContours(img_camera_copy, contours, idx, cv::Scalar(0, 255, 0), 1, 8, hierarchy, 0, cv::Point());
        }
        imshow("camera image", img_camera_copy);

        Mat R_mask = Mat::zeros(HEIGHT, WIDTH, CV_8UC1), Rc_mask,
                R = Mat::zeros(HEIGHT, WIDTH, CV_8UC1),
                Rc = Mat::zeros(HEIGHT, WIDTH, CV_8UC1);
        fillPoly(R_mask, contours, 255);
        bitwise_not(R_mask, Rc_mask);

        // this will mask out the respective part of the webcam image
        bitwise_and(img_artificial_gray2, R_mask, R);
        bitwise_and(img_artificial_gray2, Rc_mask, Rc);

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

        Matrix4f ViewMatrix = Matrix4f::Identity();
        ViewMatrix.topLeftCorner(3, 3) = rot;
        ViewMatrix.topRightCorner(3, 1) << pose(0), pose(1), pose(2);

        Eigen::Matrix4f cameraPose = ViewMatrix;

        cudaMemcpy(d_border, border.data, WIDTH * HEIGHT * sizeof(uchar), cudaMemcpyHostToDevice);
        CUDA_CHECK;
        cudaMemcpy(d_image, img_camera_gray.data, WIDTH * HEIGHT * sizeof(uchar), cudaMemcpyHostToDevice);
        CUDA_CHECK;
        // set result image an the gradients to zero
        cudaMemset(d_img_out, 0, WIDTH * HEIGHT * sizeof(uchar));
        CUDA_CHECK;
        // set constants on gpu
        cudaMemcpyToSymbol(c_cameraPose, &cameraPose(0, 0), 16 * sizeof(float));

        grad << 0, 0, 0, 0, 0, 0;

        for(uint i=0;i<modelData.size();i++) {
            for(uint j=0;j<modelData[i]->cuda_vbo_resource.size();j++) {
                // set modelPose on gpu
                cudaMemcpyToSymbol(c_modelPose, &(*modelData[i]->ModelMatrix)(0,0), 16 * sizeof(float));

                dim3 block = dim3(1, 1, 1);
                dim3 grid = dim3(modelData[i]->numberOfVertices[j] / block.x, 1, 1);

                // map OpenGL buffer object for writing from CUDA
                Vertex *vertices;
                cudaGraphicsMapResources(1, &modelData[i]->cuda_vbo_resource[j], 0);
                CUDA_CHECK;
                size_t num_bytes;
                cudaGraphicsResourceGetMappedPointer((void **)&vertices, &num_bytes, modelData[i]->cuda_vbo_resource[j]);
                CUDA_CHECK;

                costFcn <<< grid, block >>> ( vertices, modelData[i]->d_vertices_out[j], modelData[i]->d_normals_out[j], modelData[i]->d_tangents_out[j],
                                              d_border, d_image, mu_in, mu_out, sigma_in, sigma_out, d_img_out, modelData[i]->numberOfVertices[j],
                                              modelData[i]->d_gradTrans[j], modelData[i]->d_gradRot[j]);
                CUDA_CHECK;

                // unmap buffer object
                cudaGraphicsUnmapResources(1, &modelData[i]->cuda_vbo_resource[j], 0);
                CUDA_CHECK;

                cudaMemcpy( modelData[i]->vertices_out[j],  modelData[i]->d_vertices_out[j], modelData[i]->numberOfVertices[j] * sizeof(float3), cudaMemcpyDeviceToHost);
                CUDA_CHECK;
                cudaMemcpy( modelData[i]->normals_out[j],  modelData[i]->d_normals_out[j], modelData[i]->numberOfVertices[j] * sizeof(float3), cudaMemcpyDeviceToHost);
                CUDA_CHECK;
                cudaMemcpy( modelData[i]->tangents_out[j],  modelData[i]->d_tangents_out[j], modelData[i]->numberOfVertices[j] * sizeof(float3), cudaMemcpyDeviceToHost);
                CUDA_CHECK;
                cudaMemcpy( modelData[i]->gradTrans[j],  modelData[i]->d_gradTrans[j], modelData[i]->numberOfVertices[j] * sizeof(float3), cudaMemcpyDeviceToHost);
                CUDA_CHECK;
                cudaMemcpy( modelData[i]->gradRot[j],  modelData[i]->d_gradRot[j], modelData[i]->numberOfVertices[j] * sizeof(float3), cudaMemcpyDeviceToHost);
                CUDA_CHECK;

                for (uint k = 0; k <  modelData[i]->numberOfVertices[j]; k++) {
//                cout << "v: " << vertices_out[i].x << " " << vertices_out[i].y << " " << vertices_out[i].z << endl;
//                cout << "n: " << normals_out[i].x << " " << normals_out[i].y << " " << normals_out[i].z << endl;
//                cout << "g: " << gradTrans[i].x << " " << gradTrans[i].y << " " << gradTrans[i].z << endl;
//                cout << "g: " << gradRot[i].x << " " << gradRot[i].y << " " << gradRot[i].z << endl;
//                Vector3f n(normals_out[i].x, normals_out[i].y, normals_out[i].z);
//                cout << n.norm() << endl;
                    grad(0) += modelData[i]->gradTrans[j][k].x;
                    grad(1) += modelData[i]->gradTrans[j][k].y;
                    grad(2) += modelData[i]->gradTrans[j][k].z;
                    grad(3) += modelData[i]->gradRot[j][k].x;
                    grad(4) += modelData[i]->gradRot[j][k].y;
                    grad(5) += modelData[i]->gradRot[j][k].z;
                }
            }
        }

        // copy data from gpu to cpu
        cudaMemcpy(res, d_img_out, WIDTH * HEIGHT * sizeof(uchar), cudaMemcpyDeviceToHost);
        CUDA_CHECK;

        Mat img(HEIGHT, WIDTH, CV_8UC1, res);
        imshow("result", img);

        for (int idx = 0; idx < contours.size(); idx++) {
            drawContours(img_camera_gray, contours, idx, 255, 2, 8, hierarchy, 0, cv::Point());
        }
        imshow("img_camera_gray", img_camera_gray);
        cv::waitKey(1);
        return energy;
    } else {
        cout << "cannot find any contour" << endl;
        return 0;
    }
}