#include "model.hpp"

Model::Model(const char* rootDirectory, const char* modelFile) {
    filesystem = new FileSystem(rootDirectory);

    renderer = new Renderer(rootDirectory);

    path fullFilePath;
    if (filesystem->find(modelFile, &fullFilePath)) {
        if (strcmp(fullFilePath.extension().c_str(), ".sdf") == 0) {
            std::ifstream stream(fullFilePath.c_str());
            std::string str((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
            sdf::SDF sdf;
            sdf.SetFromString(str);

            // Verify correct parsing
            if (sdf.Root()->HasElement("model")) {
                sdf::ElementPtr modelElem = sdf.Root()->GetElement("model");
                // Read name attribute value
                if (modelElem->HasAttribute("name")) {
                    sdf::ParamPtr nameParam = modelElem->GetAttribute("name");
                    cout << "loading " << nameParam->GetAsString() << endl;
                    sdf::ElementPtr linkElem = modelElem->GetElement("link");
                    while (linkElem != NULL) {
                        if (linkElem->HasElement("visual")) {
                            sdf::ElementPtr uriElem = linkElem->GetElement("visual")->GetElement(
                                    "geometry")->GetElement(
                                    "mesh")->GetElement("uri");
                            sdf::ElementPtr poseElem = linkElem->GetElement("pose");
                            if (poseElem != NULL && uriElem != NULL &&
                                strcmp(uriElem->GetValue()->GetAsString().c_str(), "__default__") != 0) {
                                char file[200];
                                sprintf(file, "%s", uriElem->GetValue()->GetAsString().c_str() + 8);
                                float x, y, z, roll, pitch, yaw;
                                if (sscanf(poseElem->GetValue()->GetAsString().c_str(), "%f %f %f %f %f %f", &x, &y, &z,
                                           &roll,
                                           &pitch, &yaw) != 6)
                                    printf("error reading pose parameters\n");
                                else {
                                    char modelPath[200];
                                    sprintf(modelPath, "%smodels/%s", rootDirectory, file);
                                    path p(modelPath);
                                    if(exists(p) && is_regular_file(p)) {
                                        Mesh *mesh = new Mesh;
                                        mesh->name = linkElem->GetAttribute("name")->GetAsString();
                                        VectorXf pose(6);
                                        pose << x, y, z, roll, pitch, yaw;
                                        mesh->LoadMesh(modelPath, pose);
                                        meshes.push_back(mesh);

                                        printf("%s:  %f %f %f %f %f %f\n", modelPath, x, y, z, roll, pitch, yaw);
                                    }else{
                                        printf("error loading modelfile %s\n", modelPath);
                                    }
                                }
                            }
                        }
                        linkElem = linkElem->GetNextElement();
                    }
                }
            } else {
                printf("does not contain a model\n");
            }
        }else if(strcmp(fullFilePath.extension().c_str(), ".dae") == 0){
            Mesh *mesh = new Mesh;
            mesh->name = fullFilePath.stem().string();
            mesh->LoadMesh(fullFilePath.c_str());
            meshes.push_back(mesh);
        }else{
            cout << "unknown model format, please provide either sdf or dae file" << endl;
        }
    }else {
        cout << "could not find model file: " << modelFile << endl;
    }

    cout << "initializing poseestimator" << endl;
    poseestimator = new Poseestimator(meshes,renderer->K);
}

Model::~Model(){
    for(uint mesh=0;mesh<meshes.size();mesh++)
        delete meshes[mesh];
    delete renderer;
    delete filesystem;
}

void Model::render(VectorXd &pose, Mat &img){
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    for(uint mesh=0; mesh<meshes.size(); mesh++){
        renderer->renderColor(meshes[mesh], pose);
    }
    return renderer->getImage(img);
}

void Model::render(Mat &img){
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    for(uint mesh=0; mesh<meshes.size(); mesh++){
        renderer->renderColor(meshes[mesh]);
    }
    return renderer->getImage(img);
}

void Model::updateViewMatrix(sf::Window &window){
    float speed_trans = 0.01f , speed_rot = 0.001f;

    // Get mouse position
    sf::Vector2i windowsize = sf::Vector2i(window.getSize().x, window.getSize().y);
    double xpos, ypos;
    Matrix3f rot = Matrix3f::Identity();
    static bool sticky = false;
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::Space)){
        sticky = !sticky;
    }

    if(sticky) {
        sf::Vector2i mousepos = sf::Mouse::getPosition(window);
        sf::Vector2i delta = windowsize/2-mousepos;
        if(delta.x != 0 || delta.y != 0)
        {
            // set cursor to window center
            sf::Mouse::setPosition(windowsize/2, window);
            // Compute new orientation
            float horizontalAngle = -speed_rot * float(delta.x);
            float verticalAngle = -speed_rot * float(delta.y);

            rot = Eigen::AngleAxisf(horizontalAngle, Vector3f::UnitY()) *
                  Eigen::AngleAxisf(verticalAngle, Vector3f::UnitX());
        }
    }

    Vector3f direction = Vector3f::UnitZ();
    Vector3f right = Vector3f::UnitX();

    Vector3f dcameraPos(0,0,0);
    // Move forward
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::W)){
        dcameraPos += direction  * speed_trans;
    }
    // Move backward
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::S)){
        dcameraPos -= direction * speed_trans;
    }
    // Strafe right
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::D)){
        dcameraPos -= right * speed_trans;
    }
    // Strafe left
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::A)){
        dcameraPos += right * speed_trans;
    }

    Matrix4f RT = Matrix4f::Identity();
    RT.topLeftCorner(3,3) = rot;
    RT.topRightCorner(3,1) = dcameraPos;

    renderer->ViewMatrix = RT*renderer->ViewMatrix;
}

void Model::visualize(int type) {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr mesh_cloud_ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::Normal>::Ptr cloud_vectors_ptr(new pcl::PointCloud<pcl::Normal>);

    for(uint m=0; m<meshes.size(); m++){
        for(uint i=0;i<meshes[m]->m_Entries.size();i++) {
            for (uint j = 0; j < meshes[m]->m_Entries[i].NumVertices; j++) {
                pcl::PointXYZRGB point(100, 100, 100);
                point.x = poseestimator->modelData[m]->vertices_out[i][j].x;
                point.y = poseestimator->modelData[m]->vertices_out[i][j].y;
                point.z = poseestimator->modelData[m]->vertices_out[i][j].z;
                mesh_cloud_ptr->points.push_back(point);
                if(poseestimator->modelData[m]->normals_out[i][j].x!=0 &&
                        poseestimator->modelData[m]->normals_out[i][j].y!=0 &&
                        poseestimator->modelData[m]->normals_out[i][j].z!=0) {
                    pcl::PointXYZRGB point(0, 255, 0);
                    point.x = poseestimator->modelData[m]->vertices_out[i][j].x;
                    point.y = poseestimator->modelData[m]->vertices_out[i][j].y;
                    point.z = poseestimator->modelData[m]->vertices_out[i][j].z;
                    point_cloud_ptr->points.push_back(point);
                    switch (type) {
                        case NORMALS: {
                            pcl::Normal n(poseestimator->modelData[m]->normals_out[i][j].x,
                                          poseestimator->modelData[m]->normals_out[i][j].y,
                                          poseestimator->modelData[m]->normals_out[i][j].z);
                            cloud_vectors_ptr->push_back(n);
                            break;
                        }
                        case TANGENTS: {
                            pcl::Normal n(poseestimator->modelData[m]->tangents_out[i][j].x,
                                          poseestimator->modelData[m]->tangents_out[i][j].y,
                                          poseestimator->modelData[m]->tangents_out[i][j].z);
                            cloud_vectors_ptr->push_back(n);
                            break;
                        }
                    }
                }
            }
        }
    }
    mesh_cloud_ptr->width = (int) mesh_cloud_ptr->points.size();
    mesh_cloud_ptr->height = 1;
    point_cloud_ptr->width = (int) point_cloud_ptr->points.size();
    point_cloud_ptr->height = 1;
    cloud_vectors_ptr->width = (int) cloud_vectors_ptr->points.size();
    cloud_vectors_ptr->height = 1;

    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D viewer"));
    viewer->setBackgroundColor(0, 0, 0);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgbMesh(mesh_cloud_ptr);
    viewer->addPointCloud<pcl::PointXYZRGB>(mesh_cloud_ptr, rgbMesh, "mesh cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "mesh cloud");
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(point_cloud_ptr);
    viewer->addPointCloud<pcl::PointXYZRGB>(point_cloud_ptr, rgb, "sample cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
    viewer->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal>(point_cloud_ptr, cloud_vectors_ptr, 1, 0.05, "normals");
    viewer->initCameraParameters();

    viewer->setCameraPosition(0,0,0,0,0,-1,0,1,0);

    while (!viewer->wasStopped() ) {
        viewer->spinOnce(100);
        boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    }
}