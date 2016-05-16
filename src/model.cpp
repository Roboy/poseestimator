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
}

Model::~Model(){
    for(uint mesh=0;mesh<meshes.size();mesh++)
        delete meshes[mesh];
    delete renderer;
    delete filesystem;
}

Mat Model::render(VectorXd &pose){
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    for(uint mesh=0; mesh<meshes.size(); mesh++){
        renderer->renderColor(meshes[mesh], pose);
    }
    return renderer->getImage();
}

Mat Model::render(){
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    for(uint mesh=0; mesh<meshes.size(); mesh++){
        renderer->renderColor(meshes[mesh]);
    }
    return renderer->getImage();
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
            float horizontalAngle = speed_rot * float(delta.x);
            float verticalAngle = speed_rot * float(delta.y);

            rot = Eigen::AngleAxisf(horizontalAngle, renderer->ViewMatrix.block<1, 3>(1, 0)) *
                  Eigen::AngleAxisf(verticalAngle, renderer->ViewMatrix.block<1, 3>(0, 0));
        }
    }

    Vector3f direction = renderer->ViewMatrix.block<1,3>(2,0);
    Vector3f right = renderer->ViewMatrix.block<1,3>(0,0);

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

    renderer->ViewMatrix = renderer->ViewMatrix*RT;
}