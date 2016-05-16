#include "model.hpp"

Model::Model(const char* rootDirectory, const char* modelFile){
    filesystem = new FileSystem(rootDirectory);

    path fullFilePath;
    if(filesystem->find(modelFile,&fullFilePath)) {
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
                        sdf::ElementPtr uriElem = linkElem->GetElement("visual")->GetElement("geometry")->GetElement(
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
                                char modelsPath[200];
                                sprintf(modelsPath, "%s/%s", rootDirectory, file);
                                Mesh *mesh = new Mesh;
                                mesh->name = linkElem->GetAttribute("name")->GetAsString();
                                meshes.push_back(mesh);
                                printf("%s:  %f %f %f %f %f %f\n", modelsPath, x, y, z, roll, pitch, yaw);
                            }
                        }
                    }
                    linkElem = linkElem->GetNextElement();
                }
                string obj = nameParam->GetAsString();
                Renderer renderer(rootDirectory, obj);
            }
        } else {
            printf("does not contain a model\n");
        }
    }else{
        cout << "could not find model file: " << modelFile << endl;
    }
}

Model::~Model(){
    for(auto mesh:meshes)
        delete mesh;
}