#pragma once
#include <iostream>
#include <fstream>
#include <thread>
#include <future>
#include <sdf/parser.hh>
#include "mesh.hpp"
#include "renderer.hpp"
#include "filesystem.hpp"
#include "poseestimator.hpp"

using namespace std;

enum{ NORMALS, TANGENTS};

class Model{
public:
    Model(const char* rootDirectory, const char* modelFile);
    ~Model();
    void render(VectorXd &pose, Mat &img);
    void render(Mat &img);
    void updateViewMatrix(sf::Window &window);

    void visualize(int type = NORMALS);

    Renderer *renderer;
    Poseestimator *poseestimator;
private:
    vector<Mesh*> meshes;
    FileSystem *filesystem;
};