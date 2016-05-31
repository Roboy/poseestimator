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

class Model{
public:
    Model(const char* rootDirectory, const char* modelFile);
    ~Model();
    void render(VectorXd &pose, Mat &img);
    void render(Mat &img);
    void updateViewMatrix(sf::Window &window);
    Renderer *renderer;
private:
    vector<Mesh*> meshes;
    FileSystem *filesystem;
//    Poseestimator poseestimator(renderer.vertices[obj], renderer.normals[obj], renderer.K);
};