#pragma once
#include <iostream>
#include <fstream>
#include <thread>
#include <future>
#include <sdf/parser.hh>
#include "mesh.hpp"
#include "renderer.hpp"
#include "filesystem.hpp"

using namespace std;

class Model{
public:
    Model(const char* rootDirectory, const char* modelFile);
    ~Model();
    Mat render(VectorXd &pose);
    Mat render();
    void updateViewMatrix(sf::Window &window);
    Renderer *renderer;
private:
    vector<Mesh*> meshes;
    FileSystem *filesystem;
};