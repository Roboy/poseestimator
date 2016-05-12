#pragma once

// image magick
#define MAGICKCORE_EXCLUDE_DEPRECATED
#include <Magick++.h>
#include <iostream>
#include <map>
#include <vector>
// glew
#include <GL/glew.h>
// assimp
#include <assimp/Importer.hpp>      // C++ importer interface
#include <assimp/scene.h>       // Output data structure
#include <assimp/postprocess.h> // Post processing flags
// Eigen
#include <Eigen/Core>
#include <Eigen/Dense>

#define INVALID_OGL_VALUE 0xffffffff
#define SAFE_DELETE(p) if (p) { delete p; p = NULL; }

using namespace Eigen;

struct Vertex {
    Vector3f m_pos;
    Vector2f m_tex;
    Vector3f m_normal;

    Vertex() { }

    Vertex(const Vector3f &pos, const Vector2f &tex, const Vector3f &normal) {
        m_pos = pos;
        m_tex = tex;
        m_normal = normal;
    }
};

class Texture {
public:
    Texture(GLenum TextureTarget, const std::string &FileName);

    bool Load();

    void Bind(GLenum TextureUnit);

private:
    std::string m_fileName;
    GLenum m_textureTarget;
    GLuint m_textureObj;
    Magick::Image m_image;
    Magick::Blob m_blob;
};

class Mesh {
public:
    Mesh();

    ~Mesh();

    bool LoadMesh(const std::string &Filename);

    void Render();

private:
    bool InitFromScene(const aiScene *pScene, const std::string &Filename);

    void InitMesh(unsigned int Index, const aiMesh *paiMesh);

    bool InitMaterials(const aiScene *pScene, const std::string &Filename);

    void Clear();

#define INVALID_MATERIAL 0xFFFFFFFF

    struct MeshEntry {
        MeshEntry();

        ~MeshEntry();

        void Init(const std::vector<Vertex> &Vertices,
                  const std::vector<unsigned int> &Indices);

        GLuint VB;
        GLuint IB;
        unsigned int NumIndices;
        unsigned int MaterialIndex;
    };

    std::vector<MeshEntry> m_Entries;
    std::vector<Texture *> m_Textures;
};

