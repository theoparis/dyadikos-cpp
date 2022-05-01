#pragma once
#include <vector>
#include "glm/glm.hpp"
#include "glad/glad.h"

namespace dyadikos {
    auto compute_face_normal(glm::vec3 p1, glm::vec3 p2, glm::vec3 p3)
        -> glm::vec3 {
        // Uses p2 as a new origin for p1,p3
        auto a = p3 - p2;
        auto b = p1 - p2;
        // Compute the cross product a X b to get the face normal
        return glm::normalize(glm::cross(a, b));
    }

    struct Mesh {
            std::vector<glm::vec3> vertices;
            std::vector<unsigned short> indices;
            std::vector<glm::vec3> normals;

            void calculate_normals() {
                this->normals = std::vector<glm::vec3>(this->vertices.size());
                // For each face calculate normals and append it
                // to the corresponding vertices of the face
                for (unsigned int i = 0; i < this->indices.size(); i += 3) {
                    glm::vec3 a = this->vertices[this->indices[i]];
                    glm::vec3 b = this->vertices[this->indices[i + 1LL]];
                    glm::vec3 c = this->vertices[this->indices[i + 2LL]];
                    glm::vec3 normal = compute_face_normal(a, b, c);
                    this->normals[this->indices[i]] += normal;
                    this->normals[this->indices[i + 1LL]] += normal;
                    this->normals[this->indices[i + 2LL]] += normal;
                }
                // Normalize each normal
                for (auto &normal : this->normals)
                    normal = glm::normalize(normal);
            }
    };

    struct Material {
            glm::vec4 color;
            bool wireframe;
    };

    class MeshRenderer {
        public:
            explicit MeshRenderer(Mesh mesh)
                : mesh(std::move(mesh)), vao(-1), vbo(-1) {
                glGenVertexArrays(1, &vao);
                glGenBuffers(1, &vbo);
                glGenBuffers(1, &ibo);
                glBindVertexArray(vao);

                update();

                glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,
                                      sizeof(float) * 3, nullptr);
                glEnableVertexAttribArray(0);

                glBindBuffer(GL_ARRAY_BUFFER, 0);
                glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
            }

            void draw() {
                glBindVertexArray(vao);
                glDrawElements(GL_TRIANGLES,        // mode
                               mesh.indices.size(), // count
                               GL_UNSIGNED_SHORT,   // type
                               &mesh.indices[0]);
            }

            void update() {
                this->mesh.calculate_normals();

                glBindBuffer(GL_ARRAY_BUFFER, vbo);
                glBufferData(GL_ARRAY_BUFFER,
                             mesh.vertices.size() * sizeof(glm::vec3),
                             mesh.vertices.data(), GL_STATIC_DRAW);

                glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
                glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                             mesh.indices.size() * sizeof(unsigned short),
                             mesh.indices.data(), GL_STATIC_DRAW);
            }

        private:
            Mesh mesh;
            // Vertex array
            unsigned int vao;
            // Vertex buffer
            unsigned int vbo;
            // Index buffer
            unsigned int ibo{};
    };
} // namespace dyadikos
