#pragma once
#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/transform.hpp>
#include <vector>

namespace dyadikos {
    enum Direction { UP, RIGHT, DOWN, LEFT, FRONT, BACK };

    auto get_direction_vector(glm::vec3 target) -> Direction {
        std::vector<glm::vec3> compass = {
            glm::vec3(0.0f, 1.0f, 0.0f),  // up
            glm::vec3(1.0f, 0.0f, 0.0f),  // right
            glm::vec3(0.0f, -1.0f, 0.0f), // down
            glm::vec3(-1.0f, 0.0f, 0.0f), // left
            glm::vec3(0.0f, 0.0f, 1.0f),  // forward
            glm::vec3(0.0f, 0.0f, -1.0f), // backward
        };
        auto max = 0.0f;
        unsigned int best_match = -1;
        for (unsigned int i = 0; i < 4; i++) {
            auto dot_product = glm::dot(glm::normalize(target), compass[i]);
            if (dot_product > max) {
                max = dot_product;
                best_match = i;
            }
        }
        return (Direction)best_match;
    }

    struct Transform {
            glm::vec3 position;
            glm::vec3 rotation;
            glm::vec3 scale;

            Transform()
                : position(glm::vec3(0.0, 0.0, 0.0)),
                  rotation(glm::vec3(0.0, 0.0, 0.0)),
                  scale(glm::vec3(1.0, 1.0, 1.0)) {}

            Transform(glm::vec3 position, glm::vec3 rotation, glm::vec3 scale)
                : position(position), rotation(rotation), scale(scale) {}

            [[nodiscard]] auto get_matrix() const -> glm::mat4 {
                auto matrix = glm::mat4(1.0f);

                matrix *= glm::translate(position);
                matrix *= glm::mat4_cast(glm::quat(rotation));
                matrix *= glm::scale(scale);

                return matrix;
            }

            [[nodiscard]] auto get_front() const -> glm::vec3 {
                return glm::rotate(glm::inverse(glm::quat(rotation)),
                                   glm::vec3(0.0, 0.0, 1.0));
            }

            [[nodiscard]] auto get_up() const -> glm::vec3 {
                return glm::rotate(glm::inverse(glm::quat(rotation)),
                                   glm::vec3(0.0, 1.0, 0.0));
            }

            [[nodiscard]] auto get_right() const -> glm::vec3 {
                return glm::rotate(glm::inverse(glm::quat(rotation)),
                                   glm::vec3(1.0, 0.0, 0.0));
            }
    };
} // namespace dyadikos
