#pragma once
#include "transform.hpp"
#include "app.hpp"
#include "glm/glm.hpp"
#include <memory>

namespace dyadikos {
    struct PerspectiveCamera {
            Transform transform;
            float fov = 45.0f;
            float near = 0.01f;
            float far = 1000.0f;

            auto get_matrix(std::shared_ptr<Application> app) const
                -> glm::mat4 {
                auto windowSize = app->get_window_size();

                return glm::perspective(fov, windowSize.x / windowSize.y, near,
                                        far) *
                       transform.get_matrix();
            }
    };
} // namespace dyadikos
