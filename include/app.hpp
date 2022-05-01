#pragma once
#include "glad/glad.h"
#include <glm/glm.hpp>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <cstdlib>
#include <cstdio>
#include <spdlog/spdlog.h>
#include <functional>

namespace dyadikos {
    static void error_callback(int error, const char *description) {
        spdlog::error("Error: %s\n", description);
    }

    class Application {
        private:
            GLFWwindow *window;
            glm::vec2 window_size;

        public:
            auto get_window_size() -> glm::vec2 { return window_size; }

            auto run(std::function<void()> initFn, std::function<void()> loopFn)
                -> int {
                glfwSetErrorCallback(error_callback);

                if (!glfwInit()) exit(EXIT_FAILURE);

                glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
                glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
                glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
                glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
                glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);

                window =
                    glfwCreateWindow(640, 480, "Dyadikos", nullptr, nullptr);
                if (!window) {
                    glfwTerminate();
                    return -1;
                }

                glfwMakeContextCurrent(window);
                if (!gladLoadGL()) {
                    spdlog::error("Could not initialize OpenGL!");
                    return false;
                }

                spdlog::debug("OpenGL {}.{}", GLVersion.major, GLVersion.minor);

                initFn();

                while (!glfwWindowShouldClose(window)) {
                    int width = 0;
                    int height = 0;

                    glfwGetFramebufferSize(window, &width, &height);
                    // auto ratio = (float)width / (float)height;
                    window_size = glm::vec2(width, height);

                    glViewport(0, 0, width, height);
                    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

                    loopFn();

                    glfwSwapBuffers(window);
                    glfwPollEvents();
                }

                glfwDestroyWindow(window);
                glfwTerminate();

                return 0;
            }
    };

} // namespace dyadikos
