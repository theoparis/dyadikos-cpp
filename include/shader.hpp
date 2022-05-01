#pragma once
#include "glad/glad.h"
#include <GLFW/glfw3.h>
#include <cstdint>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <spdlog/spdlog.h>

namespace dyadikos {
    class ShaderProgram {
        public:
            ShaderProgram(const std::string &vertexShaderSource,
                          const std::string &fragmentShaderSource)
                : id(glCreateProgram()) {
                attach(vertexShaderSource, GL_VERTEX_SHADER);
                attach(fragmentShaderSource, GL_FRAGMENT_SHADER);
                glLinkProgram(id);

                // handle linking errors
                int success = 0;
                glGetProgramiv(id, GL_LINK_STATUS, &success);
                if (!success) {
                    int length = 0;
                    glGetProgramiv(id, GL_INFO_LOG_LENGTH, &length);
                    char *message = (char *)alloca(length * sizeof(char));
                    glGetProgramInfoLog(id, length, &length, message);

                    spdlog::error("Failed to link shader: {}", message);
                }
            }

            void cleanup() { glDeleteProgram(id); }

            void activate() { glUseProgram(id); }

            void set_uniform(std::string location, glm::mat4 data) {
                glUniformMatrix4fv(glGetUniformLocation(id, location.c_str()),
                                   1, GL_FALSE, glm::value_ptr(data));
            }

            void setUniform(std::string location, glm::vec3 data) {
                glUniform3fv(glGetUniformLocation(id, location.c_str()), 1,
                             glm::value_ptr(data));
            }

            void setUniform(std::string location, glm::vec4 data) {
                glUniform4fv(glGetUniformLocation(id, location.c_str()), 1,
                             glm::value_ptr(data));
            }

            void setUniform(std::string location, float data) {
                glUniform1f(glGetUniformLocation(id, location.c_str()), data);
            }

        private:
            uint32_t id;

            void attach(const std::string &shaderSource, uint32_t type) {
                auto shader = glCreateShader(type);
                auto shaderSourceC = shaderSource.c_str();

                glShaderSource(shader, 1, &shaderSourceC, nullptr);
                glCompileShader(shader);

                int success = 0;
                glGetShaderiv(shader, GL_COMPILE_STATUS, &success);

                if (!success) {
                    int length = 0;
                    glGetShaderiv(id, GL_INFO_LOG_LENGTH, &length);
                    char *message = (char *)alloca(length * sizeof(char));
                    glGetShaderInfoLog(id, length, &length, message);

                    spdlog::error("Failed to compile shader: {}", message);
                }

                glAttachShader(id, shader);
                glDeleteShader(shader);
            }
    };
} // namespace dyadikos
