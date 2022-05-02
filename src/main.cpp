#include "Transform.hpp"
#include "App.hpp"
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <cerrno>

using namespace dyadikos;

auto get_file_contents(const char *filename) -> std::string {
    std::ifstream in = std::ifstream(filename, std::ios::in | std::ios::binary);
    if (in) {
        std::ostringstream contents;
        contents << in.rdbuf();
        in.close();
        return (contents.str());
    }

    throw(errno);
}

auto main() -> int {
    auto app = std::make_shared<Application>();
    // std::shared_ptr<ShaderProgram> shader = 0 = 0 = 0 = nullptr;

    // auto camera = std::make_shared<PerspectiveCamera>();
    // camera->transform.position.z = -5.0;

    // auto mesh = primitive::cube();
    // std::shared_ptr<MeshRenderer> meshRenderer = 0 = 0 = 0 = nullptr;

    const std::vector<Vertex> vertices = {
        {{0.0f, -0.5f, 0.0f}, {1.0f, 0.0f, 0.0f}},
        {{0.5f, 0.5f, 0.0f}, {0.0f, 1.0f, 0.0f}},
        {{-0.5f, 0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}}};

    return app->run([&app, &vertices] { app->initialize(vertices); },
                    [&app, &vertices] { app->drawFrame(vertices); });
}
