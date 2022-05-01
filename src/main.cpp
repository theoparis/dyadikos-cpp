#include "app.hpp"
#include "mesh.hpp"
#include "shader.hpp"
#include "transform.hpp"
#include "primitive.hpp"
#include "camera.hpp"
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <cerrno>

using namespace dyadikos;

auto get_file_contents(const char *filename) -> std::string {
    std::ifstream in(filename, std::ios::in | std::ios::binary);
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
    std::shared_ptr<ShaderProgram> shader = nullptr;

    auto camera = std::make_shared<PerspectiveCamera>();
    camera->transform.position.z = -5.0;

    auto mesh = primitive::cube();
    std::shared_ptr<MeshRenderer> meshRenderer = nullptr;

    return app->run(
        [&shader, &mesh, &meshRenderer] {
            shader = std::make_shared<ShaderProgram>(
                get_file_contents("examples/shader/default.vert"),
                get_file_contents("examples/shader/default.frag"));

            meshRenderer = std::make_shared<MeshRenderer>(mesh);
        },
        [&shader, &camera, &app, &meshRenderer] {
            glClearColor(0.2, 0.2, 0.2, 1.0);
            shader->activate();
            shader->set_uniform("transform", camera->get_matrix(app));
            meshRenderer->draw();
        });
}
