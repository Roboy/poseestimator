#include "poseestimator.hpp"
#include <SFML/Window.hpp>

int main()
{
    sf::ContextSettings settings;
    settings.depthBits = 24;
    settings.stencilBits = 8;

    sf::Window window(sf::VideoMode(WIDTH, HEIGHT, 32), "Transform Feedback", sf::Style::Titlebar | sf::Style::Close, settings);

    // Initialize GLEW
    glewExperimental = GL_TRUE;
    glewInit();

    char file_path[] = "/home/letrend/workspace/poseestimator";

    string obj = "duck";

    PoseEstimator poseEstimator(file_path,obj);

    // run the main loop
    bool running = true;
    while (running)
    {
        // handle events
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
            {
                // end the program
                running = false;
            }
            else if (event.type == sf::Event::Resized)
            {
                // adjust the viewport when the window is resized
                glViewport(0, 0, event.size.width, event.size.height);
            }
        }
        VectorXd pose(6);
        pose << 0,-0.1,-0.7,0.5,0.5,0.5;
        Mat img_camera = poseEstimator.renderColor(obj, pose);

        VectorXd pose_estimate(6);
        pose_estimate << 0,-0.1,-0.7,0,0,0;
        float lambda = 0.0000001f;
        poseEstimator.getPose(obj,img_camera, pose_estimate, lambda);

        // end the current frame (internally swaps the front and back buffers)
        window.display();
    }

    window.close();

    return 0;
}
