#include "model.hpp"
#include <SFML/Window.hpp>

int main()
{
    sf::ContextSettings settings;
    settings.depthBits = 24;
    settings.stencilBits = 8;

    sf::Window window(sf::VideoMode(WIDTH, HEIGHT, 24), "Transform Feedback", sf::Style::Titlebar | sf::Style::Close, settings);

    // Initialize GLEW
    glewExperimental = GL_TRUE;
    glewInit();

    Model model("/home/letrend/workspace/poseestimator/","model_simplified2.sdf");

//    cv::namedWindow("camera image");
//    cv::moveWindow("camera image", 1000,0);
//
//    // run the main loop
    bool running = true;
    VectorXd pose_estimator(6);
    pose_estimator << 0,0,-1,0,0,0;

    char k;

    while (running && k!=32)
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

        static uint count = 0;

        VectorXd pose(6),grad(6);
        pose << 0,0,-1,degreesToRadians(0),degreesToRadians(0),degreesToRadians(20);
        Mat img_camera;
        model.render(pose, img_camera);

        cout << "press ENTER to run tracking, press SPACE to toggle first person view (use WASD-keys to move around)" << endl;
        while(!sf::Keyboard::isKeyPressed(sf::Keyboard::Return)) {
            model.updateViewMatrix(window);
            model.render(img_camera);
            window.display();
        }

        float lambda_trans = 0, lambda_rot = 0.0000001;
        uint iter = 0;
        while(iter<1000 && k!=32){
            Mat img_artificial;
            model.render(pose_estimator, img_artificial);
            imshow("artificial image", img_artificial);

            model.poseestimator->iterateOnce(img_camera, img_artificial, pose_estimator, grad);
            pose_estimator(0) += lambda_trans*grad(0);
            pose_estimator(1) += lambda_trans*grad(1);
            pose_estimator(2) += lambda_trans*grad(2);
            pose_estimator(3) += lambda_rot*grad(3);
            pose_estimator(4) += lambda_rot*grad(4);
            pose_estimator(5) += lambda_rot*grad(5);

            iter++;

            k = cv::waitKey(1);

//            renderer.visualize(poseestimator.vertices_out, poseestimator.normals_out, poseestimator.numberOfVertices);
//            renderer.visualize(poseestimator.vertices_out, poseestimator.tangents_out, poseestimator.numberOfVertices);
        }

        // end the current frame (internally swaps the front and back buffers)
        window.display();
    }

    window.close();

    return 0;
}
