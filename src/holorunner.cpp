#include "../main.hpp"
#include <boost/program_options.hpp>
#include <boost/tokenizer.hpp>
#include <fstream>
#include <algorithm>
#include <iostream>
#include <vector>
#include <string>

namespace po = boost::program_options;
typedef boost::char_separator<char> separator_type;

// Global variables (if required by the application)
float glob_distance;
int zoom;
const int zoommax = 100;
bool flipzero;

//------------------------------------------------------------------------------
// Function: get_file_contents
// Reads the entire file and removes all whitespace characters.
//------------------------------------------------------------------------------
std::string get_file_contents(const char* filename) {
    std::string contents, temp;
    std::fstream input(filename, std::ios::in);
    if (input.is_open()) {
        while (std::getline(input, temp)) {
            temp.erase(std::remove_if(temp.begin(),
                                        temp.end(),
                                        [](char c) {
                                            return (c == '\r' || c == '\t' || c == ' ' || c == '\n');
                                        }),
                       temp.end());
            contents += temp;
        }
    }
    return contents;
}

//------------------------------------------------------------------------------
// Main function
//------------------------------------------------------------------------------
int main(int argc, char** argv) {
    // Parse command-line options using Boost.Program_options.
    po::options_description desc("Options:");
    desc.add_options()
        ("help", "Display help message")
        ("maskfile", po::value<std::string>(), "File with the pixel list")
        ("frequency", po::value<float>(), "How many times to switch the images per second");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    // If help is requested, display options and exit.
    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 0;
    }

    // Retrieve maskfile (required).
    std::string maskfile;
    if (vm.count("maskfile")) {
        maskfile = vm["maskfile"].as<std::string>();
    } else {
        std::cerr << "Error: No maskfile provided." << std::endl;
        return 1;
    }

    // Retrieve frequency; default to 10 if not provided.
    float frequency = 10.0f;
    if (vm.count("frequency")) {
        frequency = vm["frequency"].as<float>();
    }

    // Create a window for displaying alternating images.
    cv::namedWindow("Alternating", cv::WINDOW_NORMAL);

    // Read the maskfile and tokenize the pixel (image file) list.
    std::string pixel_setup = get_file_contents(maskfile.c_str());
    boost::tokenizer<separator_type> tokens(pixel_setup, separator_type(";"));

    // Load images from the file list.
    std::vector<cv::Mat> images;
    for (const auto& token : tokens) {
        cv::Mat input_image = cv::imread(token, cv::IMREAD_COLOR);
        if (input_image.empty()) {
            std::cerr << "Warning: Could not read image: " << token << std::endl;
            continue;
        }
        images.push_back(input_image);
    }
    if (images.empty()) {
        std::cerr << "Error: No images loaded. Exiting." << std::endl;
        return 1;
    }

    // Show the first image.
    cv::imshow("Alternating", images[0]);

    // Calculate delay (in milliseconds) from frequency.
    int ms_delay = static_cast<int>(1000.0f / frequency);
    std::cout << "Delay (ms): " << ms_delay << std::endl;

    bool tiled_fs = false; // Fullscreen toggle flag.
    int idx = 0;
    int images_sz = static_cast<int>(images.size());

    // Main loop: alternate through images.
    while (true) {
        int key = cv::waitKey(ms_delay);
        try {
            idx = (idx + 1) % images_sz;
            cv::imshow("Alternating", images[idx]);
        } catch (const std::exception &e) {
            std::cerr << "Exception: " << e.what() << std::endl;
        }
        // Toggle fullscreen if the 'f' key (ASCII 102) is pressed.
        if (key == 102) {
            if (!tiled_fs) {
                cv::setWindowProperty("Alternating", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
                tiled_fs = true;
            } else {
                cv::setWindowProperty("Alternating", cv::WND_PROP_FULLSCREEN, cv::WINDOW_NORMAL);
                tiled_fs = false;
            }
        }
        // Optionally, exit on ESC key.
        if (key == 27) { // ESC
            break;
        }
    }
    return 0;
}
