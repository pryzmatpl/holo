#include "main.hpp"

using namespace std;
using namespace cv;
namespace po = boost::program_options;

// Global variables that are still used externally.
int zoom = 1;
const int zoommax = 1000;
Mat backup__;
extern float frft_angle;  // Defined elsewhere (e.g., in one of the EIT modules)
extern float wavelength;

//------------------------------------------------------------------------------
// Structure to hold command line parameters
//------------------------------------------------------------------------------
struct Options {
    string imageFilename = "_logo.jpg";
    float distance = 1.0f;
    float wavelength = 632.8e-9;
    string hologramType = "FRESNEL";
    string filImage = "NONE";
    string filCImage = "";
    string filAfft = "";
    string filOut = "";
    string refWave = "";
    bool gerchOn = true;
    string chirp = "";
    bool genTwo = false;
    float frftAngle = static_cast<float>(CV_PI / 2); // Using OpenCV's CV_PI constant
};

//------------------------------------------------------------------------------
// Structure to encapsulate zoom trackbar data
//------------------------------------------------------------------------------
struct ZoomData {
    Mat originalImage;     // Image to reprocess on zoom events
    int zoom = 500;        // Initial zoom value
    int zoomMax = 1000;    // Maximum zoom value for the trackbar
    string filterType = "SINC"; // Filter type used in the zoom callback
};

//------------------------------------------------------------------------------
// Function: parseCommandLineOptions
// Description: Uses Boost.Program_options to parse command line arguments
//------------------------------------------------------------------------------
Options parseCommandLineOptions(int argc, char** argv) {
    Options opts;
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "Display help message")
        ("image", po::value<string>(), "Input image file")
        ("distance", po::value<float>(), "Distance from SLM to hologram")
        ("type", po::value<string>(), "Type of hologram (FRESNEL, RPN, RPNOCV, FFT, FRECONV)")
        ("gerch", po::value<bool>(), "Use Gerchberg-Saxton algorithm")
        ("refwave", po::value<string>(), "Reference wavefront type (FLAT, REF, CHIRP, RAYLEIGH)")
        ("filimage", po::value<string>(), "Filter for the real input image (e.g., REMAVG, SINC, LAPLACIAN, RPN, CLEARCENTER, TWINREM)")
        ("filcimage", po::value<string>(), "Filter for the complex input image")
        ("filafft", po::value<string>(), "Filter for the FFT of the input image")
        ("filout", po::value<string>(), "Filter for the output image")
        ("wavelength", po::value<float>(), "Wavelength (default: He-Ne)")
        ("gentwo", po::value<bool>(), "Generate two images for testing with different refwaves")
        ("frft_angle", po::value<float>(), "Fractional Fourier transform angle")
        ("chirp", po::value<string>(), "Chirp function type after FFT")
    ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        cout << desc << "\n";
        exit(0);
    }

    if (vm.count("image"))
        opts.imageFilename = vm["image"].as<string>();
    if (vm.count("distance"))
        opts.distance = vm["distance"].as<float>();
    if (vm.count("wavelength"))
        opts.wavelength = vm["wavelength"].as<float>();
    if (vm.count("type"))
        opts.hologramType = vm["type"].as<string>();
    if (vm.count("filimage"))
        opts.filImage = vm["filimage"].as<string>();
    if (vm.count("filcimage"))
        opts.filCImage = vm["filcimage"].as<string>();
    if (vm.count("filafft"))
        opts.filAfft = vm["filafft"].as<string>();
    if (vm.count("filout"))
        opts.filOut = vm["filout"].as<string>();
    if (vm.count("refwave"))
        opts.refWave = vm["refwave"].as<string>();
    if (vm.count("gerch"))
        opts.gerchOn = vm["gerch"].as<bool>();
    if (vm.count("chirp"))
        opts.chirp = vm["chirp"].as<string>();
    if (vm.count("gentwo"))
        opts.genTwo = vm["gentwo"].as<bool>();
    if (vm.count("frft_angle"))
        opts.frftAngle = vm["frft_angle"].as<float>();
    else
        opts.frftAngle = static_cast<float>(CV_PI / 2);

    return opts;
}

//------------------------------------------------------------------------------
// Callback: onZoom
// Description: Called when the trackbar position changes. Applies a filter and
//              tiles the image based on the current zoom value.
//------------------------------------------------------------------------------
void onZoom(int trackPosition, void* userdata) {
    ZoomData* zoomData = reinterpret_cast<ZoomData*>(userdata);
    zoomData->zoom = trackPosition; // Update zoom

    // (Optional) Compute a scale factor if needed:
    float scale = static_cast<float>(trackPosition) / zoomData->zoomMax;

    Mat temp;
    zoomData->originalImage.copyTo(temp);

    eit::eit_hologram eitHolo;
    temp = eitHolo.holoeye_filter(temp, {});
    temp = eitHolo.tile_to_fhd(temp);
    imshow("Output - tiled", temp);
}

//------------------------------------------------------------------------------
// Main function: Processes the image and displays the output.
//------------------------------------------------------------------------------
int main(int argc, char** argv) {
    // Parse command line parameters.
    Options opts = parseCommandLineOptions(argc, argv);

    // Update external/global parameters if needed by other modules.
    glob_distance = opts.distance;
    wavelength = opts.wavelength;
    frft_angle = opts.frftAngle;

    // Create a window for output.
    namedWindow("Output - tiled", WINDOW_NORMAL);

    // Create instances of processing classes.
    eit::eit_hologram eitHolo;
    eit::gerch Gerch;  // Instance for Gerchberg-Saxton algorithm

    //--------------------------------------------------------------------------
    // Step 0: Load and prepare the input image.
    //--------------------------------------------------------------------------
    Mat g_ROI = imread(opts.imageFilename, IMREAD_GRAYSCALE);
    if (g_ROI.empty()) {
        cerr << "Error: Could not open or find the image: " << opts.imageFilename << endl;
        return -1;
    }
    eit::typetest(g_ROI); // Debug function (assumed to be defined in main.hpp)

    Mat padded;
    eitHolo.set_optimal_holosize(g_ROI, padded);
    eit::typetest(padded);

    // Copy the padded image for later processing.
    Mat intensity;
    padded.copyTo(intensity);

    //--------------------------------------------------------------------------
    // Step 1: Filter the real-valued input image.
    //--------------------------------------------------------------------------
    eitHolo.holoeye_filter(padded, opts.filImage);

    //--------------------------------------------------------------------------
    // Step 2: Add a reference wave to the input.
    //--------------------------------------------------------------------------
    eitHolo.holoeye_reference(padded, opts.refWave);

    //--------------------------------------------------------------------------
    // Step 3: Filter the complex image (with phase-only reference).
    //--------------------------------------------------------------------------
    eitHolo.holoeye_filter(padded, opts.filCImage);

    //--------------------------------------------------------------------------
    // Step 4: Perform the transform.
    //--------------------------------------------------------------------------
    if (opts.gerchOn) {
        Gerch(padded, intensity);
    } else {
        Mat adft_data;
        eitHolo.holoeye_transform(padded, adft_data, opts.hologramType);
    }

    //--------------------------------------------------------------------------
    // Step 5: Filter the FFT of the image.
    //--------------------------------------------------------------------------
    eitHolo.holoeye_filter(padded, opts.filAfft);

    //--------------------------------------------------------------------------
    // Step 6: Tile the phase image to full HD resolution.
    //--------------------------------------------------------------------------
    Mat backup;
    padded.copyTo(backup);
    Mat finalres = eitHolo.tile_to_fhd(padded);

    //--------------------------------------------------------------------------
    // Step 7: Filter the tiled image.
    //--------------------------------------------------------------------------
    eitHolo.holoeye_filter(finalres, opts.filOut);

    // Set up the zoom trackbar.
    ZoomData zoomData;
    finalres.copyTo(zoomData.originalImage);
    zoomData.zoom = 500;
    zoomData.zoomMax = 1000;
    zoomData.filterType = "SINC";  // Can be parameterized if desired

    createTrackbar("Zoom via sinc", "Output - tiled", &zoomData.zoom, zoomData.zoomMax, onZoom, &zoomData);

    //--------------------------------------------------------------------------
    // Display and save the final output.
    //--------------------------------------------------------------------------
    if (opts.gerchOn) {
        Mat final = eitHolo.tile_to_fhd(intensity);
        imshow("Output - tiled", final);
        imwrite("./latest-GS.jpg", final);
    } else {
        imshow("Output - tiled", finalres);
        imwrite("./latest.jpg", finalres);
    }

    if (opts.genTwo) {
        Mat temp;
        normalize(finalres, temp, 0, 255, NORM_MINMAX);
        imwrite("./holobo-1.jpg", temp);
    }

    cout << "Wavelength: " << wavelength << "\n";

    //--------------------------------------------------------------------------
    // Event loop: allow toggling fullscreen and exit on ESC.
    //--------------------------------------------------------------------------
    bool tiledFullScreen = false;
    while (true) {
        int key = waitKey(30);
        if (key == 27) { // ESC key: exit
            break;
        } else if (key == 102) { // 'f' key: toggle fullscreen
            if (!tiledFullScreen) {
                setWindowProperty("Output - tiled", WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);
                finalres = eitHolo.tile_to_fhd(backup);
                tiledFullScreen = true;
            } else {
                finalres = eitHolo.tile_to_fhd(backup);
                tiledFullScreen = false;
                setWindowProperty("Output - tiled", WND_PROP_FULLSCREEN, WINDOW_NORMAL);
            }
        }
        // Placeholders for additional key commands:
        else if (key == 103) { /* Additional functionality for key 'g' can go here. */ }
        else if (key == 97)  { /* Additional functionality for key 'a' can go here. */ }
    }

    return 0;
}
