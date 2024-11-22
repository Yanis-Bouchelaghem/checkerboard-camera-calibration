#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <vector>
#include <iostream>
#include <filesystem>

// Function to get all .jpg image paths from a folder
std::vector<std::string> getJpgImagesFromFolder(const std::string& folderPath) {
    std::vector<std::string> jpgImages;

    // Check if the provided folder path exists and is a directory
    if (!std::filesystem::exists(folderPath) || !std::filesystem::is_directory(folderPath)) {
        std::cerr << "Invalid folder path: " << folderPath << std::endl;
        return jpgImages;
    }

    // Iterate through the directory
    for (const auto& entry : std::filesystem::directory_iterator(folderPath)) {
        // Check if the entry is a regular file and has a .jpg extension
        if (entry.is_regular_file() && entry.path().extension() == ".jpg") {
            jpgImages.push_back(entry.path().string());
        }
    }

    return jpgImages;
}

// Example usage
int main() {
    std::string folderPath = "images/"; // Replace with your folder path

    std::vector<std::string> jpgImages = getJpgImagesFromFolder(folderPath);

    std::cout << "Found .jpg images:\n";
    for (const auto& imagePath : jpgImages) {
        std::cout << imagePath << std::endl;
    }

    // Termination criteria
    cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001);

    // Prepare object points (0,0,0), (1,0,0), ..., (6,5,0)
    std::vector<cv::Point3f> objp;
    for (int i = 0; i < 7; ++i) {
        for (int j = 0; j < 10; ++j) {
            objp.push_back(cv::Point3f(j, i, 0));
        }
    }

    // Arrays to store object points and image points
    std::vector<std::vector<cv::Point3f>> objpoints; // 3D points in real-world space
    std::vector<std::vector<cv::Point2f>> imgpoints; // 2D points in image plane

    for (const auto& imagePath : jpgImages) {
        cv::Mat img = cv::imread(imagePath);
        if (img.empty()) {
            std::cerr << "Could not read image: " << imagePath << std::endl;
            continue;
        }
        // Resize (downscale) the image to the specified dimensions
        cv::Mat resizedImg;
        cv::resize(img, resizedImg, cv::Size(1280, 720));

        cv::Mat gray;
        cv::cvtColor(resizedImg, gray, cv::COLOR_BGR2GRAY);;
        // Find the chessboard corners
        std::vector<cv::Point2f> corners;
        bool ret = cv::findChessboardCorners(gray, cv::Size(10, 7), corners);

        // If found, add object points and image points
        if (ret) {
            objpoints.push_back(objp);

            // Refine corner locations
            cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1), criteria);
            imgpoints.push_back(corners);

            // Draw and display the corners
            cv::drawChessboardCorners(resizedImg, cv::Size(10, 7), corners, ret);
            cv::imshow("Image", resizedImg);
            cv::waitKey(500);
        }
    }

    // Now that we've collected all the object points and image points, we can calibrate the camera
    cv::Mat mtx, dist, rvecs, tvecs;
    std::vector<cv::Mat> rvecs_out, tvecs_out; // Rotation and translation vectors for each image

    // Calibrate the camera
    double ret = cv::calibrateCamera(objpoints, imgpoints, cv::Size(1280, 720), mtx, dist, rvecs_out, tvecs_out);

    // Check if calibration was successful
    if (ret) {
        std::cout << "Camera Calibration Successful!" << std::endl;
        std::cout << "Camera Matrix (Intrinsic Parameters):\n" << mtx << std::endl;
        std::cout << "Distortion Coefficients:\n" << dist << std::endl;
    }
    else {
        std::cerr << "Camera Calibration Failed!" << std::endl;
    }

    cv::Mat undistorted_img;
    cv::Mat img = cv::imread("images/IMG_20241121_195338.jpg");
    cv::Mat resizedImg;
    cv::resize(img, resizedImg, cv::Size(1280, 720));
    cv::undistort(resizedImg, undistorted_img, mtx, dist);
    cv::imshow("Undistorted Image", undistorted_img);
    cv::waitKey(0);

    cv::destroyAllWindows();
}