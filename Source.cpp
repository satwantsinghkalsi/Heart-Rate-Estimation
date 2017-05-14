#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/face.hpp"

#include <iostream>
#include <stdio.h>
#include <vector>
#include <math.h>
#include <omp.h>

#include "opencv2/fft.hpp"

#define DISPLAY_WINDOW_NAME ("Video Analytics Project")

using namespace std;
using namespace cv;
using namespace cv::face;


//Variables
string inputFileName = "Yuan_Before.avi";
string output_file_name_ = "Yuan_Before_output.avi";

int input_img_width_ = 0;
int input_img_height_ = 0;
VideoCapture* input_cap_ = NULL;
int output_img_width_ = 100;
int output_img_height_ = 100;
VideoWriter* output_cap_ = NULL;
bool write_output_file_ = true;

Mat img_input_;
Mat img_input_lab_;
Mat img_spatial_filter_;
Mat img_motion_;
Mat img_motion_mag_;
vector<Mat> img_vec_lap_pyramid_;
vector<Mat> img_vec_lowpass_1_;
vector<Mat> img_vec_lowpass_2_;
vector<Mat> img_vec_filtered_;

int lap_pyramid_levels_ = 5;
double cutoff_freq_low_ = 0.05;
double cutoff_freq_high_ = 0.4;
double lambda_c_ = 16;
double alpha_ = 20;
double chrom_attenuation_ = 0.1;
double exaggeration_factor_ = 2.0;
double delta_ = 0;
double lambda_ = 0;

double loop_time_ms_ = 0;
int frame_num_ = 0;
int frame_count_ = 0;
double input_fps_ = 8;

CascadeClassifier face_cascade;

deque<double> rawValues;
double freq = DBL_MIN;
double bpm;

int levels = 4;

int getCodecNumber(string file_name)
{
	string file_extn = file_name.substr(file_name.find_last_of('.') + 1);
	// Currently supported video formats are AVI and MPEG-4
	if (file_extn == "avi")
		return CV_FOURCC('M', 'J', 'P', 'G');
	else if (file_extn == "mp4")
		return CV_FOURCC('D', 'I', 'V', 'X');
	else
		return -1;
}

bool addToQueue(double rawVal)
{
	// If queue is filled
	if (rawValues.size() >= (input_fps_ * 10))
	{
		rawValues.pop_front();
		rawValues.push_back(rawVal);
		return true;
	}
	rawValues.push_back(rawVal);
	return false;
}


bool buildGaussianPyramid(const cv::Mat &img,
	const int levels,
	std::vector<cv::Mat> &pyramid)
{
	if (levels < 1) {
		perror("Levels should be larger than 1");
		return false;
	}
	pyramid.clear();
	cv::Mat currentImg = img;
	for (int l = 0; l<levels; l++) {
		cv::Mat down;
		cv::pyrDown(currentImg, down);
		pyramid.push_back(down);
		currentImg = down;
	}
	return true;
}
void concat(const std::vector<cv::Mat> &frames,
	cv::Mat &dst)
{
	cv::Size frameSize = frames.at(0).size();
	cv::Mat temp(frameSize.width*frameSize.height, frame_count_ - 1, CV_32FC3);
	for (int i = 0; i < frame_count_ - 1; ++i) {
		// get a frame if any
		cv::Mat input = frames.at(i);
		// reshape the frame into one column
		cv::Mat reshaped = input.reshape(3, input.cols*input.rows).clone();
		cv::Mat line = temp.col(i);
		// save the reshaped frame to one column of the destinate big image
		reshaped.copyTo(line);
	}
	temp.copyTo(dst);
}


void createIdealBandpassFilter(cv::Mat &filter, double fl, double fh, double rate)
{
	int width = filter.cols;
	int height = filter.rows;

	fl = 2 * fl * width / rate;
	fh = 2 * fh * width / rate;

	double response;

	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			// filter response
			if (j >= fl && j <= fh)
				response = 1.0f;
			else
				response = 0.0f;
			filter.at<float>(i, j) = response;
		}
	}
}

void temporalIdealFilter(const cv::Mat &src,
	cv::Mat &dst)
{
	cv::Mat channels[3];

	// split into 3 channels
	cv::split(src, channels);

	for (int i = 0; i < 3; ++i) {

		cv::Mat current = channels[i];  // current channel
		cv::Mat tempImg;

		int width = cv::getOptimalDFTSize(current.cols);
		int height = cv::getOptimalDFTSize(current.rows);

		cv::copyMakeBorder(current, tempImg,
			0, height - current.rows,
			0, width - current.cols,
			cv::BORDER_CONSTANT, cv::Scalar::all(0));

		// do the DFT
		cv::dft(tempImg, tempImg, cv::DFT_ROWS | cv::DFT_SCALE);

		// construct the filter
		cv::Mat filter = tempImg.clone();
		createIdealBandpassFilter(filter, cutoff_freq_low_, cutoff_freq_high_, input_fps_);

		// apply filter
		cv::mulSpectrums(tempImg, filter, tempImg, cv::DFT_ROWS);

		// do the inverse DFT on filtered image
		cv::idft(tempImg, tempImg, cv::DFT_ROWS | cv::DFT_SCALE);

		// copy back to the current channel
		tempImg(cv::Rect(0, 0, current.cols, current.rows)).copyTo(channels[i]);
	}
	// merge channels
	cv::merge(channels, 3, dst);

	// normalize the filtered image
	cv::normalize(dst, dst, 0, 1, CV_MINMAX);
}

void amplify(const cv::Mat &src, cv::Mat &dst)
{
	dst = src * alpha_;
}

void deConcat(const cv::Mat &src,
	const cv::Size &frameSize,
	std::vector<cv::Mat> &frames)
{
	for (int i = 0; i < frame_count_ - 1; ++i) {    // get a line if any
		cv::Mat line = src.col(i).clone();
		cv::Mat reshaped = line.reshape(3, frameSize.height).clone();
		frames.push_back(reshaped);
	}
}

void upsamplingFromGaussianPyramid(const cv::Mat &src,
	const int levels,
	cv::Mat &dst)
{
	cv::Mat currentLevel = src.clone();
	for (int i = 0; i < levels; ++i) {
		cv::Mat up;
		cv::pyrUp(currentLevel, up);
		currentLevel = up;
	}
	currentLevel.copyTo(dst);
}


bool displayHeartRate(double rawValue)
{
	if (addToQueue(rawValue))
	{
		//Queue is filled. We can perform heart rate analysis
		vector<double> fftarray(rawValues.size());
		double sum = 0;
		for (int i = 0; i < rawValues.size(); i++) {
			sum += rawValues[i];
		}

		vector<double> meanValues(rawValues.size());

		double mean = sum / rawValues.size();
		for (int i = 0; i < rawValues.size(); i++) {
			meanValues[i] = rawValues[i] / mean;
		}
		double maxFreq = DBL_MIN;
		Fft::transform(meanValues, fftarray);
		for (int i = 1; i < rawValues.size(); i++) {
			
			if (fftarray[i] > maxFreq && fftarray[i]>0.83 && fftarray[i]<1.67) {
				maxFreq = fftarray[i];
			}
		}
		freq = maxFreq;
		bpm = freq * 60;
		return true;
	}
	return false;
}

int main(int argc, const char** argv)
{

	//START Initializations  ----------------------------------------------------------------------------------
	input_cap_ = new VideoCapture(inputFileName);
	if (!input_cap_->isOpened())
	{
		cerr << "Error: Unable to open input video file: " << inputFileName << endl;
		return false;
	}

	if (input_img_width_ <= 0 || input_img_height_ <= 0)
	{
		// Use default input image size
		input_img_width_ = input_cap_->get(CV_CAP_PROP_FRAME_WIDTH);
		input_img_height_ = input_cap_->get(CV_CAP_PROP_FRAME_HEIGHT);
	}
	frame_count_ = input_cap_->get(CV_CAP_PROP_FRAME_COUNT);
	input_fps_ = input_cap_->get(CV_CAP_PROP_FPS);
	cout << "Input video resolution is (" << input_img_width_ << ", " << input_img_height_ << ")" << endl;

	// Output:
	// Output Display Window
	//cvNamedWindow(DISPLAY_WINDOW_NAME, CV_WINDOW_AUTOSIZE);
	if (output_img_width_ <= 0 || output_img_height_ <= 0)
	{
		// Use input image size for output
		output_img_width_ = input_img_width_;
		output_img_height_ = input_img_height_;
	}
	cout << "Output video resolution is (" << output_img_width_ << ", " << output_img_height_ << ")" << endl;

	// Output File:
	if (!output_file_name_.empty())
		write_output_file_ = true;

	if (write_output_file_)
	{
		output_cap_ = new cv::VideoWriter(output_file_name_,// filename
			getCodecNumber(output_file_name_), // codec to be used
			input_fps_, // frame rate of the video
			cv::Size(input_img_width_, input_img_height_), // frame size
			true  // color video
			);
		if (!output_cap_->isOpened())
		{
			cerr << "Error: Unable to create output video file: " << output_file_name_ << endl;
			return false;
		}
	}

	if (!face_cascade.load("haarcascade_frontalface_alt.xml")) { printf("--(!)Error loading face cascade\n"); return -1; };

	cout << "Initializations successful!" << endl;

	//END Initializations ---------------------------------------------------------------------------------------------------

	//Start Euler Color magnification
	cout << "Running Euler Color Magnification..." << endl << endl;

	// current frame
	cv::Mat input;
	// output frame
	cv::Mat output;
	// motion image

	cv::Mat motion;
	// temp image
	cv::Mat temp;

	// video frames
	std::vector<cv::Mat> frames;
	// down-sampled frames
	std::vector<cv::Mat> downSampledFrames;
	// filtered frames
	std::vector<cv::Mat> filteredFrames;

	// concatenate image of all the down-sample frames
	cv::Mat videoMat;
	// concatenate filtered image
	cv::Mat filtered;




	while (1)
	{
		input_cap_->read(img_input_);
		if (img_input_.empty())
			break;

		//Apply face detection
		Mat gray;
		cvtColor(img_input_, gray, COLOR_BGR2GRAY);
		equalizeHist(gray, gray);
		std::vector<Rect> faces;
		face_cascade.detectMultiScale(gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(output_img_width_, output_img_height_));

		if (faces.size() > 0)
		{
			cout << "Processing step 1 image frame: (" << frame_num_ << "/" << frame_count_ << ")" << flush << endl;
			Rect face_i = faces[0];
			Rect forehead = Rect(25, 10, 40, 10);
			img_input_ = img_input_(face_i)(forehead);
			//resize(img_input_, img_input_, Size(output_img_width_, output_img_height_), 1.0, 1.0, INTER_CUBIC);

			input = img_input_;
			input.convertTo(temp, CV_32FC3);
			frames.push_back(temp.clone());
			// spatial filtering
			std::vector<cv::Mat> pyramid;
			buildGaussianPyramid(temp, levels, pyramid);
			downSampledFrames.push_back(pyramid.at(levels - 1));
			frame_num_++;
		}

		char c = waitKey(1);
		if (c == 27)
			break;
	}

	frame_count_ = frame_num_;

	// 2. concat all the frames into a single large Mat
	// where each column is a reshaped single frame
	// (for processing convenience)
	concat(downSampledFrames, videoMat);

	// 3. temporal filtering
	temporalIdealFilter(videoMat, filtered);

	// 4. amplify color motion
	amplify(filtered, filtered);

	// 5. de-concat the filtered image into filtered frames
	deConcat(filtered, downSampledFrames.at(0).size(), filteredFrames);

	// 6. amplify each frame
	// by adding frame image and motions
	// and write into video
	input_cap_ = new VideoCapture(inputFileName);

	vector<double> values(frame_count_);

	frame_count_ = input_cap_->get(CV_CAP_PROP_FRAME_COUNT);
	int count = 0;
	for (int i = 0; i < frame_count_ - 1; ++i) {
		cout << "Processing step 6 image frame: (" << (i + 1) << "/" << frame_count_ << ")" << flush << endl;
		input_cap_->read(img_input_);
		if (img_input_.empty())
			break;
		//Apply face detection
		Mat gray;
		cvtColor(img_input_, gray, COLOR_BGR2GRAY);
		equalizeHist(gray, gray);
		std::vector<Rect> faces;
		face_cascade.detectMultiScale(gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(output_img_width_, output_img_height_));
		if (faces.size() > 0)
		{
			// up-sample the motion image        
			upsamplingFromGaussianPyramid(filteredFrames.at(count), levels, motion);
			resize(motion, motion, frames.at(count).size());
			temp = frames.at(count) + motion;
			output = temp.clone();
			double minVal, maxVal;
			minMaxLoc(output, &minVal, &maxVal); //find minimum and maximum intensities
			output.convertTo(output, CV_8UC3, 255.0 / (maxVal - minVal),
				-minVal * 255.0 / (maxVal - minVal));
			//if (write_output_file_)
			//	output_cap_->write(output);
			//Processing for heart rate
			vector<Mat> channels;
			split(output, channels);
			bool toDisplay = displayHeartRate(mean(channels[1])[0]);
			Rect face_i = faces[0];
			rectangle(img_input_, face_i, CV_RGB(0, 0, 255), 1);
			if (toDisplay)
			{
				String bpmText = "Heart Rate = " + format("%f", bpm);
				cout << "BPM = " << bpm << endl;
				putText(img_input_, bpmText, Point(face_i.x, face_i.y - 5), 1, 2, CV_RGB(0, 0, 255));
			}
			count++;
		}
		//imshow(DISPLAY_WINDOW_NAME, img_input_);
		if (write_output_file_)
			output_cap_->write(img_input_);
		char c = waitKey(1);
		if (c == 27)
			break;
	}
	return 0;
}