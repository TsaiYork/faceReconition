
//
//  main.cpp
//  opencv_test2
//
//  Created by yoyo on 2017/5/2.
//  Copyright © 2017年 Tsung-Yu Tsai. All rights reserved.
//  Ref:http://docs.opencv.org/2.4/modules/contrib/doc/facerec/tutorial/facerec_video_recognition.html#aligning-face-images

#include <iostream>
#include <fstream>
#include <sstream>
#include <string.h>
#include <stdio.h>

#include <highgui/highgui.hpp>
#include <core/core.hpp>
#include <imgproc/imgproc.hpp>
#include <objdetect/objdetect.hpp>
#include <opencv/cv.hpp>
#include <opencv2/face.hpp>

#define MAX_PEOPLE 5

using namespace cv;
using namespace std;


//void detect_and_display(CascadeClassifier face_cascade,Mat frame){
//    std::vector<Rect> faces;
//    
//    Mat frame_gray;
//    
//    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
//    equalizeHist( frame_gray, frame_gray );
//    
//    //-- Detect faces
//    face_cascade.detectMultiScale(frame_gray,faces, 1.1, 2, 0|CASCADE_SCALE_IMAGE,Size(100, 100));
//    
//    if(faces.size()==0)
//        printf("NO Face!!!!!\n");
//    for( size_t i = 0; i < faces.size(); i++ )
//    {
//        printf("I GOT YOU !!!!!\n");
//        Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
//        ellipse( frame, center, Size( faces[i].width/2, faces[i].height/2),0,0,360,Scalar(255,0,255), 4,8,0);
//        
//        //Mat faceROI = frame_gray( faces[i] );
//    }
//    //-- Show what you got
//    imshow("WebCam",frame);
//}

static string split_name(string path){
    string name;
    vector<string> vec_str;
    string::size_type pos;
    int count = 0;
    for (int i=path.size(); i>0 ;i--){
        pos = path.find_last_of("/",i);
        if(pos < path.size()){
        	string s = path.substr(pos+1,i-pos);
            vec_str.push_back(s);
            i = pos;
            count++;
        }
        //vec[0]=filename, vec[0]=dir_file
        if(count>=2)
            break;
    }
    name = vec_str[vec_str.size()-1];
    return name;
}

static void read_csv(CascadeClassifier face_cascade,const string& filename, vector<Mat>& images, vector<int>& labels, string* person, char separator = ';') {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given,please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }
    string line, path, classlabel,name;
    Mat temp, crop;
    vector< Rect_<int> > faces;
    int last_labels = -1;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        name = split_name(path);
        cout << classlabel <<"\t"<< name <<"\t"<<  path << endl;
        
        if(!path.empty() && !classlabel.empty()) {
            temp = imread(path,0);
            vector<Rect> faces;
            face_cascade.detectMultiScale(temp,faces, 1.1, 1, 0|CASCADE_SCALE_IMAGE,Size(250, 250));
            if(faces.size()!=0){
                cout << faces.size() << endl;
                for(int i = 0 ;i < faces.size() ; i++){
      	          //rectangle(temp, faces[i], CV_RGB(255, 0, 0),3);
                	crop = temp(Rect(faces[0].x,faces[0].y,faces[0].width,faces[0].height));
          		      //cout << "width = " << faces[i].width << "heigh = " << faces[i].height << endl;
                	resize(crop, crop, Size(500,500));
                	images.push_back(crop);
                	labels.push_back(atoi(classlabel.c_str()));
                }
            }
            else{
                resize(temp, temp, Size(500,500));
                images.push_back(temp);
                labels.push_back(atoi(classlabel.c_str()));
            }
            
            if(last_labels != atoi(classlabel.c_str())){
                person[atoi(classlabel.c_str())] = split_name(path).c_str();
            }
            last_labels = atoi(classlabel.c_str());
        }
    }
}

static void ShowDataPic(vector<Mat> &images, vector<int> &labels, string *person){
    stringstream ss_label,ss_pic;
    string windowName;
    int count=0;
    for(int i=0; i<labels.size() ; i++){
        ss_label << labels[i];
        ss_pic << count;
        windowName = ss_label.str() + "_" + person[labels[i]] + "_" + ss_pic.str();
        cout << windowName << endl;
        imshow(windowName, images[i]);
        ss_label.str("");ss_label.clear();
        ss_pic.str("");ss_pic.clear();
        if(i!=labels.size()-1 && labels[i]!=labels[i+1])
            count = 0;
        else
            count++;
    }
}


int main(int argc, const char * argv[]) {
    
    Mat pic;
    CascadeClassifier face_cascade;

    String face_cascadeName("/usr/local/Cellar/opencv3/3.2.0/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml");
    
    string fn_csv("/Users/yoyo/Documents/Intern/Face_Recognition/face_collection.csv");
    int deviceId = 0;
    
    VideoCapture cap(deviceId);
    if(!cap.isOpened()){
        printf("--(!)Error openning WebCam failure\n");
        return -1;
    }
    
    if(!face_cascade.load(face_cascadeName)){
        printf("--(!)Error loading face cascade\n");
        return -1;
    }
    
    vector<Mat> images;
    vector<int> labels;
    string person[MAX_PEOPLE];
    
    try {
        read_csv(face_cascade,fn_csv, images, labels, person);
    } catch (cv::Exception& e) {
        cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
        // nothing more we can do
        exit(1);
    }
    int im_width = images[0].cols;
    int im_height = images[0].rows;
    cout << "width = " << images[0].cols << "heigh = " << images[0].rows << endl;
    ShowDataPic(images, labels, person);
    
    Ptr<cv::face::FaceRecognizer> model = cv::face::createFisherFaceRecognizer();
    model->train(images, labels);
    
	

    while(cap.read(pic)) {
        
        // Clone the current frame:
        Mat original = pic.clone();
        // Convert the current frame to grayscale:
        Mat gray;
        cvtColor(original, gray, CV_BGR2GRAY);
        // Find the faces in the frame:
        vector< Rect_<int> > faces;
        face_cascade.detectMultiScale(gray,faces, 1.1, 5, 0|CASCADE_SCALE_IMAGE,Size(250, 250));
        
        // At this point you have the position of the faces in
        // faces. Now we'll get the faces, make a prediction and
        // annotate it in the video. Cool or what?
        for(int i = 0; i < faces.size(); i++) {
            // Process face by face:
            Rect face_i = faces[i];
            // Crop the face from the image. So simple with OpenCV C++:
            Mat face = gray(face_i);

            Mat face_resized;
            cv::resize(face, face_resized, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);
            // Now perform the prediction, see how easy that is:
            
            int prediction = model->predict(face_resized);
            // And finally write all we've found out to the original image!
            // First of all draw a green rectangle around the detected face:
            rectangle(original, face_i, CV_RGB(0, 255,0), 3);
            // Create the text we will annotate the box with:
            string box_text = format("This is %s", person[prediction].c_str());
            // Calculate the position for annotated text (make sure we don't
            // put illegal values in there):
            int pos_x = std::max(face_i.tl().x - 10, 0);
            int pos_y = std::max(face_i.tl().y - 10, 0);
            // And now put it into the image:
            putText(original, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 3.0, CV_RGB(0,255,0), 3.5);
        }
        // Show the result:
        imshow("face_recognizer", original);
        // And display it:
        char key = (char) waitKey(20);
        // Exit this loop on escape:
        if(key == 27)
            break;
    }

    printf("it's over\n");
}

