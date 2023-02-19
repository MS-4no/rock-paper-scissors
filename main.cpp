
#include "MediaImage.hpp"

#define FirstWeek 0
#define SecondWeek 0
#define ThirdWeek 1
#define FourthWeek 0
#define LandR 1 //1なら学習，0なら認識（imageモードの時）
#define KNN_MODE 1 //KNN_MODEが1のときk-NN処理，そうでない場合はSVM
#define FifthWeek 0
#define GESTURE 1 //Gestureモードなら1，Actionモードなら0
#define NonCompulsion 1 //非同期モードなら1，同期モードなら0

#include <random>
#include<iostream>

using namespace std;
using namespace media;

const string skeleton_name[15]
= {
	"HEAD", "NECK", "TORSO", "LEFT_SHOULDER", "LEFT_ELBOW",
	"LEFT_HAND", "RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_HAND", "LEFT_HIP",
	"LEFT_KNEE", "LEFT_FOOT", "RIGHT_HIP", "RIGHT_KNEE", "RIGHT_FOOT"
};

//Main functions
int main(int rgc, char **argv)
{
	MediaImage mediaImage = MediaImage();
	mediaImage.ColorInit(100);

	const string mode = "camera"; //image, camera, rgbdのいずれか

	try
	{
		//出力結果を保存する
		std::string resPath;

		//Image processing (image)
		if (mode == "image")
		{
			//画像を読み込むための変数
			cv::Mat img;

#if FirstWeek
			/*****************************/
			/** 1st week workshop       **/
			/**   -color filtering      **/
			/**   -labeling processing  **/
			/*****************************/
			//画像の読み込み
			mediaImage.ReadImage("../datas/colorpen.jpg", img);

			//画像の表示
			mediaImage.ShowImage("color", img);

			//キー入力待ち
			cv::waitKey(0);

			resPath = "01res/";
			_mkdir(resPath.c_str());

			//抽出した色画像を格納する変数
			cv::Mat extractedColor;

			//抽出する色を指定する。
			mediaImage.ExtractColor(175, 5, 50, 255, 0, 255, img, extractedColor);
			mediaImage.ShowImage("Extracted_Color_image", extractedColor);

			//処理した画像を保存する
			mediaImage.SaveImage(extractedColor, resPath + "red", ".jpg");


			cv::Mat blue_image;
			mediaImage.ExtractColor(110, 130, 50, 255, 0, 255, img, blue_image);
			mediaImage.ShowImage("Blue", blue_image);
			mediaImage.SaveImage(blue_image, resPath + "blue", ".jpg");

			cv::Mat green_image;
			mediaImage.ExtractColor(60, 80, 50, 255, 0, 255, img, green_image);
			mediaImage.ShowImage("Green", green_image);
			mediaImage.SaveImage(green_image, resPath + "green", ".jpg");
	
			cv::waitKey(0);//キー入力待ち

			//出力Windowを削除する
			mediaImage.ReleaseWindow("Extracted_Color_image");
			mediaImage.ReleaseWindow("Green");
			mediaImage.ReleaseWindow("Blue");


			//ノイズ除去
			cv::Mat temp1 = extractedColor.clone(), temp2 = extractedColor.clone();
			mediaImage.Morphology(extractedColor, extractedColor, "op", 2);
			mediaImage.ShowImage("Morphology_image", extractedColor);

			//処理した画像を保存する
			mediaImage.SaveImage(extractedColor, resPath + "res_op_2", ".jpg");


			
			mediaImage.Morphology(temp1, temp1, "cl", 2);
			mediaImage.ShowImage("Morphology_image", temp1);
			mediaImage.SaveImage(temp1, resPath + "res_cl_2", ".jpg");

			mediaImage.Morphology(temp2, temp2, "cl", 15);
			mediaImage.ShowImage("Morphology_image", temp2);
			mediaImage.SaveImage(temp2, resPath + "res_cl_15", ".jpg");


			
			cv::waitKey(0);

			//出力Windowを削除する
			mediaImage.ReleaseWindow("Morphology_image");


			//ラベリング処理
			//ラベリング処理に必要な変数
			cv::Mat LabelImg; //ラベリングされた画像
			cv::Mat stats; //ラベルの結果を格納尾
			cv::Mat centroids; //重心の結果を格納
			int nLab; //ラベリング数を格納
			cv::Mat resLabel; //ラベリング描画結果を格納

			mediaImage.LabelingProcessing(extractedColor, LabelImg, stats, centroids, nLab);
			mediaImage.DrawLabeling(extractedColor, LabelImg, stats, centroids, nLab, resLabel);
			mediaImage.ShowImage("Labeling", resLabel);

			//処理した画像を保存する
			mediaImage.SaveImage(extractedColor, resPath + "res_cl_2", ".jpg");

			
		    cv::Mat resLabel1, resLabel2;
			mediaImage.LabelingProcessing(temp1, LabelImg, stats, centroids, nLab);
			mediaImage.DrawLabeling(temp1, LabelImg, stats, centroids, nLab, resLabel1);
			mediaImage.ShowImage("Labeling1", resLabel1);
			mediaImage.SaveImage(resLabel1, resPath + "res_label1", ".jpg");

			mediaImage.LabelingProcessing(temp2, LabelImg, stats, centroids, nLab);
			mediaImage.DrawLabeling(temp2, LabelImg, stats, centroids, nLab, resLabel2);
			mediaImage.ShowImage("Labeling2", resLabel2);
			mediaImage.SaveImage(resLabel2, resPath + "res_label2", ".jpg");

			

			cv::waitKey(0);
			//出力Windowを削除する
			mediaImage.ReleaseWindow("Labeling");


			//エッジ処理
			cv::Mat edge_img; //エッジ画像を格納する
			mediaImage.EdgeDetection(extractedColor, edge_img, "canny");
			mediaImage.ShowImage("EdgeImage", edge_img);

			//処理した画像を保存する
			mediaImage.SaveImage(edge_img, resPath + "canny", ".jpg");

			
			cv::Mat res_lap, res_sol;
			mediaImage.EdgeDetection(extractedColor, res_lap, "laplacian");
			mediaImage.ShowImage("res_lap", res_lap);
			mediaImage.SaveImage(res_lap, resPath + "laplacian", ".jpg");

			mediaImage.EdgeDetection(extractedColor, res_sol, "sobel");
			mediaImage.ShowImage("res_sol", res_sol);
			mediaImage.SaveImage(res_sol, resPath + "sobel", ".jpg");

			

			cv::waitKey(0);

			//出力Windowを削除する
			mediaImage.ReleaseWindow("EdgeImage");


			//HoG特徴量
			cv::Mat hog_hist, hog_result;
			mediaImage.SetHoGParameters(9, 20, 1);
			mediaImage.CalcHOGHistgram(img, hog_hist); //HoG特徴量を抽出
			mediaImage.GetHoGOnImage(img, hog_result); //HoG特徴量の描画した画像を取得
			std::cerr << "HoG features\n"
				<< cv::format(hog_hist, cv::Formatter::FMT_CSV) << "\n";
			mediaImage.ShowImage("HoGImage", hog_result);

			//処理した画像を保存する
			mediaImage.SaveImage(hog_result, resPath + "res_hog", ".jpg");

			cv::Mat res_hog1, res_hog2;

			mediaImage.SetHoGParameters(9, 10, 1);
			mediaImage.CalcHOGHistgram(img, hog_hist);
			mediaImage.GetHoGOnImage(img, res_hog1);
			std::cerr << "HoG featurs\n"
				<< cv::format(hog_hist, cv::Formatter::FMT_CSV) << "\n";
			mediaImage.ShowImage("HoGImage1", res_hog1);

			//処理した画像を保存する
			mediaImage.SaveImage(res_hog1, resPath + "hog1", ".jpg");

			mediaImage.SetHoGParameters(9, 20, 5);
			mediaImage.CalcHOGHistgram(img, hog_hist);
			mediaImage.GetHoGOnImage(img, res_hog2);
			std::cerr << "HoG featurs\n"
				<< cv::format(hog_hist, cv::Formatter::FMT_CSV) << "\n";
			mediaImage.ShowImage("HoGImage2", res_hog2);

			mediaImage.SaveImage(res_hog2, resPath + "hog2", ".jpg");

			cv::waitKey(0);
			//出力Windowを削除する
			mediaImage.ReleaseWindow("HoGImage");


			//Release the memory
			mediaImage.ReleaseImage(img);
			mediaImage.ReleaseImage(extractedColor);
			mediaImage.ReleaseImage(hog_hist);
			mediaImage.ReleaseImage(hog_result);
			mediaImage.ReleaseImage(LabelImg);
			mediaImage.ReleaseImage(stats);
			mediaImage.ReleaseImage(centroids);
			mediaImage.ReleaseImage(resLabel);
			mediaImage.ReleaseImage(edge_img);
			mediaImage.ReleaseImage(hog_hist);
			mediaImage.ReleaseImage(hog_result);
			mediaImage.ReleaseImage(temp1);
			mediaImage.ReleaseImage(temp2);
			mediaImage.ReleaseWindow();

			//画像を読み込む
			cv::Mat janken_img, janken_res;
			
			mediaImage.ReadImage("../datas/sample.png", janken_img);
				//roi_imageに対して、肌色抽出を行う
			mediaImage.ExtractColor(150, 50, 1, 255, 0, 255, janken_img, janken_res);
			mediaImage.ShowImage("janken_res", janken_res);

			//オープニングやクロージングを適当な回数繰り返してノイズを除去する
			mediaImage.Morphology(janken_res, janken_res, "op", 3);
			mediaImage.Morphology(janken_res, janken_res, "cl", 3);
			mediaImage.ShowImage("opening jenken_res", janken_res);
			//肌色抽出後、ラベリング処理を行い、肌色の面積を求める。
			mediaImage.LabelingProcessing(janken_res, LabelImg, stats, centroids, nLab);
			mediaImage.DrawLabeling(janken_res, LabelImg, stats, centroids, nLab, resLabel);
			mediaImage.ShowImage("Labeling", resLabel);

			//抽出した領域の面積を取得することができる。
			std::vector<unsigned int> area = mediaImage.GetSquare();

			//肌色領域の最大値を求める
			int max = 0;
			for (int i = 0; i < area.size(); ++i)
			{
				if (max < area[i])
					max = area[i];
			}
			//肌色の面積の値をコンソール上に表示
			//これでグー、チョキ、パーの閾値を決める
			std::cout << max << std::endl;

			//最大値（max）の値によりグー、チョキ、パーのいずれかにする
			std::string janken = "None";
			if (max < 22004)
			{
				janken = "gu";
			}
			else if (max >= 22004 && max < 35034)
			{
				janken = "choki";
			}
			else
			{
				janken = "par";
			}

			cv::putText(janken_res, janken, cv::Point(0, 40), cv::FONT_HERSHEY_COMPLEX, 1.5, cv::Scalar(255, 255, 255), 2);
			mediaImage.ShowImage("RESULT", janken_res);
			mediaImage.MoveWindow("RESULT", 980, 300);
			mediaImage.SaveImage(janken_res, janken, ".jpg");
			cv::waitKey(0);

#endif

#if SecondWeek

			//画像の読み込み
			mediaImage.ReadImage("../datas/faces2.jpg", img);

			//画像の表示
			mediaImage.ShowImage("face", img);
			mediaImage.MoveWindow("face", 0, 0);

			//キー入力待ち
			cv::waitKey(0);

			resPath = "02res/";
			_mkdir(resPath.c_str());

			//カスケード画像を選択する
			std::string cascade = "../datas/asm/haarcascade_frontalface_alt2.xml";
			std::string eyes_cascade = "../datas/asm/haarcascade_eye.xml";
			std::vector<cv::Rect_<int>> faces;

			//顔検出
			mediaImage.GetFaceFeatures(img, faces, cascade);
			cv::Mat facial_image = img.clone();

			//検出した顔などの場所を囲う
			mediaImage.DrawFeatures(facial_image, faces);

			//画像の表示
			mediaImage.ShowImage("Detection_Face", facial_image);
			mediaImage.MoveWindow("Detection_Face", 700, 0);

			//処理した画像を保存する
			mediaImage.SaveImage(facial_image, resPath + "face", ".jpg");

			//両目検出 ---------------------------------------------------
			std::string eyes = "../datas/asm/haarcascade_eye.xml";
			mediaImage.GetFaceFeatures(img, faces, eyes);
			cv::Mat eye_image = img.clone();

			//検出した顔などの場所を囲う
			mediaImage.DrawFeatures(eye_image, faces);

			//画像の表示
			mediaImage.ShowImage("Detection_Eyes", eye_image);
			mediaImage.MoveWindow("Detection_Eyes", 0, 240);

			//処理画像を保存する
			mediaImage.SaveImage(eye_image, resPath + "eyes", ".jpg");
			
			//右目検出 ---------------------------------------------------
			std::string right_eye = "../datas/asm/haarcascade_righteye_2splits.xml";
			mediaImage.GetFaceFeatures(img, faces, right_eye);
			cv::Mat right_image = img.clone();

			//検出した顔などの場所を囲う

			mediaImage.DrawFeatures(right_image, faces);

			//画像の表示

			mediaImage.ShowImage("Detection_EyeR", right_image);
			mediaImage.MoveWindow("Detection_EyeR", 700, 240);

			//処理画像を保存する
			mediaImage.SaveImage(right_image, resPath + "right_eye", ".jpg");

			//左目検出 ---------------------------------------------------
			std::string left_eye = "../datas/asm/haarcascade_lefteye_2splits.xml";
			mediaImage.GetFaceFeatures(img, faces, left_eye);
			cv::Mat left_image = img.clone();

			//検出した顔などの場所を囲う

			mediaImage.DrawFeatures(left_image, faces);

			//画像の表示

			mediaImage.ShowImage("Detection_EyeR", left_image);
			mediaImage.MoveWindow("Detection_EyeR", 0, 480);

			//処理画像を保存する
			mediaImage.SaveImage(left_image, resPath + "left_eye", ".jpg");

			//メガネあり検出 ---------------------------------------------------
			std::string eye_with_glass = "../datas/asm/haarcascade_eye_tree_eyeglasses.xml";
			mediaImage.GetFaceFeatures(img, faces, eye_with_glass);
			cv::Mat glass_image = img.clone();

			//検出した顔などの場所を囲う

			mediaImage.DrawFeatures(glass_image, faces);

			//画像の表示

			mediaImage.ShowImage("Detection_EyeR", glass_image);
			mediaImage.MoveWindow("Detection_EyeR", 700, 480);

			//処理画像を保存する
			mediaImage.SaveImage(glass_image, resPath + "glass", ".jpg");

			cv::waitKey(0);
			//出力Windowを削除する
			mediaImage.ReleaseWindow("Detection_Face");
			mediaImage.ReleaseWindow("Detection_Eyes");
			mediaImage.ReleaseWindow("Detection_EyeR");
			mediaImage.ReleaseWindow("Detection_EyeL");
			mediaImage.ReleaseWindow("Detection_EyeG");

			//カスケード画像を選択する
			std::vector<cv::Rect_<int>> eyes2;
			cv::Mat eyes_image2 = img.clone();

			//顔の検出してから目の検出を行う
			mediaImage.GetFaceFeatures(eyes_image2, faces, eyes2, eyes_cascade);

			//検出した場所を囲う
			mediaImage.DrawFeatures(eyes_image2, eyes2);

			//処理した画像を表示する
			mediaImage.ShowImage("Detection_EyesInFace", eyes_image2);
			mediaImage.MoveWindow("Detection_EyesInFace", 700, 0);

			//処理した画像を保存する
			mediaImage.SaveImage(eyes_image2, resPath + "eyes_in_face", ".jpg");

			cv::waitKey(0);
			//出力Windowを削除する
			mediaImage.ReleaseWindow("Detection_EyesInFace");


			//パラメータを調整する
			//カスケード画像を選択する
			std::vector<cv::Rect_<int>> faces3, faces4;
			cv::Mat face3 = img.clone();
			cv::Mat face4 = img.clone();

			//見逃し率が高い場合2.0
			mediaImage.GetFaceFeatures(face3, faces3, cascade, 2.0);

			//見逃し率が低い場合1.001
			mediaImage.GetFaceFeatures(face4, faces4, cascade, 3.001);

			//検出した場所を囲う
			mediaImage.DrawFeatures(face3, faces3);
			mediaImage.DrawFeatures(face4, faces4);

			//処理した画像を表示する
			mediaImage.ShowImage("Detection_Face3", face3);
			mediaImage.MoveWindow("Detection_Face3", 0, 350);
			mediaImage.ShowImage("Detection_Face4", face4);
			mediaImage.MoveWindow("Detection_Face4", 700, 350);

			//処理した画像を保存する
			mediaImage.SaveImage(face3, resPath + "face(2.0)", ".jpg");
			mediaImage.SaveImage(face4, resPath + "face(1.001)", ".jpg");

			cv::waitKey(0);
			//出力Windowを削除する
			mediaImage.ReleaseWindow("Detection_Eyes2");


			for (float alpha = 1.01; alpha < 1.101; alpha += 0.01) {
				std::cout << to_string(alpha) << std::endl;

				cv::Mat face_alpha = img.clone();
				std::vector<cv::Rect_<int>> faces_alpha;
				mediaImage.Restart();
				mediaImage.GetFaceFeatures(face_alpha, faces_alpha, cascade, alpha);
				mediaImage.ProcessTime();
				mediaImage.DrawFeatures(face_alpha, faces_alpha);
				std::stringstream ss;
				ss << to_string(alpha);
				mediaImage.SaveImage(face_alpha, resPath + "face(" + ss.str(), ").jpg");
				mediaImage.ShowImage("Detection_Face with different alpha", face_alpha);
				mediaImage.MoveWindow("Detection_Face with different alpha", 0, 350);
			}

			for (float alpha = 1.1; alpha < 2.001; alpha += 0.1) {
				std::cout << to_string(alpha) << std::endl;

				cv::Mat face_alpha = img.clone();
				std::vector < cv::Rect_<int>> faces_alpha;
				mediaImage.Restart();
				mediaImage.GetFaceFeatures(face_alpha, faces_alpha, cascade, alpha);
				mediaImage.ProcessTime();
				mediaImage.DrawFeatures(face_alpha, faces_alpha);
				std::stringstream ss;
				mediaImage.SaveImage(face_alpha, resPath + "face(" + ss.str(), ").jpg");
				mediaImage.ShowImage("Detection_Face with different alpha", face_alpha);
				mediaImage.MoveWindow("Detection_Face with different alpha", 0, 350);
			}
			cv::waitKey(0);
			mediaImage.ReleaseWindow("Detection_Face with different alpha");

			//顔検出
			mediaImage.GetFaceFeatures(img, faces, cascade, 1.02);
			cv::Mat facial_image3 = img.clone();

			//検出した顔などの場所を囲う
			mediaImage.DrawFeatures(facial_image3, faces);

			//画像の表示
			mediaImage.ShowImage("Detection_Face", facial_image3);
			mediaImage.MoveWindow("Detection_Face", 700, 0);
			cv::waitKey(0);

			//カスケード特徴を選択する

			std::string eyes3 = "../datas/asm/haarcascade_eye.xml";
			std::vector<cv::Rect_<int>>rect_eyes3;
			cv::Mat eyes_image3 = img.clone();

			//顔の検出をしてから目の検出を行う
			mediaImage.GetFaceFeatures(eyes_image3, faces, rect_eyes3, eyes3, 1.07);
			mediaImage.DrawFeatures(eyes_image3, rect_eyes3);
			
			//画像の表示
			mediaImage.ShowImage("Detection_EyesinFace", eyes_image3);
			mediaImage.MoveWindow("Detection_EyesinFace", 0, 300);

			//処理画像を保存する
			mediaImage.SaveImage(eyes_image3, resPath + "eyes_(1.07)", ".jpg");

			//カスケード特徴を選択する
			std::string glass3 = "../datas/asm/haarcascade_eye_tree_eyeglasses.xml";
			std::vector<cv::Rect_<int>>rect_glass3;
			cv::Mat glass_image3 = img.clone();

			//顔の検出をしてから目の検出を行う
			mediaImage.GetFaceFeatures(glass_image3, faces, rect_glass3, glass3, 1.07);
			mediaImage.DrawFeatures(glass_image3, rect_glass3);

			//画像の表示
			mediaImage.ShowImage("Detection_GlassinFace", glass_image3);
			mediaImage.MoveWindow("Detection_GlassinFace", 0, 300);

			//処理画像を保存する
			mediaImage.SaveImage(glass_image3, resPath + "glass_(1.07)", ".jpg");


			//笑顔の検出
			//画像の読み込み
			mediaImage.ReadImage("../datas/smile3.jpg", img);

			//カスケード画像を選択する
			std::string cascade_simle = "../datas/asm/haarcascade_smile.xml";
			std::vector<cv::Rect_<int>> smile;

			//顔検出と笑顔検出
			mediaImage.GetFaceFeatures(img, faces, cascade, 1.01);
			mediaImage.GetFaceFeatures(img, faces, smile, cascade_simle);
			cv::Mat smile_image = img.clone();

			//検出した顔などの場所を囲う
			mediaImage.DrawFeatures(smile_image, faces);
			mediaImage.DrawFeatures(smile_image, smile);

			//画像の表示
			mediaImage.ShowImage("Simle_Face", smile_image);
			mediaImage.MoveWindow("Simle_Face", 0, 0);

			//処理した画像を保存する
			mediaImage.SaveImage(smile_image, resPath + "smile", ".jpg");

			cv::waitKey(0);
			

			//笑顔のレベルを評価する
			vector<int> count_parts = mediaImage.GetDetectedCount();
			std::vector<float> scores;
			mediaImage.EvaluateSmile(count_parts, scores);

			//笑顔度に応じて描画する
			cv::Mat evaluated_smile_img = img.clone();
			mediaImage.DrawFeatures(evaluated_smile_img, faces, scores);
			//画像の表示
			mediaImage.ShowImage("Evaluated_Simle_Face", evaluated_smile_img);
			mediaImage.MoveWindow("Evaluated_Simle_Face", 0, 450);
			cv::waitKey(0);

			//処理した画像を保存する
			mediaImage.SaveImage(smile_image, resPath + "evaluated_smile", ".jpg");		

			//出力Windowを削除する
			mediaImage.ReleaseWindow("Simle_Face");
			mediaImage.ReleaseWindow("Evaluated_Simle_Face");

			for (float alpha = 1.01; alpha < 1.101; alpha += 0.01) {
				std::cout << to_string(alpha) << std::endl;
				std::stringstream ss;
				ss << to_string(alpha);

				mediaImage.Restart();
				mediaImage.GetFaceFeatures(img, faces, smile, cascade_simle, alpha);
				mediaImage.ProcessTime();

				cv::Mat smile_image = img.clone();

				//検出した顔などの場所を囲う
				mediaImage.DrawFeatures(smile_image, faces);
				mediaImage.DrawFeatures(smile_image, smile);

				//画像の表示
				mediaImage.ShowImage("Simle_Face", smile_image);
				mediaImage.MoveWindow("Simle_Face", 0, 0);

				//処理した画像を保存する
				mediaImage.SaveImage(smile_image, resPath + "smile_(" + ss.str() + ")", ".jpg");

				//笑顔のレベルを評価する
				vector<int> count_parts = mediaImage.GetDetectedCount();
				std::vector<float> scores;
				mediaImage.EvaluateSmile(count_parts, scores);

				//笑顔度に応じて描画する
				cv::Mat evaluated_smile_img = img.clone();
				mediaImage.DrawFeatures(evaluated_smile_img, faces,scores);

				//画像の表示
				mediaImage.ShowImage("Evaluated_Simle_Face", evaluated_smile_img);
				mediaImage.MoveWindow("Evaluated_Simle_Face", 0, 450);

				//処理した画像を保存する
				mediaImage.SaveImage(smile_image, resPath + "evaluated_smile_(" + ss.str() + ")", ".jpg");

			}

			cv::waitKey(0);
			mediaImage.ReleaseWindow("Simle_Face");
			mediaImage.ReleaseWindow("Evaluated_Simle_Face")



			// ------------ 人の検出 -----------------
			//画像の読み込み
			mediaImage.ReadImage("../datas/people.jpg", img);

			//画像の表示
			mediaImage.ShowImage("People", img);
			mediaImage.MoveWindow("People", 0, 0);

			//キー入力待ち
			cv::waitKey(0);

			//人検出用の変数
			cv::HOGDescriptor hog;
			cv::Mat gray_img;
			std::vector<cv::Rect> human;
			
			//人検出時はグレー画像に変換する必要あり
			cv::cvtColor(img, gray_img, CV_BGR2GRAY);

			//人を検出する
			mediaImage.GetHumanFeatures(gray_img, hog, human, 8, 16, 1.1);

			//描画する
			cv::Mat resHuman = img.clone();
			mediaImage.DrawFeatures(resHuman, human);
						
			//画像の表示
			mediaImage.ShowImage("Detected People", resHuman);
			mediaImage.MoveWindow("Detected People", 700, 0);
			mediaImage.SaveImage(resHuman, resPath + "Human(1.1)", ".jpg");
			cv::waitKey(0);

		for (float alpha = 1.01; alpha < 1.101; alpha += 0.01) {
			std::cout << to_string(alpha) << std::endl;
			std::stringstream ss;
			ss << to_string(alpha);

			//人を検出する
			mediaImage.Restart();
			mediaImage.GetHumanFeatures(gray_img, hog, human, 8, 16, alpha);
			mediaImage.ProcessTime();

			//描画する
			cv::Mat resHuman = img.clone();
			mediaImage.DrawFeatures(resHuman, human);

			//画像表示
			mediaImage.ShowImage("Detected People", resHuman);
			mediaImage.MoveWindow("Detectes People", 700, 0);
			mediaImage.SaveImage(resHuman, resPath + "Human(" + ss.str() + ")", ".jpg");
			}
			//スライドを1にすると検出精度はあがるが速度は遅い
			mediaImage.GetHumanFeatures(gray_img, hog, human, 1, 16, 1.1);

			//描画する
			cv::Mat resHuman5 = img.clone();
			mediaImage.DrawFeatures(resHuman5, human);

			//画像表示
			mediaImage.ShowImage("Detected People with different parameter", resHuman5);
			mediaImage.MoveWindow("Detectes People with different parameter", 700, 350);
			mediaImage.SaveImage(resHuman5, resPath + "Human(1.1)_stride(1)", ".jpg");

			cv::waitKey(0);
			mediaImage.ReleaseWindow("Detected People");
			mediaImage.ReleaseWindow("Detected People with different parameter");
		
			std::vector<cv::Rect_<int>> faces6;
			cv::Mat face_image6 = img.clone();

			//人の検出をしてから顔の検出
			mediaImage.GetHumanFeatures(gray_img, hog, human, 1, 16, 1.1);
			mediaImage.DrawFeatures(face_image6, human);
			mediaImage.GetFaceFeatures(face_image6, human, faces6, cascade, 1.02);
			mediaImage.DrawFeatures(face_image6, faces6);
			mediaImage.ShowImage("Detected People and face", face_image6);
			mediaImage.MoveWindow("Detected People and face", 700, 0);
			mediaImage.SaveImage(face_image6, resPath + "Human_and_face", ".jpg");
			cv::waitKey(0);
			
#endif

#if FourthWeek
			
			resPath = "04res/";
			_mkdir(resPath.c_str());

			//特徴量モードを選択（color or hog）
			const std::string featureMode = "color";

			//画像データベースを読み込む
			std::map<int, std::string> labels;

			//データベースを作成したときのフォルダ名を記入
			std::string my_dir = "result_images";
			mediaImage.ReadFile(labels, "../datas/" + my_dir + "/label.txt");

			//認識したい画像を読み込む
			img = cv::imread("../datas/sample2.png");

			//ヒストグラム用の変数
			const int bin = 30; //ヒストグラムを作成するときの特徴ベクトルの次元数（色の場合1～180, HoGの場合は9）
			const int channel = 1; //色特徴量を扱う場合のみに1～3へ変更可能（彩度や明度の特徴を使いたいとき）

#if KNN_MODE
			//機械学習の変数および初期値の準備
			cv::Ptr<cv::ml::KNearest> k_nn;

			//k-NNのセットアップ
			const int k = 1;

			//学習結果の保存名
			stringstream ss; ss << k;
			const string saveFileName = resPath + "train_kNN_[" + featureMode + "]_(" + ss.str() + ").xml";
#else
			//機械学習の変数および初期値の準備
			cv::Ptr<cv::ml::SVM> svm;

			const double c = 1.0;
			const double gamma = 1.0;

			//学習結果の保存名
			stringstream ss1; ss1 << c;
			stringstream ss2; ss2 << gamma;
			const string saveFileName = resPath + "train_svm_[" + featureMode + "]_(" + ss1.str() + "-" + ss2.str() + ").xml";


#endif
#if LandR
			//----- Leaning Mode ------
			//画像データベースを読み込む
			std::vector<std::string> files;
			mediaImage.ReadFile(files, "../datas/" + my_dir + "/database.txt");

			//学習モードの処理
			cv::Mat histograms(files.size(), bin * channel, CV_32FC1);
			cv::Mat label(files.size(), 1, CV_32SC1);

			//学習データの作成をする
			for (int i = 0; i < files.size(); ++i)
			{
				//画像を読み込む
				cv::Mat img = cv::imread("../datas/" + files[i]);

				//読み込みエラーは飛ばす
				if (img.empty())
				{
					cerr << files[i] << "がありません。 \n";
					continue;
				}

				//ラベルを作成する
				for (int l = 0; l < labels.size(); ++l)
				{
					if (files[i].find(labels[l].c_str()) != std::string::npos)
					{
						label.at<int>(i, 0) = l;
					}
				}

				//色特徴量を取得する
				cv::Mat hist = histograms.row(i);

				if (featureMode == "color")
				{
					if (!mediaImage.GenerateColorHistogram(img, hist, bin, channel))
						continue;
				}
				else if (featureMode == "hog")
				{
					
					mediaImage.SetHoGParameters(9, 20, 1);
					mediaImage.CalcHOGHistgram(img, hist); //HoG特徴量を抽出

				}
			}

#if KNN_MODE
			//k-NNのセットアップ
			//k = 1, アルゴリズムはBRUTEをデフォルト
			mediaImage.kNN(k);
			mediaImage.Train(histograms, label, k_nn);

			//学習結果の保存
			cerr << "Saving the classifier to " << saveFileName << "\n";
			k_nn->save(saveFileName);
#else
			
			mediaImage.SupperVctorMachine("C_SVC", "LINEAR", c, 1.2, gamma, 1.0);
			mediaImage.Train(histograms, label, svm);

			//学習結果の保存
			cerr << "Saving the classifier to " << saveFileName << "\n";
			svm->save(saveFileName);

#endif

#else
			//----- Recognition Mode ------
			//学習データを読み込む
#if KNN_MODE
			k_nn = cv::Algorithm::load<cv::ml::KNearest>(saveFileName);
#else
			
			svm = cv::Algorithm::load<cv::ml::SVM>(saveFileName);

#endif

			//認識したい画像を読み込む
			img = cv::imread("../datas/sample2.png");
			cv::Mat labelTest(1, 1, CV_32SC1);

			cerr << "ラベル番号を入力してください: ";
			cin >> labelTest.at<int>(0, 0);

			//読み込みエラーは飛ばす
			if (img.empty())
			{
				cerr << "画像ファイルがありません。 \n";
			}

			//画像サイズを統一する
			cv::Mat resize;
			cv::resize(img, resize, cv::Size(100, 100), cv::INTER_CUBIC);

			//色特徴量を取得する
			cv::Mat hist(1, bin * channel, CV_32FC1);

			if (featureMode == "color")
			{
				if (!mediaImage.GenerateColorHistogram(resize, hist, bin, channel))
				{
					cerr << "ヒストグラムが作成できませんでした\n";
					exit(1);
				}
			}
			else if (featureMode == "hog")
			{
				
				mediaImage.SetHoGParameters(9, 20, 1);
				mediaImage.CalcHOGHistgram(resize, hist); //HoG特徴量を抽出

			}
			cerr << hist << "\n";

			//識別の評価を行う
			cv::Mat responce;

#if KNN_MODE
			k_nn->predict(hist, responce);
#else
			
			svm->predict(hist, responce);
			
#endif
			cerr << "accuracy : "
				<< mediaImage.CalculateAccuracyPercent(labelTest, responce) << "%\n";

			cerr << " True label\tPredicted label\n";
			for (int i = 0; i < responce.rows; ++i)
			{
				cerr << " " << labelTest.at<int>(i, 0) << "\t\t" << responce.at<float>(i, 0) << "\n";
				cerr << " " << labels[labelTest.at<int>(i, 0)] << "\t\t" << labels[responce.at<float>(i, 0)] << "\n";
		
			}

#endif

			mediaImage.Stop();
#endif

		}

		//Video processing (web camera)
		else if (mode == "camera")
		{
			//カメラデフォルトの設定。320×240pxとなっている。
			mediaImage.InitVideo(0, 320, 240); 

#if ThirdWeek
			/*****************************/
			/** 3rd week workshop       **/
			/**   -video processing     **/
			/*****************************/
			const bool kadai[] = { 0, 0, 0, 0, 0, 1 };

			//事前に背景を取得する．
			static cv::Mat background;

			//フレーム間差分用
			cv::Mat old;

			//バックグラウンドの取得するまで，処理を行う．
			while (1 && kadai[2]|| 1 && kadai[5])
			{
				//画像をアップデートする
				mediaImage.UpdateVideo();
				cv::Mat temp = mediaImage.GetImage();
				cv::putText(temp, "Press [S] or[s] to save.", cv::Point(0, 40), cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);

				mediaImage.ShowImage("TEMP", temp);
				int key = cv::waitKey(1);

				

				if (key == 's' || key == 'S')
				{
					background = mediaImage.GetImage();
					mediaImage.ReleaseWindow();
					break;
				}
				
			
			}
#endif
			//ランダムの関数-------------------------------------------------------------
			random_device rd;
			mt19937 gen(rd());
			uniform_int_distribution<> 	dist(1, 3);

			//相手の手
			int enemy = dist(gen);
			cv::Mat eimg;
			//勝敗
			int vod = dist(gen);
			cv::Mat vodimg;

			while (1)
			{
				//画像をアップデートする
				mediaImage.UpdateVideo();
				cv::Mat img = mediaImage.GetImage();
				int key = cv::waitKey(1);

				//ビデオが終わるもしくはqキーが押されれば終了
				if (img.empty() || key == 'q' || key == 'Q')
				{
					mediaImage.ReleaseWindow();
					break;
				}

#if ThirdWeek
				resPath = "03res/";
				_mkdir(resPath.c_str());

				//元の画像を表示する
				mediaImage.ShowImage("Camera", img);
				mediaImage.MoveWindow("Camera", 0, 0);

				
				if (kadai[0])
				{
					mediaImage.ExtractColor(170, 20, 20, 255, 10, 255, img, img);
					mediaImage.Morphology(img, img, "op", 5);
					mediaImage.Morphology(img, img, "cl", 5);

					//ラベリング処理に必要な変数
					cv::Mat LabelImg; //ラベリングされた画像
					cv::Mat stats; //ラベルの結果を格納
					cv::Mat centroids; //重心の結果を格納
					int nLab; //ラベリング数を格納
					cv::Mat resLabel; //ラベリング描画結果を格納

									  //肌色抽出後、ラベリング処理を行い、肌色の面積を求める
					mediaImage.LabelingProcessing(img, LabelImg, stats, centroids, nLab);
					mediaImage.DrawLabeling(img, LabelImg, stats, centroids, nLab, resLabel);
					mediaImage.ShowImage("Labeling", resLabel);
	
				}
				else if (kadai[1])
				{
					
					std::string cascade = "../datas/asm/haarcascade_frontalface_alt2.xml";
					std::string cascade_simle = "../datas/asm/haarcascade_smile.xml";
					std::vector<cv::Rect_<int>> faces;
					std::vector<cv::Rect_<int>> smile;

					//顔検出と笑顔検出
					mediaImage.GetFaceFeatures(img, faces, cascade, 1.1);
					mediaImage.GetFaceFeatures(img, faces, smile, cascade_simle, 1.1);

					vector<int> count_parts = mediaImage.GetDetectedCount();
					std::vector<float> scores;
					mediaImage.EvaluateSmile(count_parts, scores);

					//笑顔度に応じて描画する
					cv::Mat evaluated_smile_img = img.clone();
					mediaImage.DrawFeatures(img, faces, scores);
					//画像の表示
					mediaImage.ShowImage("Evaluated_Simle_Face", img);
					mediaImage.MoveWindow("Evaluated_Simle_Face", 0, 260);

					for (int i = 0; i < scores.size(); ++i)
					{
						if (scores[i] >= 0.6)
						{
							stringstream ss;
							ss << i;
							mediaImage.SaveImage(img, resPath + "evaluated_smile(" + ss.str() + ")", ".jpg");
						}
					}


				}
				else if (kadai[2])
				{
					//----- 背景差分　デフォルトプログラム

					//背景画像を表示する
					mediaImage.ShowImage("BACKGROUND", background);
					mediaImage.MoveWindow("BACKGROUND", 320, 0);
					
					/*-------------------------- 課題3 ここから-----------------------------*/
					cv::Mat gray_img, gray_background, diff_img;
					cv::cvtColor(img, gray_img, CV_BGR2GRAY);

					cv::cvtColor(background, gray_background, CV_BGR2GRAY);

					cv::absdiff(gray_img, gray_background, diff_img);




					cv::Mat binary, bgr_binary;

					cv::threshold(diff_img, binary, 20, 255, CV_THRESH_BINARY);

					cv::cvtColor(binary, bgr_binary, CV_GRAY2BGR);

					cv::Mat result;

					cv::bitwise_and(img, bgr_binary, result);

					mediaImage.ShowImage("DIFF", result);

					mediaImage.MoveWindow("DIFF", 640, 0);
				}
				else if (kadai[3])
				{
					//最初のフレームの場合はフレーム間差分しないようにする．
					if (old.empty())
					{
						old = img.clone();
						mediaImage.ShowImage("OLD", old);
						mediaImage.MoveWindow("OLD", 320, 0);
						continue;
					}
					mediaImage.ShowImage("OLD", old);
					mediaImage.MoveWindow("OLD", 320, 0);

					cv::Mat gray_img, gray_old, diff_img;
					cv::cvtColor(img, gray_img, CV_BGR2GRAY);
					cv::cvtColor(old, gray_old, CV_BGR2GRAY);
					cv::absdiff(gray_img, gray_old, diff_img);

					cv::Mat binary, bgr_binary;
					cv::threshold(diff_img, binary, 20, 255, CV_THRESH_BINARY);
					cv::cvtColor(binary, bgr_binary, CV_GRAY2BGR);

					cv::Mat result;
					cv::bitwise_and(img, bgr_binary, result);
					mediaImage.ShowImage("DIFF", result);
					mediaImage.MoveWindow("DIFF", 640, 0);

					old = img;
					
				}
				else if (kadai[4])
				{
					//最初のフレームの場合はフレーム間差分しないようにする．
					if (old.empty())
					{
						old = img.clone();
						mediaImage.ShowImage("OLD", old);
						mediaImage.MoveWindow("OLD", 320, 0);
						continue;
					}
					mediaImage.ShowImage("OLD", old);
					mediaImage.MoveWindow("OLD", 320, 0);

					cv::Mat current = img.clone(), res = img.clone(), flow;
					std::vector<cv::Point2f> points;

					//オプティカルフローを計算する
					mediaImage.CalclateOpticalFlow(current, old, flow, points, 50);

					//オプティカルフローを表示する
					mediaImage.ShowOpticalFlow(points, flow, res);

					mediaImage.ShowImage("OPTICAL", res);
					mediaImage.MoveWindow("OPTICAL", 0, 300);
					old = img.clone();
					
					//オプティカルフローを計算する
					mediaImage.CalclateOpticalFlow(current, old, flow, points, 10); //値を例えば10に変更

					//オプティカルフローを表示する

					mediaImage.ShowOpticalFlow(points, flow, res);
					mediaImage.ShowImage("OPTICAL", res);
					mediaImage.MoveWindow("OPTICAL", 0, 300);

					int key = cv::waitKey(1);
					if (key == 'S' || key == 's')
					{
					mediaImage.SaveImage(res, resPath + "opticalflow", ".jpg");
					}
					old = img.clone();
					
				}
				else if (kadai[5])
				{
					
					
					cv::Mat gray_img, gray_background, diff_img;
					cv::cvtColor(img, gray_img, CV_BGR2GRAY);
					cv::cvtColor(background, gray_background, CV_BGR2GRAY);
					cv::absdiff(gray_img, gray_background, diff_img);

					cv::Mat binary, bgr_binary;
					cv::threshold(diff_img, binary, 20, 255, CV_THRESH_BINARY);
					cv::cvtColor(binary, bgr_binary, CV_GRAY2BGR);
					cv::Mat result;
					cv::bitwise_and(img, bgr_binary, result);
					

					//肌色抽出
					mediaImage.ExtractColor(150, 50, 1, 255, 0, 255, result, result);

					//オープニングクロージングでノイズ除去
					mediaImage.Morphology(result, result, "op", 3);
					mediaImage.Morphology(result, result, "cl", 3);

					//ラベリング処理に必要な変数
					cv::Mat LabelImg; //ラベリングされた画像
					cv::Mat stats; //ラベルの結果を格納尾
					cv::Mat centroids; //重心の結果を格納
					int nLab; //ラベリング数を格納
					cv::Mat resLabel; //ラベリング描画結果を格納

					//肌色抽出後、ラベリング処理を行い、肌色の面積を求める。
					mediaImage.LabelingProcessing(result, LabelImg, stats, centroids, nLab);
					mediaImage.DrawLabeling(result, LabelImg, stats, centroids, nLab, resLabel);
					mediaImage.ShowImage("Labeling", resLabel);
					mediaImage.MoveWindow("Labeling", 320,0);
					
					//抽出した領域の面積を取得することができる。
					std::vector<unsigned int> area = mediaImage.GetSquare();
					//肌色領域の最大値を求めるs
					int max = 0;
					for (int i = 0; i < area.size(); ++i)
					{
						if (max < area[i])
							max = area[i];
					}
					//最大値（max）の値によりグー、チョキ、パーのいずれかにする
					std::string janken = "None";
					if (5000 < max && max < 9000) //閾値は画像に写る手の大きさに大きく依存する
					{
						janken = "gu";
					}
					else if (max >= 9000 && max < 13000)
					{
						janken = "choki";
					}
					else if (max>13000)
					{
						janken = "par";
					}
					else {
						janken = "None";
					}
					cv::putText(result, janken, cv::Point(0, 40), cv::FONT_HERSHEY_COMPLEX, 1.5, cv::Scalar(255, 255, 255), 2);
					mediaImage.ShowImage("RESULT", result);
					mediaImage.MoveWindow("RESULT", 120, 320);

					//相手の表示----------------------------------------------------------------------
					if (enemy==1) 
					{
						//チョキ
						//画像の読み込み
						mediaImage.ReadImage("../datas/choki.png", eimg);
						//画像の表示
						mediaImage.ShowImage("ENEMY", eimg);
						mediaImage.MoveWindow("ENEMY", 640, 320);
					}
					else if (enemy == 2) 
					{
						//パー
						//画像の読み込み
						mediaImage.ReadImage("../datas/par.png", eimg);
						//画像の表示
						mediaImage.ShowImage("ENEMY", eimg);
						mediaImage.MoveWindow("ENEMY", 640, 320);
					}
					else if (enemy==3) 
					{
						//グー
						//画像の読み込み
						mediaImage.ReadImage("../datas/gu.png", eimg);
						//画像の表示
						mediaImage.ShowImage("ENEMY", eimg);
						mediaImage.MoveWindow("ENEMY", 640, 320);
					}

					//勝敗の指示-----------------------------------------------------------------------------
					if (vod==1) 
					{
						//勝ち
						//画像の読み込み
						mediaImage.ReadImage("../datas/kuro.jpg", vodimg);
						//画像の表示ss
						cv::putText(vodimg, "Win", cv::Point(30, 130), cv::FONT_HERSHEY_COMPLEX, 2.5, cv::Scalar(255, 255, 255), 4);
						mediaImage.ShowImage("Victory", vodimg);
						mediaImage.MoveWindow("Victory", 640, 0);
					}
					else if (vod==2) 
					{
						//負け
						//画像の読み込み
						mediaImage.ReadImage("../datas/kuro.jpg", vodimg);
						//画像の表示
						cv::putText(vodimg, "Lose", cv::Point(30, 130), cv::FONT_HERSHEY_COMPLEX, 2.5, cv::Scalar(255, 255, 255), 4);
						mediaImage.ShowImage("Victory", vodimg);
						mediaImage.MoveWindow("Victory", 640, 0);
					}
					else if (vod==3) 
					{
						//あいこ
						//画像の読み込み
						mediaImage.ReadImage("../datas/kuro.jpg", vodimg);
						//画像の表示
						cv::putText(vodimg, "Draw", cv::Point(30, 130), cv::FONT_HERSHEY_COMPLEX, 2.5, cv::Scalar(255, 255, 255), 4);
						mediaImage.ShowImage("Victory", vodimg);
						mediaImage.MoveWindow("Victory", 640, 0);
					}

					//チョキ/勝て！--------------------------------------------------------------------------------
					
						if (enemy == 1 && vod == 1 && janken == "gu")
						{
							//画像の読み込み
							mediaImage.ReadImage("../datas/kuro.jpg", vodimg);
							//画像の表示
							cv::putText(vodimg, "ture", cv::Point(50, 130), cv::FONT_HERSHEY_COMPLEX, 2.5, cv::Scalar(255, 255, 255), 4);
							mediaImage.ShowImage("JUDGE", vodimg);
							mediaImage.MoveWindow("JUDGE", 960, 0);
						}

						if (enemy == 1 && vod == 1 && janken == "par" )
						{
							//画像の読み込み
							mediaImage.ReadImage("../datas/kuro.jpg", vodimg);
							//画像の表示
							cv::putText(vodimg, "false", cv::Point(50, 130), cv::FONT_HERSHEY_COMPLEX, 2.5, cv::Scalar(255, 255, 255), 4);
							mediaImage.ShowImage("JUDGE", vodimg);
							mediaImage.MoveWindow("JUDGE", 960, 0);
						}

						if ( enemy == 1 && vod == 1 && janken == "choki")
						{
							//画像の読み込み
							mediaImage.ReadImage("../datas/kuro.jpg", vodimg);
							//画像の表示
							cv::putText(vodimg, "false", cv::Point(50, 130), cv::FONT_HERSHEY_COMPLEX, 2.5, cv::Scalar(255, 255, 255), 4);
							mediaImage.ShowImage("JUDGE", vodimg);
							mediaImage.MoveWindow("JUDGE", 960, 0);
						}
					

					//チョキに負けろ！--------------------------------------------------------------------------------
				
						if (enemy == 1 && vod == 2 && janken == "par")
						{
							//画像の読み込み
							mediaImage.ReadImage("../datas/kuro.jpg", vodimg);
							//画像の表示
							cv::putText(vodimg, "ture", cv::Point(50, 130), cv::FONT_HERSHEY_COMPLEX, 2.5, cv::Scalar(255, 255, 255), 4);
							mediaImage.ShowImage("JUDGE", vodimg);
							mediaImage.MoveWindow("JUDGE", 960, 0);
						}

						if (enemy == 1 && vod == 2 && janken == "gu" )
						{
							//画像の読み込み
							mediaImage.ReadImage("../datas/kuro.jpg", vodimg);
							//画像の表示
							cv::putText(vodimg, "false", cv::Point(50, 130), cv::FONT_HERSHEY_COMPLEX, 2.5, cv::Scalar(255, 255, 255), 4);
							mediaImage.ShowImage("JUDGE", vodimg);
							mediaImage.MoveWindow("JUDGE", 960, 0);
						}

						if (enemy == 1 && vod == 2 && janken == "choki")
						{
							//画像の読み込み
							mediaImage.ReadImage("../datas/kuro.jpg", vodimg);
							//画像の表示
							cv::putText(vodimg, "false", cv::Point(50, 130), cv::FONT_HERSHEY_COMPLEX, 2.5, cv::Scalar(255, 255, 255), 4);
							mediaImage.ShowImage("JUDGE", vodimg);
							mediaImage.MoveWindow("JUDGE", 960, 0);
						}
					

					//チョキとあいこ！-----------------------------------------------------------------------------------
				
						if (enemy == 1 && vod == 3 && janken == "choki")
						{
							//画像の読み込み
							mediaImage.ReadImage("../datas/kuro.jpg", vodimg);
							//画像の表示
							cv::putText(vodimg, "ture", cv::Point(50, 130), cv::FONT_HERSHEY_COMPLEX, 2.5, cv::Scalar(255, 255, 255), 4);
							mediaImage.ShowImage("JUDGE", vodimg);
							mediaImage.MoveWindow("JUDGE", 960, 0);
						}

						if (enemy == 1 && vod == 3 && janken == "par" )
						{
							//画像の読み込み
							mediaImage.ReadImage("../datas/kuro.jpg", vodimg);
							//画像の表示
							cv::putText(vodimg, "false", cv::Point(50, 130), cv::FONT_HERSHEY_COMPLEX, 2.5, cv::Scalar(255, 255, 255), 4);
							mediaImage.ShowImage("JUDGE", vodimg);
							mediaImage.MoveWindow("JUDGE", 960, 0);
						}

						if ( enemy == 1 && vod == 3 && janken == "gu")
						{
							//画像の読み込み
							mediaImage.ReadImage("../datas/kuro.jpg", vodimg);
							//画像の表示
							cv::putText(vodimg, "false", cv::Point(50, 130), cv::FONT_HERSHEY_COMPLEX, 2.5, cv::Scalar(255, 255, 255), 4);
							mediaImage.ShowImage("JUDGE", vodimg);
							mediaImage.MoveWindow("JUDGE", 960, 0);
						}
					

					//パーに勝て！----------------------------------------------------------------------------------------
					
						if (enemy == 2 && vod == 1 && janken == "choki")
						{
							//画像の読み込み
							mediaImage.ReadImage("../datas/kuro.jpg", vodimg);
							//画像の表示
							cv::putText(vodimg, "ture", cv::Point(50, 130), cv::FONT_HERSHEY_COMPLEX, 2.5, cv::Scalar(255, 255, 255), 4);
							mediaImage.ShowImage("JUDGE", vodimg);
							mediaImage.MoveWindow("JUDGE", 960, 0);
						}

						if (enemy == 2 && vod == 1 && janken == "par" )
						{
							//画像の読み込み
							mediaImage.ReadImage("../datas/kuro.jpg", vodimg);
							//画像の表示
							cv::putText(vodimg, "false", cv::Point(50, 130), cv::FONT_HERSHEY_COMPLEX, 2.5, cv::Scalar(255, 255, 255), 4);
							mediaImage.ShowImage("JUDGE", vodimg);
							mediaImage.MoveWindow("JUDGE", 960, 0);
						}

						if ( enemy == 2 && vod == 1 && janken == "gu")
						{
							//画像の読み込み
							mediaImage.ReadImage("../datas/kuro.jpg", vodimg);
							//画像の表示
							cv::putText(vodimg, "false", cv::Point(50, 130), cv::FONT_HERSHEY_COMPLEX, 2.5, cv::Scalar(255, 255, 255), 4);
							mediaImage.ShowImage("JUDGE", vodimg);
							mediaImage.MoveWindow("JUDGE", 960, 0);
						}
					

					//パーに負けろ！--------------------------------------------------------------------------------------
					
						if (enemy == 2 && vod == 2 && janken == "gu")
						{
							//画像の読み込み
							mediaImage.ReadImage("../datas/kuro.jpg", vodimg);
							//画像の表示
							cv::putText(vodimg, "ture", cv::Point(50, 130), cv::FONT_HERSHEY_COMPLEX, 2.5, cv::Scalar(255, 255, 255), 4);
							mediaImage.ShowImage("JUDGE", vodimg);
							mediaImage.MoveWindow("JUDGE", 960, 0);
						}

						if (enemy == 2 && vod == 2 && janken == "par" )
						{
							//画像の読み込み
							mediaImage.ReadImage("../datas/kuro.jpg", vodimg);
							//画像の表示
							cv::putText(vodimg, "false", cv::Point(50, 130), cv::FONT_HERSHEY_COMPLEX, 2.5, cv::Scalar(255, 255, 255), 4);
							mediaImage.ShowImage("JUDGE", vodimg);
							mediaImage.MoveWindow("JUDGE", 960, 0);
						}

						if (enemy == 2 && vod == 2 && janken == "choki")
						{
							//画像の読み込み
							mediaImage.ReadImage("../datas/kuro.jpg", vodimg);
							//画像の表示
							cv::putText(vodimg, "false", cv::Point(50, 130), cv::FONT_HERSHEY_COMPLEX, 2.5, cv::Scalar(255, 255, 255), 4);
							mediaImage.ShowImage("JUDGE", vodimg);
							mediaImage.MoveWindow("JUDGE", 960, 0);
						}
					

					//パーとあいこ！--------------------------------------------------------------------------------------
			
						if (enemy == 2 && vod == 3 && janken == "par")
						{
							//画像の読み込み
							mediaImage.ReadImage("../datas/kuro.jpg", vodimg);
							//画像の表示
							cv::putText(vodimg, "ture", cv::Point(50, 130), cv::FONT_HERSHEY_COMPLEX, 2.5, cv::Scalar(255, 255, 255), 4);
							mediaImage.ShowImage("JUDGE", vodimg);
							mediaImage.MoveWindow("JUDGE", 960, 0);
						}

						if (enemy == 2 && vod == 3 && janken == "gu" )
						{
							//画像の読み込み
							mediaImage.ReadImage("../datas/kuro.jpg", vodimg);
							//画像の表示
							cv::putText(vodimg, "false", cv::Point(50, 130), cv::FONT_HERSHEY_COMPLEX, 2.5, cv::Scalar(255, 255, 255), 4);
							mediaImage.ShowImage("JUDGE", vodimg);
							mediaImage.MoveWindow("JUDGE", 960, 0);
						}

						if ( enemy == 2 && vod == 3 && janken == "choki")
						{
							//画像の読み込み
							mediaImage.ReadImage("../datas/kuro.jpg", vodimg);
							//画像の表示
							cv::putText(vodimg, "false", cv::Point(50, 130), cv::FONT_HERSHEY_COMPLEX, 2.5, cv::Scalar(255, 255, 255), 4);
							mediaImage.ShowImage("JUDGE", vodimg);
							mediaImage.MoveWindow("JUDGE", 960, 0);
						}
					

					//グーに勝て！----------------------------------------------------------------------------------------
					
						if (enemy == 3 && vod == 1 && janken == "par")
						{
							//画像の読み込み
							mediaImage.ReadImage("../datas/kuro.jpg", vodimg);
							//画像の表示
							cv::putText(vodimg, "ture", cv::Point(50, 130), cv::FONT_HERSHEY_COMPLEX, 2.5, cv::Scalar(255, 255, 255), 4);
							mediaImage.ShowImage("JUDGE", vodimg);
							mediaImage.MoveWindow("JUDGE", 960, 0);
						}

						if (enemy == 3 && vod == 1 && janken == "gu" )
						{
							//画像の読み込み
							mediaImage.ReadImage("../datas/kuro.jpg", vodimg);
							//画像の表示
							cv::putText(vodimg, "false", cv::Point(50, 130), cv::FONT_HERSHEY_COMPLEX, 2.5, cv::Scalar(255, 255, 255), 4);
							mediaImage.ShowImage("JUDGE", vodimg);
							mediaImage.MoveWindow("JUDGE", 960, 0);
						}

						if (enemy == 3 && vod == 1 && janken == "choki")
						{
							//画像の読み込み
							mediaImage.ReadImage("../datas/kuro.jpg", vodimg);
							//画像の表示
							cv::putText(vodimg, "false", cv::Point(50, 130), cv::FONT_HERSHEY_COMPLEX, 2.5, cv::Scalar(255, 255, 255), 4);
							mediaImage.ShowImage("JUDGE", vodimg);
							mediaImage.MoveWindow("JUDGE", 960, 0);
						}
					

					//グーに負けろ！--------------------------------------------------------------------------------------
					
						if (enemy == 3 && vod == 2 && janken == "choki")
						{
							//画像の読み込み
							mediaImage.ReadImage("../datas/kuro.jpg", vodimg);
							//画像の表示
							cv::putText(vodimg, "ture", cv::Point(50, 130), cv::FONT_HERSHEY_COMPLEX, 2.5, cv::Scalar(255, 255, 255), 4);
							mediaImage.ShowImage("JUDGE", vodimg);
							mediaImage.MoveWindow("JUDGE", 960, 0);
						}

						if (enemy == 3 && vod == 2 && janken == "par" )
						{
							//画像の読み込み
							mediaImage.ReadImage("../datas/kuro.jpg", vodimg);
							//画像の表示
							cv::putText(vodimg, "false", cv::Point(50, 130), cv::FONT_HERSHEY_COMPLEX, 2.5, cv::Scalar(255, 255, 255), 4);
							mediaImage.ShowImage("JUDGE", vodimg);
							mediaImage.MoveWindow("JUDGE", 960, 0);
						}

						if (enemy == 3 && vod == 2 && janken == "gu")
						{
							//画像の読み込み
							mediaImage.ReadImage("../datas/kuro.jpg", vodimg);
							//画像の表示
							cv::putText(vodimg, "false", cv::Point(50, 130), cv::FONT_HERSHEY_COMPLEX, 2.5, cv::Scalar(255, 255, 255), 4);
							mediaImage.ShowImage("JUDGE", vodimg);
							mediaImage.MoveWindow("JUDGE", 960, 0);
						}
					

					//グーとあいこ！-------------------------------------------------------------------------------------
					
						if (enemy == 3 && vod == 3 && janken == "gu")
						{
							//画像の読み込み
							mediaImage.ReadImage("../datas/kuro.jpg", vodimg);
							//画像の表示
							cv::putText(vodimg, "ture", cv::Point(50, 130), cv::FONT_HERSHEY_COMPLEX, 2.5, cv::Scalar(255, 255, 255), 4);
							mediaImage.ShowImage("JUDGE", vodimg);
							mediaImage.MoveWindow("JUDGE", 960, 0);
						}

						if (enemy == 3 && vod == 3 && janken == "parqq" )
						{
							//画像の読み込み
							mediaImage.ReadImage("../datas/kuro.jpg", vodimg);
							//画像の表示
							cv::putText(vodimg, "false", cv::Point(50, 130), cv::FONT_HERSHEY_COMPLEX, 2.5, cv::Scalar(255, 255, 255), 4);
							mediaImage.ShowImage("JUDGE", vodimg);
							mediaImage.MoveWindow("JUDGE", 960, 0);
						}

						if (enemy == 3 && vod == 3 && janken == "choki")
						{
							//画像の読み込み
							mediaImage.ReadImage("../datas/kuro.jpg", vodimg);
							//画像の表示
							cv::putText(vodimg, "false", cv::Point(50, 130), cv::FONT_HERSHEY_COMPLEX, 2.5, cv::Scalar(255, 255, 255), 4);
							mediaImage.ShowImage("JUDGE", vodimg);
							mediaImage.MoveWindow("JUDGE", 960, 0);
						}
					
					
				}
					
#endif		   	


#if FourthWeek
				
				resPath = "04res/";
				_mkdir(resPath.c_str());

				//特徴量モードを選択（color or hog）
				const std::string featureMode = "color";

				//画像データベースを読み込む
				std::map<int, std::string> labels;

				//データベースを作成したときのフォルダ名を記入
				std::string my_dir = "result_images";
				mediaImage.ReadFile(labels, "../datas/" + my_dir + "/label.txt");

				//ヒストグラム用の変数
				const int bin = 30; //ヒストグラムを作成するときの特徴ベクトルの次元数（色の場合1～180, HoGの場合は9）
				const int channel = 1; //色特徴量を扱う場合のみに1～3へ変更可能（彩度や明度の特徴を使いたいとき）

#if KNN_MODE
				//機械学習の変数および初期値の準備
				cv::Ptr<cv::ml::KNearest> k_nn;

				//k-NNのセットアップ
				const int k = 1;

				//学習結果の保存名
				stringstream ss; ss << k;
				const string saveFileName = resPath + "train_kNN_[" + featureMode + "]_(" + ss.str() + ").xml";
#else
				
				//機械学習の変数および初期値の準備
				cv::Ptr<cv::ml::SVM> svm;

				const double c = 10.0;
				const double gamma = 1.0;

				//学習結果の保存名
				stringstream ss1; ss1 << c;
				stringstream ss2; ss2 << gamma;
				const string saveFileName = resPath + "train_svm_[" + featureMode + "]_(" + ss1.str() + "-" + ss2.str() + ").xml";


#endif

				//学習データを読み込む
#if KNN_MODE
				k_nn = cv::Algorithm::load<cv::ml::KNearest>(saveFileName);
#else
				
				svm = cv::Algorithm::load<cv::ml::SVM>(saveFileName);
				
#endif
				//読み込みエラーは飛ばす
				if (img.empty())
				{
					cerr << "映像ファイルがありません。 \n";
					continue;
				}

				//画像サイズを統一する
				cv::Mat resize;
				cv::resize(img, resize, cv::Size(100, 100), cv::INTER_CUBIC);

				//色特徴量を取得する
				cv::Mat hist(1, bin * channel, CV_32FC1);

				if (featureMode == "color")
				{
					if (!mediaImage.GenerateColorHistogram(resize, hist, bin, channel))
					{
						cerr << "ヒストグラムが作成できませんでした\n";
						exit(1);
					}
				}
				else if (featureMode == "hog")
				{
					
					mediaImage.SetHoGParameters(9, 20, 1);
					mediaImage.CalcHOGHistgram(resize, hist); //HoG特徴量を抽出

				}

				//識別の評価を行う
				cv::Mat responce;

#if KNN_MODE
				k_nn->predict(hist, responce);
#else
				
				svm->predict(hist, responce);

#endif

				for (int i = 0; i < responce.rows; ++i)
				{
					cerr << labels[responce.at<float>(i, 0)] << "\n";
					cv::putText(img, labels[responce.at<float>(i, 0)], cv::Point(0, 40 * (i + 1)), cv::FONT_HERSHEY_COMPLEX, 1.5, cv::Scalar(0, 0, 0), 2);
				}
				
				mediaImage.ShowImage("Janken", img);
#endif
			}

		}

		//Video processing (xtion 2)
		else if (mode == "rgbd")
		{
			const bool kadai[] = { 1, 0 };


			//OpenNIの初期化
			mediaImage.InitOpenNI(640);

			
			while (1)
			{
				//whileの初めに必ず必要な1文
				mediaImage.UpDate();

				//RGB画像を出力
				cv::Mat image = mediaImage.GetImage();
				cv::Mat depth = mediaImage.GetDepth();
				mediaImage.ShowImage("COLOR", image);
				mediaImage.ShowImage("DEPTH", depth);
				mediaImage.MoveWindow("COLOR", 0, 0);
				mediaImage.MoveWindow("DEPTH", 400, 0);


#if FifthWeek
				
				resPath = "05res/";
				_mkdir(resPath.c_str());

				cv::Mat img = image.clone();

				if (kadai[0])
				{


					//骨格情報を取得する
					cv::Mat skeleton_img = image.clone();

					//骨格情報を取得し，画像に書き込む
					mediaImage.CreateSkeleton(skeleton_img);
					mediaImage.ShowImage("SKELETON", skeleton_img);
					mediaImage.MoveWindow("SKELETON", 0, 300);

					//骨格の座標点を取得する
					vector<vector<nite::Point3f> > userPoints = mediaImage.GetUserPoint();

					//検出されたユーザ番号を取得する
					vector <bool> checkUser = mediaImage.CheckUser();

					//検出されたユーザの3軸の値を出力する
					for (int j = 0; j < checkUser.size(); ++j)
					{
						if (checkUser[j])
						{
							for (int i = 0; i < userPoints.size(); ++i)
								cerr << skeleton_name[j] << ":\t" << j << "\t" << userPoints[j][i].x << "\t" << userPoints[j][i].y << "\t" << userPoints[j][i].z << endl;
							cerr << "\n\n\n";
						}
					}

					
				}
				else if (kadai[1])
				{

					//骨格情報を取得する
					cv::Mat skeleton_img = image.clone();

					//骨格情報を取得し，画像に書き込む
					mediaImage.CreateSkeleton(skeleton_img);
					mediaImage.ShowImage("SKELETON", skeleton_img);
					mediaImage.MoveWindow("SKELETON", 0, 300);

					//骨格の座標点を取得する
					vector<vector<nite::Point3f> > userPoints = mediaImage.GetUserPoint();

					//検出されたユーザ番号を取得する
					vector <bool> checkUser = mediaImage.CheckUser();
#if GESTURE	
#if NonCompulsion				
					//動作認識に関するプログラム
					//検出されたユーザに対して処理を行う
					for (int i = 0; i < checkUser.size(); ++i)
					{
						if (checkUser[i])
						{
							//左手を挙げていればOKを出力
							if (mediaImage.GetGestures(i) == MediaGesture::LEFT_HAND_UP_1
								|| mediaImage.GetGestures(i) == MediaGesture::LEFT_HAND_UP_2)
							{
								cerr << "OK\n\n\n";
							}

						}
					}
				}
#else
					//強制動作プログラムに関するプログラム
					//検出されたユーザに対して処理を行う
					for (int i = 0; i < checkUser.size(); ++i)
					{
						if (checkUser[i])
						{

							
							//左手が挙げるまで待機
							if (mediaImage.PleaseGesture(MediaGesture::LEFT_HAND_UP_2, -1, i))
							{
								cerr << "OK\n\n\n";
							}

						}
					}
#endif
			}
#else

					//動作認識に関するプログラム
					//検出されたユーザに対して処理を行う
					for (int i = 0; i < checkUser.size(); ++i)
					{
						if (checkUser[i])
						{
							
							//右手が挙げられているかどうか
							if (mediaImage.PleaseAction(MediaAction::BYE_BYE, -1, i))
							{
								cerr << "OK\n\n\n";
							}
						}
					}
				}
#endif
#endif
				//qキーで終了
				int key = cv::waitKey(1);
				if (key == 'q' || key == 'Q')
				{
					mediaImage.ReleaseImage();
					break;
				}
			}

		}
		else
		{
			throw("You must select 【image】or【camera】or【rgbd】.");
		}
	}
	catch (std::exception& ex)
	{
		std::cout << ex.what() << std::endl;
		mediaImage.Stop();
	}
	return 0;
}