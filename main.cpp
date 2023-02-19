
#include "MediaImage.hpp"

#define FirstWeek 0
#define SecondWeek 0
#define ThirdWeek 1
#define FourthWeek 0
#define LandR 1 //1�Ȃ�w�K�C0�Ȃ�F���iimage���[�h�̎��j
#define KNN_MODE 1 //KNN_MODE��1�̂Ƃ�k-NN�����C�����łȂ��ꍇ��SVM
#define FifthWeek 0
#define GESTURE 1 //Gesture���[�h�Ȃ�1�CAction���[�h�Ȃ�0
#define NonCompulsion 1 //�񓯊����[�h�Ȃ�1�C�������[�h�Ȃ�0

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

	const string mode = "camera"; //image, camera, rgbd�̂����ꂩ

	try
	{
		//�o�͌��ʂ�ۑ�����
		std::string resPath;

		//Image processing (image)
		if (mode == "image")
		{
			//�摜��ǂݍ��ނ��߂̕ϐ�
			cv::Mat img;

#if FirstWeek
			/*****************************/
			/** 1st week workshop       **/
			/**   -color filtering      **/
			/**   -labeling processing  **/
			/*****************************/
			//�摜�̓ǂݍ���
			mediaImage.ReadImage("../datas/colorpen.jpg", img);

			//�摜�̕\��
			mediaImage.ShowImage("color", img);

			//�L�[���͑҂�
			cv::waitKey(0);

			resPath = "01res/";
			_mkdir(resPath.c_str());

			//���o�����F�摜���i�[����ϐ�
			cv::Mat extractedColor;

			//���o����F���w�肷��B
			mediaImage.ExtractColor(175, 5, 50, 255, 0, 255, img, extractedColor);
			mediaImage.ShowImage("Extracted_Color_image", extractedColor);

			//���������摜��ۑ�����
			mediaImage.SaveImage(extractedColor, resPath + "red", ".jpg");


			cv::Mat blue_image;
			mediaImage.ExtractColor(110, 130, 50, 255, 0, 255, img, blue_image);
			mediaImage.ShowImage("Blue", blue_image);
			mediaImage.SaveImage(blue_image, resPath + "blue", ".jpg");

			cv::Mat green_image;
			mediaImage.ExtractColor(60, 80, 50, 255, 0, 255, img, green_image);
			mediaImage.ShowImage("Green", green_image);
			mediaImage.SaveImage(green_image, resPath + "green", ".jpg");
	
			cv::waitKey(0);//�L�[���͑҂�

			//�o��Window���폜����
			mediaImage.ReleaseWindow("Extracted_Color_image");
			mediaImage.ReleaseWindow("Green");
			mediaImage.ReleaseWindow("Blue");


			//�m�C�Y����
			cv::Mat temp1 = extractedColor.clone(), temp2 = extractedColor.clone();
			mediaImage.Morphology(extractedColor, extractedColor, "op", 2);
			mediaImage.ShowImage("Morphology_image", extractedColor);

			//���������摜��ۑ�����
			mediaImage.SaveImage(extractedColor, resPath + "res_op_2", ".jpg");


			
			mediaImage.Morphology(temp1, temp1, "cl", 2);
			mediaImage.ShowImage("Morphology_image", temp1);
			mediaImage.SaveImage(temp1, resPath + "res_cl_2", ".jpg");

			mediaImage.Morphology(temp2, temp2, "cl", 15);
			mediaImage.ShowImage("Morphology_image", temp2);
			mediaImage.SaveImage(temp2, resPath + "res_cl_15", ".jpg");


			
			cv::waitKey(0);

			//�o��Window���폜����
			mediaImage.ReleaseWindow("Morphology_image");


			//���x�����O����
			//���x�����O�����ɕK�v�ȕϐ�
			cv::Mat LabelImg; //���x�����O���ꂽ�摜
			cv::Mat stats; //���x���̌��ʂ��i�[��
			cv::Mat centroids; //�d�S�̌��ʂ��i�[
			int nLab; //���x�����O�����i�[
			cv::Mat resLabel; //���x�����O�`�挋�ʂ��i�[

			mediaImage.LabelingProcessing(extractedColor, LabelImg, stats, centroids, nLab);
			mediaImage.DrawLabeling(extractedColor, LabelImg, stats, centroids, nLab, resLabel);
			mediaImage.ShowImage("Labeling", resLabel);

			//���������摜��ۑ�����
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
			//�o��Window���폜����
			mediaImage.ReleaseWindow("Labeling");


			//�G�b�W����
			cv::Mat edge_img; //�G�b�W�摜���i�[����
			mediaImage.EdgeDetection(extractedColor, edge_img, "canny");
			mediaImage.ShowImage("EdgeImage", edge_img);

			//���������摜��ۑ�����
			mediaImage.SaveImage(edge_img, resPath + "canny", ".jpg");

			
			cv::Mat res_lap, res_sol;
			mediaImage.EdgeDetection(extractedColor, res_lap, "laplacian");
			mediaImage.ShowImage("res_lap", res_lap);
			mediaImage.SaveImage(res_lap, resPath + "laplacian", ".jpg");

			mediaImage.EdgeDetection(extractedColor, res_sol, "sobel");
			mediaImage.ShowImage("res_sol", res_sol);
			mediaImage.SaveImage(res_sol, resPath + "sobel", ".jpg");

			

			cv::waitKey(0);

			//�o��Window���폜����
			mediaImage.ReleaseWindow("EdgeImage");


			//HoG������
			cv::Mat hog_hist, hog_result;
			mediaImage.SetHoGParameters(9, 20, 1);
			mediaImage.CalcHOGHistgram(img, hog_hist); //HoG�����ʂ𒊏o
			mediaImage.GetHoGOnImage(img, hog_result); //HoG�����ʂ̕`�悵���摜���擾
			std::cerr << "HoG features\n"
				<< cv::format(hog_hist, cv::Formatter::FMT_CSV) << "\n";
			mediaImage.ShowImage("HoGImage", hog_result);

			//���������摜��ۑ�����
			mediaImage.SaveImage(hog_result, resPath + "res_hog", ".jpg");

			cv::Mat res_hog1, res_hog2;

			mediaImage.SetHoGParameters(9, 10, 1);
			mediaImage.CalcHOGHistgram(img, hog_hist);
			mediaImage.GetHoGOnImage(img, res_hog1);
			std::cerr << "HoG featurs\n"
				<< cv::format(hog_hist, cv::Formatter::FMT_CSV) << "\n";
			mediaImage.ShowImage("HoGImage1", res_hog1);

			//���������摜��ۑ�����
			mediaImage.SaveImage(res_hog1, resPath + "hog1", ".jpg");

			mediaImage.SetHoGParameters(9, 20, 5);
			mediaImage.CalcHOGHistgram(img, hog_hist);
			mediaImage.GetHoGOnImage(img, res_hog2);
			std::cerr << "HoG featurs\n"
				<< cv::format(hog_hist, cv::Formatter::FMT_CSV) << "\n";
			mediaImage.ShowImage("HoGImage2", res_hog2);

			mediaImage.SaveImage(res_hog2, resPath + "hog2", ".jpg");

			cv::waitKey(0);
			//�o��Window���폜����
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

			//�摜��ǂݍ���
			cv::Mat janken_img, janken_res;
			
			mediaImage.ReadImage("../datas/sample.png", janken_img);
				//roi_image�ɑ΂��āA���F���o���s��
			mediaImage.ExtractColor(150, 50, 1, 255, 0, 255, janken_img, janken_res);
			mediaImage.ShowImage("janken_res", janken_res);

			//�I�[�v�j���O��N���[�W���O��K���ȉ񐔌J��Ԃ��ăm�C�Y����������
			mediaImage.Morphology(janken_res, janken_res, "op", 3);
			mediaImage.Morphology(janken_res, janken_res, "cl", 3);
			mediaImage.ShowImage("opening jenken_res", janken_res);
			//���F���o��A���x�����O�������s���A���F�̖ʐς����߂�B
			mediaImage.LabelingProcessing(janken_res, LabelImg, stats, centroids, nLab);
			mediaImage.DrawLabeling(janken_res, LabelImg, stats, centroids, nLab, resLabel);
			mediaImage.ShowImage("Labeling", resLabel);

			//���o�����̈�̖ʐς��擾���邱�Ƃ��ł���B
			std::vector<unsigned int> area = mediaImage.GetSquare();

			//���F�̈�̍ő�l�����߂�
			int max = 0;
			for (int i = 0; i < area.size(); ++i)
			{
				if (max < area[i])
					max = area[i];
			}
			//���F�̖ʐς̒l���R���\�[����ɕ\��
			//����ŃO�[�A�`���L�A�p�[��臒l�����߂�
			std::cout << max << std::endl;

			//�ő�l�imax�j�̒l�ɂ��O�[�A�`���L�A�p�[�̂����ꂩ�ɂ���
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

			//�摜�̓ǂݍ���
			mediaImage.ReadImage("../datas/faces2.jpg", img);

			//�摜�̕\��
			mediaImage.ShowImage("face", img);
			mediaImage.MoveWindow("face", 0, 0);

			//�L�[���͑҂�
			cv::waitKey(0);

			resPath = "02res/";
			_mkdir(resPath.c_str());

			//�J�X�P�[�h�摜��I������
			std::string cascade = "../datas/asm/haarcascade_frontalface_alt2.xml";
			std::string eyes_cascade = "../datas/asm/haarcascade_eye.xml";
			std::vector<cv::Rect_<int>> faces;

			//�猟�o
			mediaImage.GetFaceFeatures(img, faces, cascade);
			cv::Mat facial_image = img.clone();

			//���o������Ȃǂ̏ꏊ���͂�
			mediaImage.DrawFeatures(facial_image, faces);

			//�摜�̕\��
			mediaImage.ShowImage("Detection_Face", facial_image);
			mediaImage.MoveWindow("Detection_Face", 700, 0);

			//���������摜��ۑ�����
			mediaImage.SaveImage(facial_image, resPath + "face", ".jpg");

			//���ڌ��o ---------------------------------------------------
			std::string eyes = "../datas/asm/haarcascade_eye.xml";
			mediaImage.GetFaceFeatures(img, faces, eyes);
			cv::Mat eye_image = img.clone();

			//���o������Ȃǂ̏ꏊ���͂�
			mediaImage.DrawFeatures(eye_image, faces);

			//�摜�̕\��
			mediaImage.ShowImage("Detection_Eyes", eye_image);
			mediaImage.MoveWindow("Detection_Eyes", 0, 240);

			//�����摜��ۑ�����
			mediaImage.SaveImage(eye_image, resPath + "eyes", ".jpg");
			
			//�E�ڌ��o ---------------------------------------------------
			std::string right_eye = "../datas/asm/haarcascade_righteye_2splits.xml";
			mediaImage.GetFaceFeatures(img, faces, right_eye);
			cv::Mat right_image = img.clone();

			//���o������Ȃǂ̏ꏊ���͂�

			mediaImage.DrawFeatures(right_image, faces);

			//�摜�̕\��

			mediaImage.ShowImage("Detection_EyeR", right_image);
			mediaImage.MoveWindow("Detection_EyeR", 700, 240);

			//�����摜��ۑ�����
			mediaImage.SaveImage(right_image, resPath + "right_eye", ".jpg");

			//���ڌ��o ---------------------------------------------------
			std::string left_eye = "../datas/asm/haarcascade_lefteye_2splits.xml";
			mediaImage.GetFaceFeatures(img, faces, left_eye);
			cv::Mat left_image = img.clone();

			//���o������Ȃǂ̏ꏊ���͂�

			mediaImage.DrawFeatures(left_image, faces);

			//�摜�̕\��

			mediaImage.ShowImage("Detection_EyeR", left_image);
			mediaImage.MoveWindow("Detection_EyeR", 0, 480);

			//�����摜��ۑ�����
			mediaImage.SaveImage(left_image, resPath + "left_eye", ".jpg");

			//���K�l���茟�o ---------------------------------------------------
			std::string eye_with_glass = "../datas/asm/haarcascade_eye_tree_eyeglasses.xml";
			mediaImage.GetFaceFeatures(img, faces, eye_with_glass);
			cv::Mat glass_image = img.clone();

			//���o������Ȃǂ̏ꏊ���͂�

			mediaImage.DrawFeatures(glass_image, faces);

			//�摜�̕\��

			mediaImage.ShowImage("Detection_EyeR", glass_image);
			mediaImage.MoveWindow("Detection_EyeR", 700, 480);

			//�����摜��ۑ�����
			mediaImage.SaveImage(glass_image, resPath + "glass", ".jpg");

			cv::waitKey(0);
			//�o��Window���폜����
			mediaImage.ReleaseWindow("Detection_Face");
			mediaImage.ReleaseWindow("Detection_Eyes");
			mediaImage.ReleaseWindow("Detection_EyeR");
			mediaImage.ReleaseWindow("Detection_EyeL");
			mediaImage.ReleaseWindow("Detection_EyeG");

			//�J�X�P�[�h�摜��I������
			std::vector<cv::Rect_<int>> eyes2;
			cv::Mat eyes_image2 = img.clone();

			//��̌��o���Ă���ڂ̌��o���s��
			mediaImage.GetFaceFeatures(eyes_image2, faces, eyes2, eyes_cascade);

			//���o�����ꏊ���͂�
			mediaImage.DrawFeatures(eyes_image2, eyes2);

			//���������摜��\������
			mediaImage.ShowImage("Detection_EyesInFace", eyes_image2);
			mediaImage.MoveWindow("Detection_EyesInFace", 700, 0);

			//���������摜��ۑ�����
			mediaImage.SaveImage(eyes_image2, resPath + "eyes_in_face", ".jpg");

			cv::waitKey(0);
			//�o��Window���폜����
			mediaImage.ReleaseWindow("Detection_EyesInFace");


			//�p�����[�^�𒲐�����
			//�J�X�P�[�h�摜��I������
			std::vector<cv::Rect_<int>> faces3, faces4;
			cv::Mat face3 = img.clone();
			cv::Mat face4 = img.clone();

			//���������������ꍇ2.0
			mediaImage.GetFaceFeatures(face3, faces3, cascade, 2.0);

			//�����������Ⴂ�ꍇ1.001
			mediaImage.GetFaceFeatures(face4, faces4, cascade, 3.001);

			//���o�����ꏊ���͂�
			mediaImage.DrawFeatures(face3, faces3);
			mediaImage.DrawFeatures(face4, faces4);

			//���������摜��\������
			mediaImage.ShowImage("Detection_Face3", face3);
			mediaImage.MoveWindow("Detection_Face3", 0, 350);
			mediaImage.ShowImage("Detection_Face4", face4);
			mediaImage.MoveWindow("Detection_Face4", 700, 350);

			//���������摜��ۑ�����
			mediaImage.SaveImage(face3, resPath + "face(2.0)", ".jpg");
			mediaImage.SaveImage(face4, resPath + "face(1.001)", ".jpg");

			cv::waitKey(0);
			//�o��Window���폜����
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

			//�猟�o
			mediaImage.GetFaceFeatures(img, faces, cascade, 1.02);
			cv::Mat facial_image3 = img.clone();

			//���o������Ȃǂ̏ꏊ���͂�
			mediaImage.DrawFeatures(facial_image3, faces);

			//�摜�̕\��
			mediaImage.ShowImage("Detection_Face", facial_image3);
			mediaImage.MoveWindow("Detection_Face", 700, 0);
			cv::waitKey(0);

			//�J�X�P�[�h������I������

			std::string eyes3 = "../datas/asm/haarcascade_eye.xml";
			std::vector<cv::Rect_<int>>rect_eyes3;
			cv::Mat eyes_image3 = img.clone();

			//��̌��o�����Ă���ڂ̌��o���s��
			mediaImage.GetFaceFeatures(eyes_image3, faces, rect_eyes3, eyes3, 1.07);
			mediaImage.DrawFeatures(eyes_image3, rect_eyes3);
			
			//�摜�̕\��
			mediaImage.ShowImage("Detection_EyesinFace", eyes_image3);
			mediaImage.MoveWindow("Detection_EyesinFace", 0, 300);

			//�����摜��ۑ�����
			mediaImage.SaveImage(eyes_image3, resPath + "eyes_(1.07)", ".jpg");

			//�J�X�P�[�h������I������
			std::string glass3 = "../datas/asm/haarcascade_eye_tree_eyeglasses.xml";
			std::vector<cv::Rect_<int>>rect_glass3;
			cv::Mat glass_image3 = img.clone();

			//��̌��o�����Ă���ڂ̌��o���s��
			mediaImage.GetFaceFeatures(glass_image3, faces, rect_glass3, glass3, 1.07);
			mediaImage.DrawFeatures(glass_image3, rect_glass3);

			//�摜�̕\��
			mediaImage.ShowImage("Detection_GlassinFace", glass_image3);
			mediaImage.MoveWindow("Detection_GlassinFace", 0, 300);

			//�����摜��ۑ�����
			mediaImage.SaveImage(glass_image3, resPath + "glass_(1.07)", ".jpg");


			//�Ί�̌��o
			//�摜�̓ǂݍ���
			mediaImage.ReadImage("../datas/smile3.jpg", img);

			//�J�X�P�[�h�摜��I������
			std::string cascade_simle = "../datas/asm/haarcascade_smile.xml";
			std::vector<cv::Rect_<int>> smile;

			//�猟�o�ƏΊ猟�o
			mediaImage.GetFaceFeatures(img, faces, cascade, 1.01);
			mediaImage.GetFaceFeatures(img, faces, smile, cascade_simle);
			cv::Mat smile_image = img.clone();

			//���o������Ȃǂ̏ꏊ���͂�
			mediaImage.DrawFeatures(smile_image, faces);
			mediaImage.DrawFeatures(smile_image, smile);

			//�摜�̕\��
			mediaImage.ShowImage("Simle_Face", smile_image);
			mediaImage.MoveWindow("Simle_Face", 0, 0);

			//���������摜��ۑ�����
			mediaImage.SaveImage(smile_image, resPath + "smile", ".jpg");

			cv::waitKey(0);
			

			//�Ί�̃��x����]������
			vector<int> count_parts = mediaImage.GetDetectedCount();
			std::vector<float> scores;
			mediaImage.EvaluateSmile(count_parts, scores);

			//�Ί�x�ɉ����ĕ`�悷��
			cv::Mat evaluated_smile_img = img.clone();
			mediaImage.DrawFeatures(evaluated_smile_img, faces, scores);
			//�摜�̕\��
			mediaImage.ShowImage("Evaluated_Simle_Face", evaluated_smile_img);
			mediaImage.MoveWindow("Evaluated_Simle_Face", 0, 450);
			cv::waitKey(0);

			//���������摜��ۑ�����
			mediaImage.SaveImage(smile_image, resPath + "evaluated_smile", ".jpg");		

			//�o��Window���폜����
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

				//���o������Ȃǂ̏ꏊ���͂�
				mediaImage.DrawFeatures(smile_image, faces);
				mediaImage.DrawFeatures(smile_image, smile);

				//�摜�̕\��
				mediaImage.ShowImage("Simle_Face", smile_image);
				mediaImage.MoveWindow("Simle_Face", 0, 0);

				//���������摜��ۑ�����
				mediaImage.SaveImage(smile_image, resPath + "smile_(" + ss.str() + ")", ".jpg");

				//�Ί�̃��x����]������
				vector<int> count_parts = mediaImage.GetDetectedCount();
				std::vector<float> scores;
				mediaImage.EvaluateSmile(count_parts, scores);

				//�Ί�x�ɉ����ĕ`�悷��
				cv::Mat evaluated_smile_img = img.clone();
				mediaImage.DrawFeatures(evaluated_smile_img, faces,scores);

				//�摜�̕\��
				mediaImage.ShowImage("Evaluated_Simle_Face", evaluated_smile_img);
				mediaImage.MoveWindow("Evaluated_Simle_Face", 0, 450);

				//���������摜��ۑ�����
				mediaImage.SaveImage(smile_image, resPath + "evaluated_smile_(" + ss.str() + ")", ".jpg");

			}

			cv::waitKey(0);
			mediaImage.ReleaseWindow("Simle_Face");
			mediaImage.ReleaseWindow("Evaluated_Simle_Face")



			// ------------ �l�̌��o -----------------
			//�摜�̓ǂݍ���
			mediaImage.ReadImage("../datas/people.jpg", img);

			//�摜�̕\��
			mediaImage.ShowImage("People", img);
			mediaImage.MoveWindow("People", 0, 0);

			//�L�[���͑҂�
			cv::waitKey(0);

			//�l���o�p�̕ϐ�
			cv::HOGDescriptor hog;
			cv::Mat gray_img;
			std::vector<cv::Rect> human;
			
			//�l���o���̓O���[�摜�ɕϊ�����K�v����
			cv::cvtColor(img, gray_img, CV_BGR2GRAY);

			//�l�����o����
			mediaImage.GetHumanFeatures(gray_img, hog, human, 8, 16, 1.1);

			//�`�悷��
			cv::Mat resHuman = img.clone();
			mediaImage.DrawFeatures(resHuman, human);
						
			//�摜�̕\��
			mediaImage.ShowImage("Detected People", resHuman);
			mediaImage.MoveWindow("Detected People", 700, 0);
			mediaImage.SaveImage(resHuman, resPath + "Human(1.1)", ".jpg");
			cv::waitKey(0);

		for (float alpha = 1.01; alpha < 1.101; alpha += 0.01) {
			std::cout << to_string(alpha) << std::endl;
			std::stringstream ss;
			ss << to_string(alpha);

			//�l�����o����
			mediaImage.Restart();
			mediaImage.GetHumanFeatures(gray_img, hog, human, 8, 16, alpha);
			mediaImage.ProcessTime();

			//�`�悷��
			cv::Mat resHuman = img.clone();
			mediaImage.DrawFeatures(resHuman, human);

			//�摜�\��
			mediaImage.ShowImage("Detected People", resHuman);
			mediaImage.MoveWindow("Detectes People", 700, 0);
			mediaImage.SaveImage(resHuman, resPath + "Human(" + ss.str() + ")", ".jpg");
			}
			//�X���C�h��1�ɂ���ƌ��o���x�͂����邪���x�͒x��
			mediaImage.GetHumanFeatures(gray_img, hog, human, 1, 16, 1.1);

			//�`�悷��
			cv::Mat resHuman5 = img.clone();
			mediaImage.DrawFeatures(resHuman5, human);

			//�摜�\��
			mediaImage.ShowImage("Detected People with different parameter", resHuman5);
			mediaImage.MoveWindow("Detectes People with different parameter", 700, 350);
			mediaImage.SaveImage(resHuman5, resPath + "Human(1.1)_stride(1)", ".jpg");

			cv::waitKey(0);
			mediaImage.ReleaseWindow("Detected People");
			mediaImage.ReleaseWindow("Detected People with different parameter");
		
			std::vector<cv::Rect_<int>> faces6;
			cv::Mat face_image6 = img.clone();

			//�l�̌��o�����Ă����̌��o
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

			//�����ʃ��[�h��I���icolor or hog�j
			const std::string featureMode = "color";

			//�摜�f�[�^�x�[�X��ǂݍ���
			std::map<int, std::string> labels;

			//�f�[�^�x�[�X���쐬�����Ƃ��̃t�H���_�����L��
			std::string my_dir = "result_images";
			mediaImage.ReadFile(labels, "../datas/" + my_dir + "/label.txt");

			//�F���������摜��ǂݍ���
			img = cv::imread("../datas/sample2.png");

			//�q�X�g�O�����p�̕ϐ�
			const int bin = 30; //�q�X�g�O�������쐬����Ƃ��̓����x�N�g���̎������i�F�̏ꍇ1�`180, HoG�̏ꍇ��9�j
			const int channel = 1; //�F�����ʂ������ꍇ�݂̂�1�`3�֕ύX�\�i�ʓx�▾�x�̓������g�������Ƃ��j

#if KNN_MODE
			//�@�B�w�K�̕ϐ�����я����l�̏���
			cv::Ptr<cv::ml::KNearest> k_nn;

			//k-NN�̃Z�b�g�A�b�v
			const int k = 1;

			//�w�K���ʂ̕ۑ���
			stringstream ss; ss << k;
			const string saveFileName = resPath + "train_kNN_[" + featureMode + "]_(" + ss.str() + ").xml";
#else
			//�@�B�w�K�̕ϐ�����я����l�̏���
			cv::Ptr<cv::ml::SVM> svm;

			const double c = 1.0;
			const double gamma = 1.0;

			//�w�K���ʂ̕ۑ���
			stringstream ss1; ss1 << c;
			stringstream ss2; ss2 << gamma;
			const string saveFileName = resPath + "train_svm_[" + featureMode + "]_(" + ss1.str() + "-" + ss2.str() + ").xml";


#endif
#if LandR
			//----- Leaning Mode ------
			//�摜�f�[�^�x�[�X��ǂݍ���
			std::vector<std::string> files;
			mediaImage.ReadFile(files, "../datas/" + my_dir + "/database.txt");

			//�w�K���[�h�̏���
			cv::Mat histograms(files.size(), bin * channel, CV_32FC1);
			cv::Mat label(files.size(), 1, CV_32SC1);

			//�w�K�f�[�^�̍쐬������
			for (int i = 0; i < files.size(); ++i)
			{
				//�摜��ǂݍ���
				cv::Mat img = cv::imread("../datas/" + files[i]);

				//�ǂݍ��݃G���[�͔�΂�
				if (img.empty())
				{
					cerr << files[i] << "������܂���B \n";
					continue;
				}

				//���x�����쐬����
				for (int l = 0; l < labels.size(); ++l)
				{
					if (files[i].find(labels[l].c_str()) != std::string::npos)
					{
						label.at<int>(i, 0) = l;
					}
				}

				//�F�����ʂ��擾����
				cv::Mat hist = histograms.row(i);

				if (featureMode == "color")
				{
					if (!mediaImage.GenerateColorHistogram(img, hist, bin, channel))
						continue;
				}
				else if (featureMode == "hog")
				{
					
					mediaImage.SetHoGParameters(9, 20, 1);
					mediaImage.CalcHOGHistgram(img, hist); //HoG�����ʂ𒊏o

				}
			}

#if KNN_MODE
			//k-NN�̃Z�b�g�A�b�v
			//k = 1, �A���S���Y����BRUTE���f�t�H���g
			mediaImage.kNN(k);
			mediaImage.Train(histograms, label, k_nn);

			//�w�K���ʂ̕ۑ�
			cerr << "Saving the classifier to " << saveFileName << "\n";
			k_nn->save(saveFileName);
#else
			
			mediaImage.SupperVctorMachine("C_SVC", "LINEAR", c, 1.2, gamma, 1.0);
			mediaImage.Train(histograms, label, svm);

			//�w�K���ʂ̕ۑ�
			cerr << "Saving the classifier to " << saveFileName << "\n";
			svm->save(saveFileName);

#endif

#else
			//----- Recognition Mode ------
			//�w�K�f�[�^��ǂݍ���
#if KNN_MODE
			k_nn = cv::Algorithm::load<cv::ml::KNearest>(saveFileName);
#else
			
			svm = cv::Algorithm::load<cv::ml::SVM>(saveFileName);

#endif

			//�F���������摜��ǂݍ���
			img = cv::imread("../datas/sample2.png");
			cv::Mat labelTest(1, 1, CV_32SC1);

			cerr << "���x���ԍ�����͂��Ă�������: ";
			cin >> labelTest.at<int>(0, 0);

			//�ǂݍ��݃G���[�͔�΂�
			if (img.empty())
			{
				cerr << "�摜�t�@�C��������܂���B \n";
			}

			//�摜�T�C�Y�𓝈ꂷ��
			cv::Mat resize;
			cv::resize(img, resize, cv::Size(100, 100), cv::INTER_CUBIC);

			//�F�����ʂ��擾����
			cv::Mat hist(1, bin * channel, CV_32FC1);

			if (featureMode == "color")
			{
				if (!mediaImage.GenerateColorHistogram(resize, hist, bin, channel))
				{
					cerr << "�q�X�g�O�������쐬�ł��܂���ł���\n";
					exit(1);
				}
			}
			else if (featureMode == "hog")
			{
				
				mediaImage.SetHoGParameters(9, 20, 1);
				mediaImage.CalcHOGHistgram(resize, hist); //HoG�����ʂ𒊏o

			}
			cerr << hist << "\n";

			//���ʂ̕]�����s��
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
			//�J�����f�t�H���g�̐ݒ�B320�~240px�ƂȂ��Ă���B
			mediaImage.InitVideo(0, 320, 240); 

#if ThirdWeek
			/*****************************/
			/** 3rd week workshop       **/
			/**   -video processing     **/
			/*****************************/
			const bool kadai[] = { 0, 0, 0, 0, 0, 1 };

			//���O�ɔw�i���擾����D
			static cv::Mat background;

			//�t���[���ԍ����p
			cv::Mat old;

			//�o�b�N�O���E���h�̎擾����܂ŁC�������s���D
			while (1 && kadai[2]|| 1 && kadai[5])
			{
				//�摜���A�b�v�f�[�g����
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
			//�����_���̊֐�-------------------------------------------------------------
			random_device rd;
			mt19937 gen(rd());
			uniform_int_distribution<> 	dist(1, 3);

			//����̎�
			int enemy = dist(gen);
			cv::Mat eimg;
			//���s
			int vod = dist(gen);
			cv::Mat vodimg;

			while (1)
			{
				//�摜���A�b�v�f�[�g����
				mediaImage.UpdateVideo();
				cv::Mat img = mediaImage.GetImage();
				int key = cv::waitKey(1);

				//�r�f�I���I����������q�L�[���������ΏI��
				if (img.empty() || key == 'q' || key == 'Q')
				{
					mediaImage.ReleaseWindow();
					break;
				}

#if ThirdWeek
				resPath = "03res/";
				_mkdir(resPath.c_str());

				//���̉摜��\������
				mediaImage.ShowImage("Camera", img);
				mediaImage.MoveWindow("Camera", 0, 0);

				
				if (kadai[0])
				{
					mediaImage.ExtractColor(170, 20, 20, 255, 10, 255, img, img);
					mediaImage.Morphology(img, img, "op", 5);
					mediaImage.Morphology(img, img, "cl", 5);

					//���x�����O�����ɕK�v�ȕϐ�
					cv::Mat LabelImg; //���x�����O���ꂽ�摜
					cv::Mat stats; //���x���̌��ʂ��i�[
					cv::Mat centroids; //�d�S�̌��ʂ��i�[
					int nLab; //���x�����O�����i�[
					cv::Mat resLabel; //���x�����O�`�挋�ʂ��i�[

									  //���F���o��A���x�����O�������s���A���F�̖ʐς����߂�
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

					//�猟�o�ƏΊ猟�o
					mediaImage.GetFaceFeatures(img, faces, cascade, 1.1);
					mediaImage.GetFaceFeatures(img, faces, smile, cascade_simle, 1.1);

					vector<int> count_parts = mediaImage.GetDetectedCount();
					std::vector<float> scores;
					mediaImage.EvaluateSmile(count_parts, scores);

					//�Ί�x�ɉ����ĕ`�悷��
					cv::Mat evaluated_smile_img = img.clone();
					mediaImage.DrawFeatures(img, faces, scores);
					//�摜�̕\��
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
					//----- �w�i�����@�f�t�H���g�v���O����

					//�w�i�摜��\������
					mediaImage.ShowImage("BACKGROUND", background);
					mediaImage.MoveWindow("BACKGROUND", 320, 0);
					
					/*-------------------------- �ۑ�3 ��������-----------------------------*/
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
					//�ŏ��̃t���[���̏ꍇ�̓t���[���ԍ������Ȃ��悤�ɂ���D
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
					//�ŏ��̃t���[���̏ꍇ�̓t���[���ԍ������Ȃ��悤�ɂ���D
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

					//�I�v�e�B�J���t���[���v�Z����
					mediaImage.CalclateOpticalFlow(current, old, flow, points, 50);

					//�I�v�e�B�J���t���[��\������
					mediaImage.ShowOpticalFlow(points, flow, res);

					mediaImage.ShowImage("OPTICAL", res);
					mediaImage.MoveWindow("OPTICAL", 0, 300);
					old = img.clone();
					
					//�I�v�e�B�J���t���[���v�Z����
					mediaImage.CalclateOpticalFlow(current, old, flow, points, 10); //�l��Ⴆ��10�ɕύX

					//�I�v�e�B�J���t���[��\������

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
					

					//���F���o
					mediaImage.ExtractColor(150, 50, 1, 255, 0, 255, result, result);

					//�I�[�v�j���O�N���[�W���O�Ńm�C�Y����
					mediaImage.Morphology(result, result, "op", 3);
					mediaImage.Morphology(result, result, "cl", 3);

					//���x�����O�����ɕK�v�ȕϐ�
					cv::Mat LabelImg; //���x�����O���ꂽ�摜
					cv::Mat stats; //���x���̌��ʂ��i�[��
					cv::Mat centroids; //�d�S�̌��ʂ��i�[
					int nLab; //���x�����O�����i�[
					cv::Mat resLabel; //���x�����O�`�挋�ʂ��i�[

					//���F���o��A���x�����O�������s���A���F�̖ʐς����߂�B
					mediaImage.LabelingProcessing(result, LabelImg, stats, centroids, nLab);
					mediaImage.DrawLabeling(result, LabelImg, stats, centroids, nLab, resLabel);
					mediaImage.ShowImage("Labeling", resLabel);
					mediaImage.MoveWindow("Labeling", 320,0);
					
					//���o�����̈�̖ʐς��擾���邱�Ƃ��ł���B
					std::vector<unsigned int> area = mediaImage.GetSquare();
					//���F�̈�̍ő�l�����߂�s
					int max = 0;
					for (int i = 0; i < area.size(); ++i)
					{
						if (max < area[i])
							max = area[i];
					}
					//�ő�l�imax�j�̒l�ɂ��O�[�A�`���L�A�p�[�̂����ꂩ�ɂ���
					std::string janken = "None";
					if (5000 < max && max < 9000) //臒l�͉摜�Ɏʂ��̑傫���ɑ傫���ˑ�����
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

					//����̕\��----------------------------------------------------------------------
					if (enemy==1) 
					{
						//�`���L
						//�摜�̓ǂݍ���
						mediaImage.ReadImage("../datas/choki.png", eimg);
						//�摜�̕\��
						mediaImage.ShowImage("ENEMY", eimg);
						mediaImage.MoveWindow("ENEMY", 640, 320);
					}
					else if (enemy == 2) 
					{
						//�p�[
						//�摜�̓ǂݍ���
						mediaImage.ReadImage("../datas/par.png", eimg);
						//�摜�̕\��
						mediaImage.ShowImage("ENEMY", eimg);
						mediaImage.MoveWindow("ENEMY", 640, 320);
					}
					else if (enemy==3) 
					{
						//�O�[
						//�摜�̓ǂݍ���
						mediaImage.ReadImage("../datas/gu.png", eimg);
						//�摜�̕\��
						mediaImage.ShowImage("ENEMY", eimg);
						mediaImage.MoveWindow("ENEMY", 640, 320);
					}

					//���s�̎w��-----------------------------------------------------------------------------
					if (vod==1) 
					{
						//����
						//�摜�̓ǂݍ���
						mediaImage.ReadImage("../datas/kuro.jpg", vodimg);
						//�摜�̕\��ss
						cv::putText(vodimg, "Win", cv::Point(30, 130), cv::FONT_HERSHEY_COMPLEX, 2.5, cv::Scalar(255, 255, 255), 4);
						mediaImage.ShowImage("Victory", vodimg);
						mediaImage.MoveWindow("Victory", 640, 0);
					}
					else if (vod==2) 
					{
						//����
						//�摜�̓ǂݍ���
						mediaImage.ReadImage("../datas/kuro.jpg", vodimg);
						//�摜�̕\��
						cv::putText(vodimg, "Lose", cv::Point(30, 130), cv::FONT_HERSHEY_COMPLEX, 2.5, cv::Scalar(255, 255, 255), 4);
						mediaImage.ShowImage("Victory", vodimg);
						mediaImage.MoveWindow("Victory", 640, 0);
					}
					else if (vod==3) 
					{
						//������
						//�摜�̓ǂݍ���
						mediaImage.ReadImage("../datas/kuro.jpg", vodimg);
						//�摜�̕\��
						cv::putText(vodimg, "Draw", cv::Point(30, 130), cv::FONT_HERSHEY_COMPLEX, 2.5, cv::Scalar(255, 255, 255), 4);
						mediaImage.ShowImage("Victory", vodimg);
						mediaImage.MoveWindow("Victory", 640, 0);
					}

					//�`���L/���āI--------------------------------------------------------------------------------
					
						if (enemy == 1 && vod == 1 && janken == "gu")
						{
							//�摜�̓ǂݍ���
							mediaImage.ReadImage("../datas/kuro.jpg", vodimg);
							//�摜�̕\��
							cv::putText(vodimg, "ture", cv::Point(50, 130), cv::FONT_HERSHEY_COMPLEX, 2.5, cv::Scalar(255, 255, 255), 4);
							mediaImage.ShowImage("JUDGE", vodimg);
							mediaImage.MoveWindow("JUDGE", 960, 0);
						}

						if (enemy == 1 && vod == 1 && janken == "par" )
						{
							//�摜�̓ǂݍ���
							mediaImage.ReadImage("../datas/kuro.jpg", vodimg);
							//�摜�̕\��
							cv::putText(vodimg, "false", cv::Point(50, 130), cv::FONT_HERSHEY_COMPLEX, 2.5, cv::Scalar(255, 255, 255), 4);
							mediaImage.ShowImage("JUDGE", vodimg);
							mediaImage.MoveWindow("JUDGE", 960, 0);
						}

						if ( enemy == 1 && vod == 1 && janken == "choki")
						{
							//�摜�̓ǂݍ���
							mediaImage.ReadImage("../datas/kuro.jpg", vodimg);
							//�摜�̕\��
							cv::putText(vodimg, "false", cv::Point(50, 130), cv::FONT_HERSHEY_COMPLEX, 2.5, cv::Scalar(255, 255, 255), 4);
							mediaImage.ShowImage("JUDGE", vodimg);
							mediaImage.MoveWindow("JUDGE", 960, 0);
						}
					

					//�`���L�ɕ�����I--------------------------------------------------------------------------------
				
						if (enemy == 1 && vod == 2 && janken == "par")
						{
							//�摜�̓ǂݍ���
							mediaImage.ReadImage("../datas/kuro.jpg", vodimg);
							//�摜�̕\��
							cv::putText(vodimg, "ture", cv::Point(50, 130), cv::FONT_HERSHEY_COMPLEX, 2.5, cv::Scalar(255, 255, 255), 4);
							mediaImage.ShowImage("JUDGE", vodimg);
							mediaImage.MoveWindow("JUDGE", 960, 0);
						}

						if (enemy == 1 && vod == 2 && janken == "gu" )
						{
							//�摜�̓ǂݍ���
							mediaImage.ReadImage("../datas/kuro.jpg", vodimg);
							//�摜�̕\��
							cv::putText(vodimg, "false", cv::Point(50, 130), cv::FONT_HERSHEY_COMPLEX, 2.5, cv::Scalar(255, 255, 255), 4);
							mediaImage.ShowImage("JUDGE", vodimg);
							mediaImage.MoveWindow("JUDGE", 960, 0);
						}

						if (enemy == 1 && vod == 2 && janken == "choki")
						{
							//�摜�̓ǂݍ���
							mediaImage.ReadImage("../datas/kuro.jpg", vodimg);
							//�摜�̕\��
							cv::putText(vodimg, "false", cv::Point(50, 130), cv::FONT_HERSHEY_COMPLEX, 2.5, cv::Scalar(255, 255, 255), 4);
							mediaImage.ShowImage("JUDGE", vodimg);
							mediaImage.MoveWindow("JUDGE", 960, 0);
						}
					

					//�`���L�Ƃ������I-----------------------------------------------------------------------------------
				
						if (enemy == 1 && vod == 3 && janken == "choki")
						{
							//�摜�̓ǂݍ���
							mediaImage.ReadImage("../datas/kuro.jpg", vodimg);
							//�摜�̕\��
							cv::putText(vodimg, "ture", cv::Point(50, 130), cv::FONT_HERSHEY_COMPLEX, 2.5, cv::Scalar(255, 255, 255), 4);
							mediaImage.ShowImage("JUDGE", vodimg);
							mediaImage.MoveWindow("JUDGE", 960, 0);
						}

						if (enemy == 1 && vod == 3 && janken == "par" )
						{
							//�摜�̓ǂݍ���
							mediaImage.ReadImage("../datas/kuro.jpg", vodimg);
							//�摜�̕\��
							cv::putText(vodimg, "false", cv::Point(50, 130), cv::FONT_HERSHEY_COMPLEX, 2.5, cv::Scalar(255, 255, 255), 4);
							mediaImage.ShowImage("JUDGE", vodimg);
							mediaImage.MoveWindow("JUDGE", 960, 0);
						}

						if ( enemy == 1 && vod == 3 && janken == "gu")
						{
							//�摜�̓ǂݍ���
							mediaImage.ReadImage("../datas/kuro.jpg", vodimg);
							//�摜�̕\��
							cv::putText(vodimg, "false", cv::Point(50, 130), cv::FONT_HERSHEY_COMPLEX, 2.5, cv::Scalar(255, 255, 255), 4);
							mediaImage.ShowImage("JUDGE", vodimg);
							mediaImage.MoveWindow("JUDGE", 960, 0);
						}
					

					//�p�[�ɏ��āI----------------------------------------------------------------------------------------
					
						if (enemy == 2 && vod == 1 && janken == "choki")
						{
							//�摜�̓ǂݍ���
							mediaImage.ReadImage("../datas/kuro.jpg", vodimg);
							//�摜�̕\��
							cv::putText(vodimg, "ture", cv::Point(50, 130), cv::FONT_HERSHEY_COMPLEX, 2.5, cv::Scalar(255, 255, 255), 4);
							mediaImage.ShowImage("JUDGE", vodimg);
							mediaImage.MoveWindow("JUDGE", 960, 0);
						}

						if (enemy == 2 && vod == 1 && janken == "par" )
						{
							//�摜�̓ǂݍ���
							mediaImage.ReadImage("../datas/kuro.jpg", vodimg);
							//�摜�̕\��
							cv::putText(vodimg, "false", cv::Point(50, 130), cv::FONT_HERSHEY_COMPLEX, 2.5, cv::Scalar(255, 255, 255), 4);
							mediaImage.ShowImage("JUDGE", vodimg);
							mediaImage.MoveWindow("JUDGE", 960, 0);
						}

						if ( enemy == 2 && vod == 1 && janken == "gu")
						{
							//�摜�̓ǂݍ���
							mediaImage.ReadImage("../datas/kuro.jpg", vodimg);
							//�摜�̕\��
							cv::putText(vodimg, "false", cv::Point(50, 130), cv::FONT_HERSHEY_COMPLEX, 2.5, cv::Scalar(255, 255, 255), 4);
							mediaImage.ShowImage("JUDGE", vodimg);
							mediaImage.MoveWindow("JUDGE", 960, 0);
						}
					

					//�p�[�ɕ�����I--------------------------------------------------------------------------------------
					
						if (enemy == 2 && vod == 2 && janken == "gu")
						{
							//�摜�̓ǂݍ���
							mediaImage.ReadImage("../datas/kuro.jpg", vodimg);
							//�摜�̕\��
							cv::putText(vodimg, "ture", cv::Point(50, 130), cv::FONT_HERSHEY_COMPLEX, 2.5, cv::Scalar(255, 255, 255), 4);
							mediaImage.ShowImage("JUDGE", vodimg);
							mediaImage.MoveWindow("JUDGE", 960, 0);
						}

						if (enemy == 2 && vod == 2 && janken == "par" )
						{
							//�摜�̓ǂݍ���
							mediaImage.ReadImage("../datas/kuro.jpg", vodimg);
							//�摜�̕\��
							cv::putText(vodimg, "false", cv::Point(50, 130), cv::FONT_HERSHEY_COMPLEX, 2.5, cv::Scalar(255, 255, 255), 4);
							mediaImage.ShowImage("JUDGE", vodimg);
							mediaImage.MoveWindow("JUDGE", 960, 0);
						}

						if (enemy == 2 && vod == 2 && janken == "choki")
						{
							//�摜�̓ǂݍ���
							mediaImage.ReadImage("../datas/kuro.jpg", vodimg);
							//�摜�̕\��
							cv::putText(vodimg, "false", cv::Point(50, 130), cv::FONT_HERSHEY_COMPLEX, 2.5, cv::Scalar(255, 255, 255), 4);
							mediaImage.ShowImage("JUDGE", vodimg);
							mediaImage.MoveWindow("JUDGE", 960, 0);
						}
					

					//�p�[�Ƃ������I--------------------------------------------------------------------------------------
			
						if (enemy == 2 && vod == 3 && janken == "par")
						{
							//�摜�̓ǂݍ���
							mediaImage.ReadImage("../datas/kuro.jpg", vodimg);
							//�摜�̕\��
							cv::putText(vodimg, "ture", cv::Point(50, 130), cv::FONT_HERSHEY_COMPLEX, 2.5, cv::Scalar(255, 255, 255), 4);
							mediaImage.ShowImage("JUDGE", vodimg);
							mediaImage.MoveWindow("JUDGE", 960, 0);
						}

						if (enemy == 2 && vod == 3 && janken == "gu" )
						{
							//�摜�̓ǂݍ���
							mediaImage.ReadImage("../datas/kuro.jpg", vodimg);
							//�摜�̕\��
							cv::putText(vodimg, "false", cv::Point(50, 130), cv::FONT_HERSHEY_COMPLEX, 2.5, cv::Scalar(255, 255, 255), 4);
							mediaImage.ShowImage("JUDGE", vodimg);
							mediaImage.MoveWindow("JUDGE", 960, 0);
						}

						if ( enemy == 2 && vod == 3 && janken == "choki")
						{
							//�摜�̓ǂݍ���
							mediaImage.ReadImage("../datas/kuro.jpg", vodimg);
							//�摜�̕\��
							cv::putText(vodimg, "false", cv::Point(50, 130), cv::FONT_HERSHEY_COMPLEX, 2.5, cv::Scalar(255, 255, 255), 4);
							mediaImage.ShowImage("JUDGE", vodimg);
							mediaImage.MoveWindow("JUDGE", 960, 0);
						}
					

					//�O�[�ɏ��āI----------------------------------------------------------------------------------------
					
						if (enemy == 3 && vod == 1 && janken == "par")
						{
							//�摜�̓ǂݍ���
							mediaImage.ReadImage("../datas/kuro.jpg", vodimg);
							//�摜�̕\��
							cv::putText(vodimg, "ture", cv::Point(50, 130), cv::FONT_HERSHEY_COMPLEX, 2.5, cv::Scalar(255, 255, 255), 4);
							mediaImage.ShowImage("JUDGE", vodimg);
							mediaImage.MoveWindow("JUDGE", 960, 0);
						}

						if (enemy == 3 && vod == 1 && janken == "gu" )
						{
							//�摜�̓ǂݍ���
							mediaImage.ReadImage("../datas/kuro.jpg", vodimg);
							//�摜�̕\��
							cv::putText(vodimg, "false", cv::Point(50, 130), cv::FONT_HERSHEY_COMPLEX, 2.5, cv::Scalar(255, 255, 255), 4);
							mediaImage.ShowImage("JUDGE", vodimg);
							mediaImage.MoveWindow("JUDGE", 960, 0);
						}

						if (enemy == 3 && vod == 1 && janken == "choki")
						{
							//�摜�̓ǂݍ���
							mediaImage.ReadImage("../datas/kuro.jpg", vodimg);
							//�摜�̕\��
							cv::putText(vodimg, "false", cv::Point(50, 130), cv::FONT_HERSHEY_COMPLEX, 2.5, cv::Scalar(255, 255, 255), 4);
							mediaImage.ShowImage("JUDGE", vodimg);
							mediaImage.MoveWindow("JUDGE", 960, 0);
						}
					

					//�O�[�ɕ�����I--------------------------------------------------------------------------------------
					
						if (enemy == 3 && vod == 2 && janken == "choki")
						{
							//�摜�̓ǂݍ���
							mediaImage.ReadImage("../datas/kuro.jpg", vodimg);
							//�摜�̕\��
							cv::putText(vodimg, "ture", cv::Point(50, 130), cv::FONT_HERSHEY_COMPLEX, 2.5, cv::Scalar(255, 255, 255), 4);
							mediaImage.ShowImage("JUDGE", vodimg);
							mediaImage.MoveWindow("JUDGE", 960, 0);
						}

						if (enemy == 3 && vod == 2 && janken == "par" )
						{
							//�摜�̓ǂݍ���
							mediaImage.ReadImage("../datas/kuro.jpg", vodimg);
							//�摜�̕\��
							cv::putText(vodimg, "false", cv::Point(50, 130), cv::FONT_HERSHEY_COMPLEX, 2.5, cv::Scalar(255, 255, 255), 4);
							mediaImage.ShowImage("JUDGE", vodimg);
							mediaImage.MoveWindow("JUDGE", 960, 0);
						}

						if (enemy == 3 && vod == 2 && janken == "gu")
						{
							//�摜�̓ǂݍ���
							mediaImage.ReadImage("../datas/kuro.jpg", vodimg);
							//�摜�̕\��
							cv::putText(vodimg, "false", cv::Point(50, 130), cv::FONT_HERSHEY_COMPLEX, 2.5, cv::Scalar(255, 255, 255), 4);
							mediaImage.ShowImage("JUDGE", vodimg);
							mediaImage.MoveWindow("JUDGE", 960, 0);
						}
					

					//�O�[�Ƃ������I-------------------------------------------------------------------------------------
					
						if (enemy == 3 && vod == 3 && janken == "gu")
						{
							//�摜�̓ǂݍ���
							mediaImage.ReadImage("../datas/kuro.jpg", vodimg);
							//�摜�̕\��
							cv::putText(vodimg, "ture", cv::Point(50, 130), cv::FONT_HERSHEY_COMPLEX, 2.5, cv::Scalar(255, 255, 255), 4);
							mediaImage.ShowImage("JUDGE", vodimg);
							mediaImage.MoveWindow("JUDGE", 960, 0);
						}

						if (enemy == 3 && vod == 3 && janken == "parqq" )
						{
							//�摜�̓ǂݍ���
							mediaImage.ReadImage("../datas/kuro.jpg", vodimg);
							//�摜�̕\��
							cv::putText(vodimg, "false", cv::Point(50, 130), cv::FONT_HERSHEY_COMPLEX, 2.5, cv::Scalar(255, 255, 255), 4);
							mediaImage.ShowImage("JUDGE", vodimg);
							mediaImage.MoveWindow("JUDGE", 960, 0);
						}

						if (enemy == 3 && vod == 3 && janken == "choki")
						{
							//�摜�̓ǂݍ���
							mediaImage.ReadImage("../datas/kuro.jpg", vodimg);
							//�摜�̕\��
							cv::putText(vodimg, "false", cv::Point(50, 130), cv::FONT_HERSHEY_COMPLEX, 2.5, cv::Scalar(255, 255, 255), 4);
							mediaImage.ShowImage("JUDGE", vodimg);
							mediaImage.MoveWindow("JUDGE", 960, 0);
						}
					
					
				}
					
#endif		   	


#if FourthWeek
				
				resPath = "04res/";
				_mkdir(resPath.c_str());

				//�����ʃ��[�h��I���icolor or hog�j
				const std::string featureMode = "color";

				//�摜�f�[�^�x�[�X��ǂݍ���
				std::map<int, std::string> labels;

				//�f�[�^�x�[�X���쐬�����Ƃ��̃t�H���_�����L��
				std::string my_dir = "result_images";
				mediaImage.ReadFile(labels, "../datas/" + my_dir + "/label.txt");

				//�q�X�g�O�����p�̕ϐ�
				const int bin = 30; //�q�X�g�O�������쐬����Ƃ��̓����x�N�g���̎������i�F�̏ꍇ1�`180, HoG�̏ꍇ��9�j
				const int channel = 1; //�F�����ʂ������ꍇ�݂̂�1�`3�֕ύX�\�i�ʓx�▾�x�̓������g�������Ƃ��j

#if KNN_MODE
				//�@�B�w�K�̕ϐ�����я����l�̏���
				cv::Ptr<cv::ml::KNearest> k_nn;

				//k-NN�̃Z�b�g�A�b�v
				const int k = 1;

				//�w�K���ʂ̕ۑ���
				stringstream ss; ss << k;
				const string saveFileName = resPath + "train_kNN_[" + featureMode + "]_(" + ss.str() + ").xml";
#else
				
				//�@�B�w�K�̕ϐ�����я����l�̏���
				cv::Ptr<cv::ml::SVM> svm;

				const double c = 10.0;
				const double gamma = 1.0;

				//�w�K���ʂ̕ۑ���
				stringstream ss1; ss1 << c;
				stringstream ss2; ss2 << gamma;
				const string saveFileName = resPath + "train_svm_[" + featureMode + "]_(" + ss1.str() + "-" + ss2.str() + ").xml";


#endif

				//�w�K�f�[�^��ǂݍ���
#if KNN_MODE
				k_nn = cv::Algorithm::load<cv::ml::KNearest>(saveFileName);
#else
				
				svm = cv::Algorithm::load<cv::ml::SVM>(saveFileName);
				
#endif
				//�ǂݍ��݃G���[�͔�΂�
				if (img.empty())
				{
					cerr << "�f���t�@�C��������܂���B \n";
					continue;
				}

				//�摜�T�C�Y�𓝈ꂷ��
				cv::Mat resize;
				cv::resize(img, resize, cv::Size(100, 100), cv::INTER_CUBIC);

				//�F�����ʂ��擾����
				cv::Mat hist(1, bin * channel, CV_32FC1);

				if (featureMode == "color")
				{
					if (!mediaImage.GenerateColorHistogram(resize, hist, bin, channel))
					{
						cerr << "�q�X�g�O�������쐬�ł��܂���ł���\n";
						exit(1);
					}
				}
				else if (featureMode == "hog")
				{
					
					mediaImage.SetHoGParameters(9, 20, 1);
					mediaImage.CalcHOGHistgram(resize, hist); //HoG�����ʂ𒊏o

				}

				//���ʂ̕]�����s��
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


			//OpenNI�̏�����
			mediaImage.InitOpenNI(640);

			
			while (1)
			{
				//while�̏��߂ɕK���K�v��1��
				mediaImage.UpDate();

				//RGB�摜���o��
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


					//���i�����擾����
					cv::Mat skeleton_img = image.clone();

					//���i�����擾���C�摜�ɏ�������
					mediaImage.CreateSkeleton(skeleton_img);
					mediaImage.ShowImage("SKELETON", skeleton_img);
					mediaImage.MoveWindow("SKELETON", 0, 300);

					//���i�̍��W�_���擾����
					vector<vector<nite::Point3f> > userPoints = mediaImage.GetUserPoint();

					//���o���ꂽ���[�U�ԍ����擾����
					vector <bool> checkUser = mediaImage.CheckUser();

					//���o���ꂽ���[�U��3���̒l���o�͂���
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

					//���i�����擾����
					cv::Mat skeleton_img = image.clone();

					//���i�����擾���C�摜�ɏ�������
					mediaImage.CreateSkeleton(skeleton_img);
					mediaImage.ShowImage("SKELETON", skeleton_img);
					mediaImage.MoveWindow("SKELETON", 0, 300);

					//���i�̍��W�_���擾����
					vector<vector<nite::Point3f> > userPoints = mediaImage.GetUserPoint();

					//���o���ꂽ���[�U�ԍ����擾����
					vector <bool> checkUser = mediaImage.CheckUser();
#if GESTURE	
#if NonCompulsion				
					//����F���Ɋւ���v���O����
					//���o���ꂽ���[�U�ɑ΂��ď������s��
					for (int i = 0; i < checkUser.size(); ++i)
					{
						if (checkUser[i])
						{
							//����������Ă����OK���o��
							if (mediaImage.GetGestures(i) == MediaGesture::LEFT_HAND_UP_1
								|| mediaImage.GetGestures(i) == MediaGesture::LEFT_HAND_UP_2)
							{
								cerr << "OK\n\n\n";
							}

						}
					}
				}
#else
					//��������v���O�����Ɋւ���v���O����
					//���o���ꂽ���[�U�ɑ΂��ď������s��
					for (int i = 0; i < checkUser.size(); ++i)
					{
						if (checkUser[i])
						{

							
							//���肪������܂őҋ@
							if (mediaImage.PleaseGesture(MediaGesture::LEFT_HAND_UP_2, -1, i))
							{
								cerr << "OK\n\n\n";
							}

						}
					}
#endif
			}
#else

					//����F���Ɋւ���v���O����
					//���o���ꂽ���[�U�ɑ΂��ď������s��
					for (int i = 0; i < checkUser.size(); ++i)
					{
						if (checkUser[i])
						{
							
							//�E�肪�������Ă��邩�ǂ���
							if (mediaImage.PleaseAction(MediaAction::BYE_BYE, -1, i))
							{
								cerr << "OK\n\n\n";
							}
						}
					}
				}
#endif
#endif
				//q�L�[�ŏI��
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
			throw("You must select �yimage�zor�ycamera�zor�yrgbd�z.");
		}
	}
	catch (std::exception& ex)
	{
		std::cout << ex.what() << std::endl;
		mediaImage.Stop();
	}
	return 0;
}