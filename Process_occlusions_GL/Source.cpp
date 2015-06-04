//ITP we divide an RGB image into smaller images of each body part, with occluding body parts marked in 255 blue.
//we will use marching cubes and GLUT to render the body parts.

#include <stdlib.h>
#include <stdio.h>

#include <GL/glut.h>

#include <map>
#include <vector>
#include <sstream>

#include <opencv2\opencv.hpp>

#include <AssimpOpenGL.h>
#include <recons_common.h>
#include <ReconsVoxel.h>

#include <cv_pointmat_common.h>
#include <cv_draw_common.h>

#include <glcv.h>
#include <gh_occlusion.h>

float zNear = 0.1, zFar = 50.0;

#define USE_KINECT_INTRINSICS 1
float ki_alpha, ki_beta, ki_gamma, ki_u0, ki_v0;

//manual zoom
float zoom = 1.f;

//mouse
int mouse_x, mouse_y;
bool mouse_down = false;
bool auto_rotate = true;

#define ZOOM_VALUE 0.1
#define ROTATE_VALUE 1
#define ANIM_DEFAULT_FPS 12

//window dimensions
int win_width, win_height;
//window name
int window1;

float fovy = 45.;

GLint prev_time = 0;
GLint prev_fps_time = 0;
int frames = 0;

//std::vector<TRIANGLE> tris;

std::string video_directory;
std::vector<std::string> filenames;

std::vector<std::vector<float>> triangle_vertices;
std::vector<std::vector<unsigned int>> triangle_indices;
std::vector<std::vector<unsigned char>> triangle_colors;

BodyPartDefinitionVector bpdv;
//std::vector<SkeletonNodeHardMap> snhmaps;
std::vector<Cylinder> cylinders;
std::vector<VoxelMatrix> voxels;
float voxel_size;

//std::vector<FrameData> frame_datas;

cv::Mat opengl_projection;
cv::Mat opengl_modelview;

cv::Vec3b clear_color(0x01, 0x01, 0x01);

float tsdf_offset = 0;

/* ---------------------------------------------------------------------------- */
void reshape(int width, int height)
{
	const double aspectRatio = (float)width / height, fieldOfView = fovy;

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	if (USE_KINECT_INTRINSICS){
		int viewport[4];
		cv::Mat proj_t = build_opengl_projection_for_intrinsics(viewport, -ki_alpha, ki_beta, ki_gamma, ki_u0, ki_v0+10, width, height, zNear, zFar).t(); //note: ki_alpha is negative. NOTE2: the +10 is FUDGE
		glMultMatrixf(proj_t.ptr<float>());
	}
	else{
		gluPerspective(fieldOfView, aspectRatio,
			zNear, zFar);  /* Znear and Zfar */
	}
	glViewport(0, 0, width, height);
	win_width = width;
	win_height = height;

	opengl_projection.create(4, 4, CV_32F);
	glGetFloatv(GL_PROJECTION_MATRIX, (GLfloat*)opengl_projection.data);
	opengl_projection = opengl_projection.t();
}

/* ---------------------------------------------------------------------------- */
void do_motion(void)
{

	int time = glutGet(GLUT_ELAPSED_TIME);
	//angle += (time - prev_time)*0.01;
	prev_time = time;

	frames += 1;
	if ((time - prev_fps_time) > 1000) /* update every seconds */
	{
		int current_fps = frames * 1000 / (time - prev_fps_time);
		printf("%d fps\n", current_fps);
		frames = 0;
		prev_fps_time = time;
	}


	glutPostRedisplay();
}


/* ---------------------------------------------------------------------------- */
void display(void)
{
	glutSetWindow(window1);

	float tmp;

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	

	static int anim_frame = 0;

	double time;
	cv::Mat camera_matrix, camera_pose;
	SkeletonNodeHard root;
	cv::Mat color, fullcolor, depth;
	int facing;
	bool loaded = load_input_frame(filenames[anim_frame], time, camera_pose, camera_matrix, root, color, fullcolor, depth, facing);

	if (loaded){

		{
			//opengl_modelview = frame_datas[anim_frame].mmCameraPose;
			opengl_modelview = camera_pose;
			cv::Mat opengl_modelview_t = opengl_modelview.t();
			glMultMatrixf(opengl_modelview_t.ptr<float>());
		}


		SkeletonNodeHardMap snhmap;
		cv_draw_and_build_skeleton(&root, cv::Mat::eye(4, 4, CV_32F), camera_matrix, camera_pose, &snhmap);

		glEnableClientState(GL_VERTEX_ARRAY);

		for (int i = 0; i < bpdv.size(); ++i){
			glPushMatrix();
			cv::Mat transform_t = (get_bodypart_transform(bpdv[i], snhmap, cv::Mat::eye(4, 4, CV_32F)) * get_voxel_transform(voxels[i].width, voxels[i].height, voxels[i].depth, voxel_size)).t();
			glMultMatrixf(transform_t.ptr<float>());

			glVertexPointer(3, GL_FLOAT, 0, triangle_vertices[i].data());
			glColorPointer(3, GL_UNSIGNED_BYTE, 0, triangle_colors[i].data());

			glColor3fv(bpdv[i].mColor);

			glDrawElements(GL_TRIANGLES, triangle_indices[i].size(), GL_UNSIGNED_INT, triangle_indices[i].data());

			glPopMatrix();
		}

		glDisableClientState(GL_VERTEX_ARRAY);



		//now take the different body part colors and map em to the proper textures

		cv::Mat render_pretexture;
		{
			unsigned char * cp;
			cp = (unsigned char*)malloc(win_width*win_height*sizeof(unsigned char) * 3);
			glReadPixels(0, 0, win_width, win_height, GL_BGR_EXT, GL_UNSIGNED_BYTE, cp);

			cv::Mat render_pretexture_ = cv::Mat(win_height, win_width, CV_8UC3, cp).clone();

			free(cp);

			cv::flip(render_pretexture_, render_pretexture, 0);
		}

		cv::Mat render_depth;
		{
			float * dp;
			dp = (float*)malloc(win_width*win_height*sizeof(float));
			glReadPixels(0, 0, win_width, win_height, GL_DEPTH_COMPONENT, GL_FLOAT, dp);

			cv::Mat render_depth_ = depth_to_z(cv::Mat(win_height, win_width, CV_32F, dp).clone(), opengl_projection);

			free(dp);

			cv::flip(render_depth_, render_depth, 0);
		}

		process_and_save_occlusions(render_pretexture, render_depth, anim_frame, bpdv, clear_color, color, fullcolor, facing, video_directory);

	}
	else{
		std::cout << "failed to load " << filenames[anim_frame] << std::endl;
	}

	glutSwapBuffers();
	do_motion();

	++anim_frame;

	if (anim_frame >= filenames.size()){
		std::cout << "FINISHED" << std::endl;

		//time to cluster frames!

		glutDestroyWindow(window1);
		exit(0);

	}
}


/* ---------------------------------------------------------------------------- */
int main(int argc, char **argv)
{

	if (USE_KINECT_INTRINSICS){
		cv::FileStorage fs;
		fs.open("out_cameramatrix_test.yml", cv::FileStorage::READ);
		fs["alpha"] >> ki_alpha;
		fs["beta"] >> ki_beta;
		fs["gamma"] >> ki_gamma;
		fs["u"] >> ki_u0;
		fs["v"] >> ki_v0;
	}

	std::string voxel_recons_path;
	int numframes = 10;

	for (int i = 1; i < argc; ++i){
		if (strcmp(argv[i], "-d") == 0){
			video_directory = std::string(argv[i + 1]);
			++i;
		}
		else if (strcmp(argv[i], "-v") == 0){
			voxel_recons_path = std::string(argv[i + 1]);
			++i;
		}
		else if (strcmp(argv[i], "-n") == 0){
			numframes = atoi(argv[i + 1]);
			++i;
		}
		else if (strcmp(argv[i], "-t") == 0){
			tsdf_offset = atof(argv[i + 1]);
			++i;
		}
		else{
			std::cout << "Options: -d [video directory] -v [voxel path] -n [num frames] -t [tsdf offset]\n";
			return 0;
		}
	}

	if (video_directory == ""){
		std::cout << "Specify video directory!\n";
		return 0;
	}

	if (voxel_recons_path == ""){
		std::cout << "Specify voxel path!\n";
		return 0;
	}

	std::stringstream filenameSS;
	int startframe = 0;
	cv::FileStorage fs;

	filenameSS << video_directory << "/bodypartdefinitions.xml.gz";

	fs.open(filenameSS.str(), cv::FileStorage::READ);
	for (auto it = fs["bodypartdefinitions"].begin();
		it != fs["bodypartdefinitions"].end();
		++it){
		BodyPartDefinition bpd;
		read(*it, bpd);
		bpdv.push_back(bpd);
	}
	fs.release();

	for (int frame = startframe; frame < startframe + numframes; ++frame){
		filenameSS.str("");
		filenameSS << video_directory << "/" << frame << ".xml.gz";

		filenames.push_back(filenameSS.str());

	}

	//std::vector<PointMap> point_maps;

	//load_frames(filenames, point_maps, frame_datas, false);
	//
	//for (int i = 0; i < frame_datas.size(); ++i){
	//	snhmaps.push_back(SkeletonNodeHardMap());
	//	cv::Mat debug_im = frame_datas[i].mmColor.clone();
	//	cv_draw_and_build_skeleton(&frame_datas[i].mmRoot, cv::Mat::eye(4, 4, CV_32F), frame_datas[i].mmCameraMatrix, frame_datas[i].mmCameraPose, &snhmaps[i]);// , debug_im);
	//	//cv::imshow("debug_im", debug_im);
	//	//cv::waitKey(20);
	//}

	std::vector<cv::Mat> TSDF_array;
	std::vector<cv::Mat> weight_array;

	load_voxels(voxel_recons_path, cylinders, voxels, TSDF_array, weight_array, voxel_size);

	triangle_vertices.resize(bpdv.size());
	triangle_indices.resize(bpdv.size());
	triangle_colors.resize(bpdv.size());

	for (int i = 0; i < bpdv.size(); ++i){
		std::vector<TRIANGLE> tri_add;

		cv::add(tsdf_offset * cv::Mat::ones(TSDF_array[i].rows, TSDF_array[i].cols, CV_32F), TSDF_array[i], TSDF_array[i]);

		if (TSDF_array[i].empty()){
			tri_add = marchingcubes_bodypart(voxels[i], voxel_size);
		}
		else{
			tri_add = marchingcubes_bodypart(voxels[i], TSDF_array[i], voxel_size);
		}
		std::vector<cv::Vec4f> vertices;
		std::vector<unsigned int> vertex_indices;
		for (int j = 0; j < tri_add.size(); ++j){
			for (int k = 0; k < 3; ++k){
				cv::Vec4f candidate_vertex = tri_add[j].p[k];

				bool vertices_contains_vertex = false;
				int vertices_index;
				for (int l = 0; l < vertices.size(); ++l){
					if (vertices[l] == candidate_vertex){
						vertices_contains_vertex = true;
						vertices_index = l;
						break;
					}
				}
				if (!vertices_contains_vertex){
					vertices.push_back(candidate_vertex);
					vertices_index = vertices.size() - 1;
				}
				vertex_indices.push_back(vertices_index);
			}
		}
		triangle_vertices[i].reserve(vertices.size() * 3);
		triangle_colors[i].reserve(vertices.size() * 3);
		triangle_indices[i].reserve(vertex_indices.size());
		for (int j = 0; j < vertices.size(); ++j){
			triangle_vertices[i].push_back(vertices[j](0));
			triangle_vertices[i].push_back(vertices[j](1));
			triangle_vertices[i].push_back(vertices[j](2));
			triangle_colors[i].push_back(bpdv[i].mColor[0] * 255);
			triangle_colors[i].push_back(bpdv[i].mColor[1] * 255);
			triangle_colors[i].push_back(bpdv[i].mColor[2] * 255);
		}
		for (int j = 0; j < vertex_indices.size(); ++j){
			triangle_indices[i].push_back(vertex_indices[j]);
		}
	}

	{
		bool loaded;
		do{
			int anim_frame = 0;
			double time;
			cv::Mat camera_matrix, camera_pose;
			SkeletonNodeHard root;
			cv::Mat color, fullcolor, depth;
			int facing;
			loaded = load_input_frame(filenames[anim_frame], time, camera_pose, camera_matrix, root, color, fullcolor, depth, facing);

			if (loaded){
				win_width = color.cols;
				win_height = color.rows;
			}
		} while (!loaded);
	}

	glutInitWindowSize(win_width, win_height);
	glutInitWindowPosition(100, 100);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
	glutInit(&argc, argv);

	window1 = glutCreateWindow(argv[0]);
	glutDisplayFunc(display);
	glutReshapeFunc(reshape);

	glClearColor(clear_color(0) / 255.f,
		clear_color(1) / 255.f,
		clear_color(2) / 255.f,
		1.f);

	//glEnable(GL_LIGHTING);
	//glEnable(GL_LIGHT0);    /* Uses default lighting parameters */

	glEnable(GL_DEPTH_TEST);

	//glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
	glEnable(GL_NORMALIZE);


	//glColorMaterial(GL_FRONT_AND_BACK, GL_DIFFUSE);

	glutGet(GLUT_ELAPSED_TIME);

	opengl_modelview = cv::Mat::eye(4, 4, CV_32F);

	glutMainLoop();

	return 0;
}
