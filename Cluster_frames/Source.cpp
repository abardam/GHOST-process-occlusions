#include <gh_search.h>
#include <gh_common.h>
#include <opencv2\opencv.hpp>
#include <AssimpCV.h>
#include <recons_common.h>

std::string video_directory;

std::vector<FrameDataProcessed> frame_datas_processed;
BodyPartDefinitionVector bpdv;
std::vector<SkeletonNodeHardMap> snhmaps;

int main(int argc, char **argv)
{
	if (argc <= 1){
		printf("Please enter directory\n");
		return 0;
	}

	video_directory = std::string(argv[1]);

	std::stringstream filenameSS;
	int startframe = 0;
	int numframes = 1000;

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
	std::vector<std::string> filenames;

	for (int frame = startframe; frame < startframe + numframes; ++frame){
		filenameSS.str("");
		filenameSS << video_directory << "/" << frame << ".xml.gz";

		filenames.push_back(filenameSS.str());

	}

	load_processed_frames(filenames, bpdv.size(), frame_datas_processed);

	for (int i = 0; i < frame_datas_processed.size(); ++i){
		snhmaps.push_back(SkeletonNodeHardMap());
		cv_draw_and_build_skeleton(&frame_datas_processed[i].mRoot, frame_datas_processed[i].mCameraPose, frame_datas_processed[i].mCameraMatrix, &snhmaps[i]);
	}

	BodypartFrameCluster bodypart_frame_clusters = cluster_frames(64, bpdv, snhmaps, frame_datas_processed, 2147483647);

	filenameSS.str("");
	filenameSS << video_directory << "/clusters-" << "startframe" << startframe << "numframes" << numframes << ".xml.gz";

	fs.open(filenameSS.str(), cv::FileStorage::WRITE);

	write(fs, "bodypart_frame_clusters", bodypart_frame_clusters);

	fs.release();
}