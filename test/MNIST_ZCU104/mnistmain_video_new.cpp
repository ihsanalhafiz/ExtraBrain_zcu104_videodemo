#include <vector>
#include <string>
#include <sys/time.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

#include "Globals.h"
#include "Pop.h"
#include "Prj.h"
#include "Parseparam.h"
#include "PatternFactory.h"
#include "Analys.h"
#include "BCPNN_Kernel.h"

#include "xcl2.hpp"
#include <algorithm>
#include <thread>
#include <mutex>
#include <atomic>

template <typename T>
using AlignedVector = std::vector<T, aligned_allocator<T>>;

using namespace std;
using namespace Globals;

long seed = 0;
int Hx = 5, Mx = 5, Nx, Hin = Hx, Min = Mx, Nin, Hhid = Hin, Mhid = Min, Hut = 1, Mut = 10, Nut, nhlayer = 1, updconnint = -1;
bool inbinarize = false, doshuffle = true, logenabled = true;
float eps = -1, maxfq = 1. / timestep, taup = -1, bgain = 1, wgain = 1, again = 1, cthres = 0.1, nampl = 0,
      nfreq = 1, nswap = 0, nrepl = 0, swaprthr = 1.1, replrthr = 1, replthr = -1;
string lrule = "BCP", actfn = "WTA", patype = "hrand";
string trainpatfile = "", trainlblfile = "", testpatfile = "", testlblfile = "";
int nactHi = 0, nsilHi = 0, nrep = 1, nepoch = 1, trnpat = 10, tenpat = -1, nloop = -1, nhblank = 0, nhflip = 0, myverbosity = 1;
PatternFactory *trainpatfac = nullptr, *trainlblfac = nullptr, *testpatfac = nullptr, *testlblfac = nullptr;
Pop *inpop, *hidpop, *utpop;
Prj *ihprj, *huprj;

std::vector<int> rndPoisson_hid;
std::vector<int> Hihjhi_ih, Chjhi_ih;
std::vector<float> Zj_ih, Zi_ih, Pj_ih, Pi_ih, Pji_ih, Wji_ih, Wji_ih1, Wji_ih2, Bj_ih;
std::vector<float> Zj_hu, Zi_hu, Pj_hu, Pi_hu, Pji_hu, Wji_hu, Bj_hu;
std::vector<char> needsupdbw(2);
std::vector<float> constant_hbm(21);

float nampl_in = 0.001;
int nfreq_in = 1;
float igain0_in = 1.0;
float igain2_in = 1.0;
float bwgain1_in = 1.0;
float bwgain2_in = 1.0;
float taumdt0_in = 1.0;
float taumdt1_in = 1.0;
float taumdt2_in = 1.0;

struct timeval total_time;
struct timeval inference_time;
float infer_time_single = 0.0;
float infer_time = 0.0;

int modeOps = UNSUPERVISED;

auto check_and_copy = [](float* dest, const AlignedVector<float>& src, int size, const std::string& name) {
    for (int i = 0; i < size; i++) {
        dest[i] = src[i];
        if (std::isnan(src[i])) {
            std::cout << "NaN detected in " << name << " at index " << i << std::endl;
        }
    }
};

// Global variables to hold the latest frame and control synchronization.
cv::Mat latestFrame;
std::mutex frameMutex;
std::atomic<bool> keepRunning(true);

void allocVarKernel() {
    rndPoisson_hid.resize(N_hid);
    Hihjhi_ih.resize(H_hid * denHi_ih);
    Chjhi_ih.resize(H_hid * denHi_ih);
    Zj_ih.resize(N_hid);
    Zi_ih.resize(H_hid * denNi_ih);
    Pj_ih.resize(N_hid);
    Pi_ih.resize(H_hid * denNi_ih);
    Pji_ih.resize(N_hid * denNi_ih);
    Wji_ih.resize(N_hid * denNi_ih);
    Wji_ih1.resize(N_hid * denNi_ih);
    Wji_ih2.resize(N_hid * denNi_ih);
    Bj_ih.resize(N_hid);
    Zj_hu.resize(N_ut);
    Zi_hu.resize(H_ut * denNi_hu);
    Pj_hu.resize(N_ut);
    Pi_hu.resize(H_ut * denNi_hu);
    Pji_hu.resize(N_ut * denNi_hu);
    Wji_hu.resize(N_ut * denNi_hu);
    Bj_hu.resize(N_ut);
}

void copyVectorToCl(AlignedVector<float>& Bj_ih_cl, AlignedVector<float>& Wji_ih_cl,
                      AlignedVector<float>& Bj_hu_cl, AlignedVector<float>& Wji_hu_cl,
                      AlignedVector<int>& rndPoisson_hid_cl, AlignedVector<int>& Hihjhi_ih_cl)
{
    auto assign_with_nan_check = [](float& dest, float value, const std::string& name, int idx) {
        dest = value;
        if (std::isnan(value)) {
            std::cout << "NaN detected in " << name << " at index " << idx << std::endl;
        }
    };

    for (int i = 0; i < N_ut; ++i) {
        assign_with_nan_check(Bj_hu_cl[i], huprj->Bj[i], "Bj_hu", i);
    }
    for (int i = 0; i < N_hid; ++i) {
        gsetpoissonmean(nfreq);
        rndPoisson_hid_cl[i] = int(gnextpoisson());
        assign_with_nan_check(Bj_ih_cl[i], Bj_ih[i], "Bj_ih", i);
    }

    for (int i = 0; i < H_hid * denHi_ih; ++i) {
        Hihjhi_ih_cl[i] = Hihjhi_ih[i];
    }

    for (int i = 0; i < N_hid * denNi_ih; ++i) {
        assign_with_nan_check(Wji_ih_cl[i], Wji_ih[i], "Wji_ih", i);
    }

    for (int i = 0; i < N_ut * denNi_hu; ++i) {
        assign_with_nan_check(Wji_hu_cl[i], Wji_hu[i], "Wji_hu", i);
    }
}

void copyDataToVector(Prj *ihprj, Prj *huprj, AlignedVector<float>& Bj_ih_cl, AlignedVector<float>& Wji_ih_cl,
                      AlignedVector<float>& Bj_hu_cl, AlignedVector<float>& Wji_hu_cl,
                      AlignedVector<int>& rndPoisson_hid_cl, AlignedVector<int>& Hihjhi_ih_cl) {

    // Helper to check and assign float values
    auto assign_with_nan_check = [](float& dest, float value, const std::string& name, int idx) {
        dest = value;
        if (std::isnan(value)) {
            std::cout << "NaN detected in " << name << " at index " << idx << std::endl;
        }
    };

    for (int i = 0; i < N_ut; ++i) {
        assign_with_nan_check(Bj_hu_cl[i], huprj->Bj[i], "Bj_hu", i);
    }

    for (int i = 0; i < N_hid; ++i) {
        gsetpoissonmean(nfreq);
        rndPoisson_hid_cl[i] = int(gnextpoisson());
        assign_with_nan_check(Bj_ih_cl[i], ihprj->Bj[i], "Bj_ih", i);
    }

    for (int i = 0; i < H_hid * denHi_ih; ++i) {
        Hihjhi_ih_cl[i] = ihprj->Hihjhi[i];
    }

    for (int i = 0; i < N_hid * denNi_ih; ++i) {
        float wji = ihprj->Wji[i];
        Wji_ih_cl[i] = wji;
    }
    for (int i = 0; i < N_ut * denNi_hu; ++i) {
        Wji_hu_cl[i] = huprj->Wji[i];
    }
}

void copyBufToData(Prj *ihprj, Prj *huprj) {
    // Lambda to check for NaN values in a std::vector
    auto hasNaN = [](const std::vector<float>& arr, const char *name) -> bool {
        for (size_t i = 0; i < arr.size(); i++) {
            if (std::isnan(arr[i])) {
                printf("Error: NaN detected in %s at index %zu\n", name, i);
                return true;
            }
        }
        return false;
    };

    // Check all arrays before copying
    bool nanFound = false;
    nanFound |= hasNaN(Pj_ih, "Pj_ih");
    nanFound |= hasNaN(Pi_ih, "Pi_ih");
    nanFound |= hasNaN(Pji_ih, "Pji_ih");
    nanFound |= hasNaN(Wji_ih, "Wji_ih");
    nanFound |= hasNaN(Bj_ih, "Bj_ih");
    nanFound |= hasNaN(Pj_hu, "Pj_hu");
    nanFound |= hasNaN(Pi_hu, "Pi_hu");
    nanFound |= hasNaN(Pji_hu, "Pji_hu");
    nanFound |= hasNaN(Wji_hu, "Wji_hu");
    nanFound |= hasNaN(Bj_hu, "Bj_hu");

    if (nanFound) {
        printf("Error: NaN values detected. Copy aborted.\n");
        return;
    }

    for (size_t i = 0; i < Pji_ih.size(); i++) {
        ihprj->Wji[i] = Wji_ih[i];
    }
    std::copy(Bj_ih.begin(), Bj_ih.end(), ihprj->Bj);
    std::copy(Wji_hu.begin(), Wji_hu.end(), huprj->Wji);
    std::copy(Bj_hu.begin(), Bj_hu.end(), huprj->Bj);
}


void copyDataToBuffer(Pop *inpop, Pop *hidpop, Pop *utpop, Prj *ihprj, Prj *huprj, int modeOps) {
    constant_hbm[0] = ihprj->bgain;
    constant_hbm[1] = ihprj->wgain;
    constant_hbm[2] = ihprj->ewgain;
    constant_hbm[3] = ihprj->iwgain;
    constant_hbm[4] = huprj->bgain;
    constant_hbm[5] = huprj->wgain;
    constant_hbm[6] = huprj->ewgain;
    constant_hbm[7] = huprj->iwgain;
    constant_hbm[8] = hidpop->nampl;
    constant_hbm[9] = hidpop->nfreq;
    constant_hbm[10] = (modeOps == UNSUPERVISED) ? ihprj->taupdt : 0.0f;
    constant_hbm[11] = (modeOps == SUPERVISED) ? huprj->taupdt : 0.0f;
    constant_hbm[12] = inpop->igain;
    constant_hbm[13] = hidpop->igain;
    constant_hbm[14] = utpop->igain;
    constant_hbm[15] = inpop->bwgain;
    constant_hbm[16] = hidpop->bwgain;
    constant_hbm[17] = utpop->bwgain;
    constant_hbm[18] = inpop->taumdt;
    constant_hbm[19] = hidpop->taumdt;
    constant_hbm[20] = utpop->taumdt;

    needsupdbw[0] = ihprj->needsupdbw;
    needsupdbw[1] = huprj->needsupdbw;

    gsetpoissonmean(hidpop->nfreq);
    for (int i = 0; i < N_hid; i++)
        rndPoisson_hid[i] = gnextpoisson();

    std::copy(ihprj->Hihjhi, ihprj->Hihjhi + H_hid * denHi_ih, Hihjhi_ih.begin());
    std::copy(ihprj->Chjhi, ihprj->Chjhi + H_hid * denHi_ih, Chjhi_ih.begin());
    std::copy(ihprj->Pj, ihprj->Pj + N_hid, Pj_ih.begin());
    std::copy(ihprj->Pi, ihprj->Pi + H_hid * denNi_ih, Pi_ih.begin());
    std::copy(ihprj->Bj, ihprj->Bj + N_hid, Bj_ih.begin());

    for (int i = 0; i < N_hid * denNi_ih; i++) {
        Pji_ih[i] = ihprj->Pji[i];
        Wji_ih[i] = ihprj->Wji[i];
        Wji_ih1[i] = ihprj->Wji[i];
        Wji_ih2[i] = ihprj->Wji[i];
    }

    std::copy(huprj->Pj, huprj->Pj + N_ut, Pj_hu.begin());
    std::copy(huprj->Pi, huprj->Pi + H_ut * denNi_hu, Pi_hu.begin());
    std::copy(huprj->Pji, huprj->Pji + N_ut * denNi_hu, Pji_hu.begin());
    std::copy(huprj->Wji, huprj->Wji + N_ut * denNi_hu, Wji_hu.begin());
    std::copy(huprj->Bj, huprj->Bj + N_ut, Bj_hu.begin());
}

void loadVectorsFromFile(const std::string &filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        fprintf(stderr, "Failed to open file for loading: %s\n", filename.c_str());
        return;
    }

    auto readVec = [&in](auto &vec) {
        size_t size = 0;
        in.read(reinterpret_cast<char*>(&size), sizeof(size));
        vec.resize(size);
        in.read(reinterpret_cast<char*>(vec.data()), size * sizeof(typename std::remove_reference<decltype(vec)>::type::value_type));
    };

    readVec(Hihjhi_ih);
    readVec(Bj_ih);
    readVec(Wji_ih);
    readVec(Bj_hu);
    readVec(Wji_hu);
    readVec(constant_hbm);
    in.close();

    nampl_in = constant_hbm[8];
    nfreq_in = constant_hbm[9];
    igain0_in = constant_hbm[13];
    igain2_in = constant_hbm[14];
    bwgain1_in = constant_hbm[16];
    bwgain2_in = constant_hbm[17];
    taumdt0_in = constant_hbm[18];
    taumdt1_in = constant_hbm[19];
    taumdt2_in = constant_hbm[20];
}

void parseparams(std::string paramfile) {
    Parseparam *parseparam = new Parseparam(paramfile);
    parseparam->postparam("seed", &seed, Long);
    parseparam->postparam("Hx", &Hx, Int);
    parseparam->postparam("Mx", &Mx, Int);
    parseparam->postparam("Hin", &Hin, Int);
    parseparam->postparam("Min", &Min, Int);
    parseparam->postparam("Hhid", &Hhid, Int);
    parseparam->postparam("Mhid", &Mhid, Int);
    parseparam->postparam("Hut", &Hut, Int);
    parseparam->postparam("Mut", &Mut, Int);
    parseparam->postparam("lrule", &lrule, String);
    parseparam->postparam("actfn", &actfn, String);
    parseparam->postparam("maxfq", &maxfq, Float);
    parseparam->postparam("nactHi", &nactHi, Int);
    parseparam->postparam("nsilHi", &nsilHi, Int);
    parseparam->postparam("nfreq", &nfreq, Float);
    parseparam->postparam("nampl", &nampl, Float);
    parseparam->postparam("again", &again, Float);
    parseparam->postparam("eps", &eps, Float);
    parseparam->postparam("taup", &taup, Float);
    parseparam->postparam("bgain", &bgain, Float);
    parseparam->postparam("wgain", &wgain, Float);
    parseparam->postparam("swaprthr", &swaprthr, Float);
    parseparam->postparam("replrthr", &replrthr, Float);
    parseparam->postparam("replthr", &replthr, Float);
    parseparam->postparam("updconnint", &updconnint, Int);
    parseparam->postparam("inbinarize", &inbinarize, Boole);
    parseparam->postparam("trainpatfile", &trainpatfile, String);
    parseparam->postparam("trainlblfile", &trainlblfile, String);
    parseparam->postparam("testpatfile", &testpatfile, String);
    parseparam->postparam("testlblfile", &testlblfile, String);
    parseparam->postparam("nrep", &nrep, Int);
    parseparam->postparam("patype", &patype, String);
    parseparam->postparam("nswap", &nswap, Float);
    parseparam->postparam("nrepl", &nrepl, Float);
    parseparam->postparam("nepoch", &nepoch, Int);
    parseparam->postparam("trnpat", &trnpat, Int);
    parseparam->postparam("tenpat", &tenpat, Int);
    parseparam->postparam("nloop", &nloop, Int);
    parseparam->postparam("nhblank", &nhblank, Int);
    parseparam->postparam("nhflip", &nhflip, Int);
    parseparam->postparam("doshuffle", &doshuffle, Boole);
    parseparam->postparam("cthres", &cthres, Float);
    parseparam->postparam("logenabled", &logenabled, Boole);
    parseparam->postparam("libverbosity", &verbosity, Int);
    parseparam->postparam("verbosity", &myverbosity, Int);
    parseparam->postparam("verbosity", &verbosity, Int);
    parseparam->doparse();

    if (replthr != -1)
        error("mnistmain::parseparam","parameter 'replthr' changed name to 'replrthr'");

    Nx = Hx * Mx;
    Nin = Hin * Min;
    Nut = Hut * Mut;
    if (tenpat < 0)
        tenpat = trnpat;
}

void maketrainpats(string type = "all") {
    if (type == "all" or type == "pats") {
        trainpatfac = new PatternFactory(Nx, Hx, patype);
        trainpatfac->mkpats(trnpat);
    }
    if (type == "all" or type == "lbls") {
        trainlblfac = new PatternFactory(Nut, Hut, patype);
        trainlblfac->mkpats(trnpat);
    }
}

void maketestpats(string type = "all") {
    if (type == "all" or type == "pats") {
        testpatfac = new PatternFactory(trainpatfac);
        testpatfac->hflippats(nhflip);
        testpatfac->hblankpats(nhblank);
    }
    if (type == "all" or type == "lbls") {
        testlblfac = new PatternFactory(trainlblfac);
    }
}

void readtrainpats(string type = "all", bool binarize = false) {
    if (type == "all" or type == "pats") {
        if (Nx == Hx and binarize == false)
            error("main::readtrainpats", "Illegal: Nx == Hx and not binarize");
        trainpatfac = new PatternFactory(Nx, Hx, patype);
        trainpatfac->readpats(trainpatfile, trnpat);
        if (binarize) {
            trainpatfac->binarizepats(trainpatfac->pats);
        }
    }
    if (type == "all" or type == "lbls") {
        trainlblfac = new PatternFactory(Nut, Hut, patype);
        trainlblfac->readpats(trainlblfile, trnpat);
    }
}

void readtestpats(string type = "all", bool binarize = false) {
    if (type == "all" or type == "pats") {
        testpatfac = new PatternFactory(Nx, Hx, patype);
        testpatfac->readpats(testpatfile, tenpat);
        if (binarize)
            testpatfac->binarizepats(testpatfac->pats);
        if (verbosity > 1)
            fprintf(stderr, "nhflip = %d\n", nhflip);
        //testpatfac->hflippats(nhflip);
        //testpatfac->hblankpats(nhblank);
    }
    if (type == "all" or type == "lbls") {
        testlblfac = new PatternFactory(Nut, Hut, patype);
        testlblfac->readpats(testlblfile, tenpat);
    }
}

void makepats(bool binarize = false) {
    if (trainpatfile == "")
        maketrainpats("pats");
    else
        readtrainpats("pats", binarize);
    if (testpatfile == "")
        maketestpats("pats");
    else
        readtestpats("pats", binarize);
    if (trainlblfile == "")
        maketrainpats("lbls");
    else
        readtrainpats("lbls");
    if (testlblfile == "")
        maketestpats("lbls");
    else
        readtestpats("lbls");
}

// Modify the setKernelArguments function
void setKernelArguments(
    cl::Kernel &kernel,
    cl::CommandQueue &qq,
    cl::Buffer &buf_inputdata,
    cl::Buffer &buf_outputdata,
    cl::Buffer &buf_rndPoisson_hid,
    cl::Buffer &buf_Hihjhi_ih,
    cl::Buffer &buf_Bj_ih,
    cl::Buffer &buf_Wji_ih,
    cl::Buffer &buf_Bj_hu,
    cl::Buffer &buf_Wji_hu,
    float nampl, int nfreq, float igain0, float igain2, float bwgain1, 
    float bwgain2, float taumdt0, float taumdt1, float taumdt2)
{
    cl_int err;
    int narg = 0;
    OCL_CHECK(err, err = kernel.setArg(narg++, buf_inputdata));
    OCL_CHECK(err, err = kernel.setArg(narg++, buf_outputdata));
    OCL_CHECK(err, err = kernel.setArg(narg++, buf_rndPoisson_hid));
    OCL_CHECK(err, err = kernel.setArg(narg++, buf_Hihjhi_ih));
    OCL_CHECK(err, err = kernel.setArg(narg++, buf_Bj_ih));
    OCL_CHECK(err, err = kernel.setArg(narg++, buf_Wji_ih));
    OCL_CHECK(err, err = kernel.setArg(narg++, buf_Bj_hu));
    OCL_CHECK(err, err = kernel.setArg(narg++, buf_Wji_hu));
    OCL_CHECK(err, err = kernel.setArg(narg++, nampl));
    OCL_CHECK(err, err = kernel.setArg(narg++, nfreq));
    OCL_CHECK(err, err = kernel.setArg(narg++, igain0));
    OCL_CHECK(err, err = kernel.setArg(narg++, igain2));
    OCL_CHECK(err, err = kernel.setArg(narg++, bwgain1));
    OCL_CHECK(err, err = kernel.setArg(narg++, bwgain2));
    OCL_CHECK(err, err = kernel.setArg(narg++, taumdt0));
    OCL_CHECK(err, err = kernel.setArg(narg++, taumdt1));
    OCL_CHECK(err, err = kernel.setArg(narg++, taumdt2));
    OCL_CHECK(err, err = qq.enqueueMigrateMemObjects({buf_inputdata, buf_rndPoisson_hid, 
        buf_Hihjhi_ih, buf_Bj_ih, buf_Wji_ih, buf_Bj_hu, buf_Wji_hu}, 0));

    OCL_CHECK(err, err = qq.enqueueTask(kernel));

    OCL_CHECK(err, err = qq.enqueueMigrateMemObjects({buf_outputdata}, CL_MIGRATE_MEM_OBJECT_HOST));
    OCL_CHECK(err, err = qq.finish());
}

void setKernelArguments_infer(
    cl::Kernel &kernel,
    cl::CommandQueue &qq,
    cl::Buffer &buf_inputdata,
    cl::Buffer &buf_rndPoisson_hid,
    cl::Buffer &buf_outputdata)
{
    cl_int err;
    int narg = 0;
    OCL_CHECK(err, err = kernel.setArg(narg++, buf_inputdata));
    narg++;
    OCL_CHECK(err, err = kernel.setArg(narg++, buf_rndPoisson_hid));
    OCL_CHECK(err, err = qq.enqueueMigrateMemObjects({buf_inputdata, buf_rndPoisson_hid}, 0));
    OCL_CHECK(err, err = qq.enqueueTask(kernel));
    OCL_CHECK(err, err = qq.enqueueMigrateMemObjects({buf_outputdata}, CL_MIGRATE_MEM_OBJECT_HOST));
    OCL_CHECK(err, err = qq.finish());
}

int main(int argc, char **args) {
    //cv::VideoCapture cap(0, cv::CAP_V4L2);
    //if (!cap.isOpened()) {
    //    std::cerr << "ERROR: Could not open camera." << std::endl;
    //    return 1;
    //}
    //// Request MJPG format for better performance
    //cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    //cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    //cap.set(cv::CAP_PROP_FPS, 60); 

    std::string filename = "/home/ubuntu/ExtraBrain_zcu104_videodemo/video_input/output.avi";

    gettimeofday(&total_time, 0);
    string paramfile = "mnistmain.par";
    std::string binaryFile;
    std::string trained_data_file = "trained_data.bin";
    if (argc > 1)
        paramfile = args[1];
    if (argc > 2)
        binaryFile = args[2];
    if (argc > 3)
        trained_data_file = args[3];
    if (argc > 4)
        filename = args[4];
    parseparams(paramfile);
    
    cv::VideoCapture cap(filename);
    if (!cap.isOpened()) {
        printf("Error: Could not open the video file: %s\n", filename.c_str());
    }
    double fps = cap.get(cv::CAP_PROP_FPS);
    int delay = static_cast<int>(1000 / fps); // delay in ms per frame
    std::cout << "Playing video at " << fps << " FPS. Press ESC to stop." << std::endl;
    cv::Mat frame;

    std::string outputVideoPath = "inference_result.avi";
    int codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G'); // or 'X', 'V', 'I', 'D'
    double outputFps = 30.0; // or cap.get(cv::CAP_PROP_FPS);
    cv::Size frameSize(640, 480); // Make sure it matches your grayscale image size

    cv::VideoWriter outputVideo(outputVideoPath, codec, outputFps, frameSize, false); // false = grayscale
    if (!outputVideo.isOpened()) {
        std::cerr << "Failed to open output video for write: " << outputVideoPath << std::endl;
        return -1;
    }

    int counter = 0;
    auto last_tick = std::chrono::steady_clock::now();



    cl_int err;
    cl::Context context;
    cl::Kernel krnl_bcpnn;
    cl::CommandQueue qq;

    std::vector<int, aligned_allocator<int>> rndPoisson_hid_cl(N_hid);
    std::vector<int, aligned_allocator<int>>   Hihjhi_ih_cl(H_hid * denHi_ih);
    std::vector<float, aligned_allocator<float>> Bj_ih_cl(N_hid);
    std::vector<float, aligned_allocator<float>> Wji_ih_cl(N_hid * denNi_ih);
    std::vector<float, aligned_allocator<float>> Bj_hu_cl(N_ut);
    std::vector<float, aligned_allocator<float>> Wji_hu_cl(N_ut * denNi_hu);
    std::vector<float, aligned_allocator<float>> inputdata(N_in);
    std::vector<float, aligned_allocator<float>> outputdata(N_ut);

    allocVarKernel();

    auto devices = xcl::get_xil_devices();
    auto fileBuf = xcl::read_binary_file(binaryFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    bool valid_device = false;
    for (unsigned int i = 0; i < devices.size(); i++) {
        auto device = devices[i];
        // Creating Context and Command Queue for selected Device
        OCL_CHECK(err, context = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, qq = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
        std::cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        cl::Program program(context, {device}, bins, nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
        } else {
            std::cout << "Device[" << i << "]: program successful!\n";
            OCL_CHECK(err, krnl_bcpnn = cl::Kernel(program, "BCPNN_infer_float", &err));
            valid_device = true;
            break; // we break because we found a valid device
        }
    }
    if (!valid_device) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }

    // Create device buffers using CL_MEM_ALLOC_HOST_PTR
    OCL_CHECK(err, cl::Buffer buf_rndPoisson_hid(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(int)*N_hid, rndPoisson_hid_cl.data(), &err));
    OCL_CHECK(err, cl::Buffer buf_Hihjhi_ih(context,      CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(int)*H_hid*denHi_ih, Hihjhi_ih_cl.data(), &err));
    OCL_CHECK(err, cl::Buffer buf_Wji_ih(context,         CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(float)*N_hid*denNi_ih, Wji_ih_cl.data(), &err));
    OCL_CHECK(err, cl::Buffer buf_Bj_ih(context,          CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(float)*N_hid, Bj_ih_cl.data(), &err));
    OCL_CHECK(err, cl::Buffer buf_Wji_hu(context,         CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(float)*N_ut*denNi_hu, Wji_hu_cl.data(), &err));
    OCL_CHECK(err, cl::Buffer buf_Bj_hu(context,          CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(float)*N_ut, Bj_hu_cl.data(), &err));
    OCL_CHECK(err, cl::Buffer buf_inputdata(context,      CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,  sizeof(float)*N_in, inputdata.data(), &err));
    OCL_CHECK(err, cl::Buffer buf_outputdata(context,     CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(float)*N_ut, outputdata.data(), &err));
    
    float trfcorr = 0, tefcorr = 0;
    vector<float> trfcorrs, tefcorrs;
    Analys *analys1 = new Analys(Hut, Mut, cthres);
    bool pnln = false;
    if (taup < 0) {
        taup = trnpat * timestep / 6.0;
        if (myverbosity > 2) {
            printf("taup = %.3f ", taup);
            fflush(stdout);
        }
        pnln = true;
    }
    if (eps < 0) {
        eps = 1. / (1 + trnpat);
        if (myverbosity > 2) {
            printf("eps = %.1e ", eps);
            fflush(stdout);
        }
        pnln = true;
    }
    if (updconnint < 0) {
        updconnint = 500 * trnpat/60000.0;
        if (myverbosity > 2)
            printf("updconnint = %d", updconnint);
        pnln = true;
    }
    if (myverbosity > 2 and pnln) {
        printf("\n");
        fflush(stdout);
    }
    printf("enter main\n");
    for (int rep = 0, ncorr; rep < nrep; rep++) {
        if (myverbosity > 2 and nrep > 1)
            fprintf(stderr, "rep = %d ", rep);
        clear();
        gsetseed(seed);
        //makepats(inbinarize);
        inpop = new Pop(Hin, Min, "inpop");
        inpop->setactfn(actfn);
        inpop->setmaxfq(maxfq);
        hidpop = new Pop(Hhid, Mhid, "hidpop");
        hidpop->setactfn(actfn);
        hidpop->setnfreq(nfreq);
        hidpop->setnampl(nampl);
        utpop = new Pop(Hut, Mut, "utpop");
        utpop->setactfn(actfn);
        ihprj = new Prj(inpop, hidpop, nactHi, nsilHi, "ihprj");
        ihprj->settaup(taup);
        ihprj->seteps(eps);
        ihprj->setwgain(wgain);
        ihprj->setbgain(bgain);
        ihprj->setnswap(nswap);
        ihprj->setnrepl(nrepl);
        ihprj->setswaprthr(swaprthr);
        ihprj->setreplrthr(replrthr);

        huprj = new Prj(hidpop, utpop, "huprj");
        huprj->settaup(taup);
        huprj->seteps(eps);

        vector<float> prns(Prj::prjs.size());
        fill(prns.begin(), prns.end(), 0);

        copyDataToBuffer(inpop, hidpop, utpop, ihprj, huprj, INFERENCES);
        for(int i = 0; i < N_in; i++) {
            inputdata[i] = 0.0;
        }

        printf("load trained data\n");
        loadVectorsFromFile(trained_data_file);
        printf("copyBufToData\n");
        copyBufToData(ihprj, huprj);
        printf("copyVectorToCl\n");
        copyVectorToCl(Bj_ih_cl, Wji_ih_cl, Bj_hu_cl, Wji_hu_cl, rndPoisson_hid_cl, Hihjhi_ih_cl);
        printf("setKernelArguments\n");

        nampl_in = 0.001;
        nfreq_in = 100;
        igain0_in = 1.0;
        igain2_in = 1.0;
        bwgain1_in = 1.0;
        bwgain2_in = 1.0;
        
        setKernelArguments( krnl_bcpnn, qq, buf_inputdata, buf_outputdata, buf_rndPoisson_hid,
            buf_Hihjhi_ih, buf_Bj_ih, buf_Wji_ih, buf_Bj_hu, buf_Wji_hu,
            nampl_in, nfreq_in, igain0_in, igain2_in, bwgain1_in, bwgain2_in,
            taumdt0_in, taumdt1_in, taumdt2_in);

        int counter = 0;


        while (true) {
            
            bool success = cap.read(frame);
            if (!success || frame.empty()) {
                std::cout << "End of video or failed to read frame." << std::endl;
                break;
            }
            if (!latestFrame.empty()){
                frame = latestFrame.clone();
                printf("getting frame\n");
            }
            if (frame.empty()) {
                // If no frame is available yet, wait a bit and continue.
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }            
            // Convert captured frame to grayscale for processing and display
            cv::Mat gray;
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

            // --- Detect square on white paper ---
            cv::Mat blurred;
            cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);
            cv::Mat edged;
            cv::Canny(blurred, edged, 50, 150);

            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(edged, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            std::vector<cv::Point> square;
            for (const auto &cnt : contours) {
                double area = cv::contourArea(cnt);
                if (area < 1000) continue; // filter out small contours

                std::vector<cv::Point> approx;
                cv::approxPolyDP(cnt, approx, 0.02 * cv::arcLength(cnt, true), true);
                // Check for quadrilateral and convexity
                if (approx.size() == 4 && cv::isContourConvex(approx)) {
                    cv::Rect r = cv::boundingRect(approx);
                    float aspectRatio = static_cast<float>(r.width) / r.height;
                    if (aspectRatio >= 0.8 && aspectRatio <= 1.2) {
                        square = approx;
                        break; // take the first acceptable square found
                    }
                }
            }

            cv::Mat warp; // 28x28 warped grayscale image
            // Vectors to hold the flattened warp image and its binarized version
            std::vector<float> warpVec;
            std::vector<float> wrap_binarize;
            std::vector<float> outputClass(N_ut);
            warpVec.resize(28 * 28);
            wrap_binarize.resize(2 * 28 * 28);

            if (!square.empty()) {
                // Draw the detected square on the grayscale image (for visualization)
                for (int i = 0; i < 4; i++) {
                    cv::line(gray, square[i], square[(i+1)%4], cv::Scalar(0), 2);
                }
            
                // --- Order square corners ---
                std::vector<cv::Point2f> pts;
                for (auto pt : square) pts.push_back(cv::Point2f(pt.x, pt.y));
            
                // Find corners: top-left, top-right, bottom-right, bottom-left
                cv::Point2f tl, tr, br, bl;
                float sumMin = 1e9, sumMax = -1e9;
                float diffMin = 1e9, diffMax = -1e9;
                for (auto p : pts) {
                    float sum = p.x + p.y;
                    float diff = p.y - p.x;
                    if (sum < sumMin) { sumMin = sum; tl = p; }
                    if (sum > sumMax) { sumMax = sum; br = p; }
                    if (diff < diffMin) { diffMin = diff; tr = p; }
                    if (diff > diffMax) { diffMax = diff; bl = p; }
                }
            
                std::vector<cv::Point2f> ordered = {tl, tr, br, bl};
            
                // --- Apply inward offset ---
                float offset_ratio = 0.08f;  // 8% inward
                cv::Point2f center(0, 0);
                for (auto &p : ordered) center += p;
                center *= (1.0f / 4.0f);
            
                for (auto &p : ordered) {
                    p = center + (p - center) * (1.0f - offset_ratio);
                }
            
                // --- Warp to 28x28 ---
                std::vector<cv::Point2f> dst = {
                    {0, 0}, {27, 0}, {27, 27}, {0, 27}
                };
                cv::Mat M = cv::getPerspectiveTransform(ordered, dst);
                cv::warpPerspective(gray, warp, M, cv::Size(28, 28));
            
                // --- Increase contrast and invert ---
                cv::normalize(warp, warp, 0, 255, cv::NORM_MINMAX);  // enhance contrast
                cv::threshold(warp, warp, 128, 255, cv::THRESH_BINARY);  // binarize
                cv::bitwise_not(warp, warp);  // invert
            
                // --- Convert to float vector ---
                for (int i = 0; i < warp.rows; i++) {
                    for (int j = 0; j < warp.cols; j++) {
                        warpVec[i * warp.cols + j] = warp.at<uchar>(i, j) / 255.0f;
                    }
                }
            
                for (size_t i = 0; i < warpVec.size(); i++) {
                    inputdata[2 * i] = 1 - warpVec[i];
                    inputdata[2 * i + 1] = warpVec[i];
                }
            
                gsetpoissonmean(nfreq_in);
                for (int i = 0; i < N_hid; ++i) {
                   rndPoisson_hid_cl[i] = (gnextpoisson());
                }
                
                setKernelArguments_infer(krnl_bcpnn, qq, buf_inputdata, buf_rndPoisson_hid, buf_outputdata);
                for (int i = 0; i < N_ut; i++) {
                    outputClass[i] = outputdata[i];
                }
            }
            
            auto max_iter = std::max_element(outputClass.begin(), outputClass.end());
            int max_index = std::distance(outputClass.begin(), max_iter);

            // Overlay counter text on the grayscale image (using white text)
            std::string text = "Class: " + std::to_string(max_index);
            //counter++;
            printf("Detected Class: %d\n", max_index);
            int fontFace = cv::FONT_HERSHEY_SIMPLEX;
            double fontScale = 1.0;
            int thickness = 2;
            cv::Point textOrg(20, 50);
            cv::putText(gray, text, textOrg, fontFace, fontScale, cv::Scalar(255), thickness);

            // Display the grayscale camera feed
            cv::imshow("Grayscale Camera", gray);

            // Display the 28x28 warped grayscale image if available
            if (!warp.empty()) {
                cv::imshow("Warped (28x28 Grayscale)", warp);
            }

            // Exit if ESC is pressed
            if (cv::waitKey(delay) == 27) break;
        }
        
    }
    // -------------------------------------------------
    if (inpop) {
        delete inpop;
        inpop = nullptr;
    }
    if (hidpop) {
        delete hidpop;
        hidpop = nullptr;
    }
    if (utpop) {
        delete utpop;
        utpop = nullptr;
    }
    if (ihprj) {
        delete ihprj;
        ihprj = nullptr;
    }
    if (huprj) {
        delete huprj;
        huprj = nullptr;
    }

    // Delete dynamically allocated arrays
    rndPoisson_hid.clear();
    Hihjhi_ih.clear();
    Chjhi_ih.clear();
    Zj_ih.clear();
    Zi_ih.clear();
    Pj_ih.clear();
    Pi_ih.clear();
    Pji_ih.clear();
    Wji_ih.clear();
    Wji_ih1.clear();
    Wji_ih2.clear();
    Bj_ih.clear();
    Zj_hu.clear();
    Zi_hu.clear();
    Pj_hu.clear();
    Pi_hu.clear();
    Pji_hu.clear();
    Wji_hu.clear();
    Bj_hu.clear();
    needsupdbw.clear();
    constant_hbm.clear();

    // Optionally, force OpenCL objects to be cleaned up explicitly.
    // For example, reset the command queue, context, and kernel objects:
    krnl_bcpnn = cl::Kernel();   // Reset kernel
    qq = cl::CommandQueue();       // Reset command queue
    context = cl::Context();       // Reset context

    keepRunning = false;
    cap.release();
    cv::destroyAllWindows();

    std::exit(0);
}