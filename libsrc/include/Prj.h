/*****************************************************************

  Author: Anders Lansner, Naresh Ravichandran

  Created: 2024-07-10     Modified: 2024-07-10

*****************************************************************/
/*****************************************************************

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

******************************************************************/

#ifndef __Prj_included
#define __Prj_included

#include <vector>
#include <cstring>
#include <math.h>
#include <limits>

//#include <cublas_v2.h>

#include "Globals.h"
#include "Pop.h"

#define ABSENT 0
#define SILENT 1
#define ACTIVE 2

class Pop;
class Prj;
class Axo;

class Prj {

//********** Static variables and methods **********

    public:
        static std::vector<Prj *> prjs;

    public:
        static bool prjnamexists(std::string name);
        static int nprj();
        static void clear();
        static ulong nallocbyteall();
        static void resetall();
        static void upddenactall();
        static void updtracesall(std::vector<float> prns = std::vector<float>(0));
        static void updbwall(bool force = false);
        static void updbwsupall();
        static void contributeall();
        static void updconnsall();

//************** Class variables and methods **************

    public:
        std::string name;
        int id;
        bool recurrent, needsupdbw;
        Pop *srcpop, *trgpop;
        Axo *axo;
        int lrule;
        int Mi, axoHi, axoNi, denHi, denNi, nactHi, nsilHi, Hj, Mj, Nj, selfc;
        float eps, taupdt, tauedt, tauzidt, tauzjdt, wgain, bgain, ewgain, iwgain, fgain,
              nswap, nrepl, swaprthr, replrthr;
        int *Chjhi, *Hihjhi;
        uint *rnduints;
        float P0; // nbrav
        float *Zi, *Zj, *Zji, *Ei, *Ej, *Eji, *Pi, *Pj, *Bj, *Pji, *Wji, *MIhjhi, *nMIhjhi;
        float *srcpopact, *trgpopact, *trgpopbwsup, *axoact, *denact;
        float *bwsupinf, *bwsup;
        int *Hifanout;
        /********** Variables for monitoring ***********/
        float gminactsc, gmaxactsc, gminsilsc, gmaxsilsc;
        int *nswapped, *nrepled;
        int gnswapped, gnrepled;
        /********** Variables for printout and logging ***********/
        int *xfieldi;
        float *xfieldf;
    public:
        Prj(Pop *srcpop, Pop *trgpop, std::string name = "");
        Prj(Pop *srcpop, Pop *trgpop, int nactHi, int nsilHi, std::string name = "");
        ~Prj();
    protected:
        void initialize();
        void allocmem();
        void reinitialize();
        void reset();
        bool ISABSENT(int hj, int hi);
        void initconns(int nactHi, int nsilHi);
        void updhifanout();
        ulong nallocbyte();
    public:
        void setlrule(std::string lrule);
        void seteps(float eps);
        void settauzi(float tauzi);
        void settauzj(float tauzj);
        void settaue(float taue);
        void settaup(float taup);
        void setselfc(std::string selfc = "HON");
        void setrandPji();
        void pscrambleps(float kN = 1);
        void setBj(float *Bj);
        void setWji(float *Wji);
        void setbgain(float bgain);
        void setwgain(float wgain);
        void setwgainx(float ewgain, float iwgain);
        void setbwgain(float bwgain);
        void setnswap(float nswap);
        void setnrepl(float nrepl);
        void setswaprthr(float swaprthr);
        void setreplrthr(float replrthr);
        void setdelays(float delay, float spread);
        void setdelays(std::vector<std::vector<float> > delaymat);
        void upddenact();
        void putsrcact(float *srcact);
        void updtraces(float *denact, float *trgact, float prn = 1);
        void updtraces(float prn = 0);
        void updbw(bool force = false);
        void updbwsup();
        void contribute();
        void updMIsc();
        void nrmMIsc();
        void resetcmonitor();
        void miscsum(std::string filename = "");
        void swapconns();
        void replconns();
        int getnelem(int field);
        int *getfieldi(int field);
        float *getfieldf(int field);
        int *expandfieldi(int field);
        float *expandfieldf(int field);
        void prnfield(int field, std::string filename = "");
        void prnfield(std::string field, std::string filename = "");
};

#endif // __Prj_included
