/*****************************************************************

  Author: Anders Lansner

  Created: 2023-09-08     Modified: 2024-02-02

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

#ifndef __Netw_included
#define __Netw_included

#include <string>
#include <vector>
#include <cstring>
#include "Pop.h"
#include "Prj.h"
#include "Globals.h"

class Pop;
class Prj;

class Netw {
//********** Static variables and methods **********
    protected:
        static std::vector<Netw *> netws;

    public:
        static bool netwnamexists(std::string name);
        static void clear();

        friend class Pop;
        friend class Prj;

// ************** Class variables and methods **************
        std::string name;
        int id;
        std::vector<Pop *> pops;
        std::vector<Pop *> inpops;
        std::vector<Pop *> utpops;
        std::vector<Prj *> prjs;

        Netw(std::string name = "");
        ~Netw();
        Pop *addPop(Pop *pop, bool isinpop = false, bool isutpop = false);
        Prj *addPrj(Prj *prj);
        void setinpopinput(std::vector<float> Xi = {});
        void setinpopinputs(std::vector<std::vector<float> > Xis = {});
        void setutpopinput(std::vector<float> Xj = {});
        void setutpopinputs(std::vector<std::vector<float> > Xjs = {});
        void popsreset(bool resetaxdelbuf = true);
        void popsresetbwsup();
        void popsupdsup();
        void popsupdact();
        void prjsupdact();
        void prjsreset();
        void prjsupdbwsup();
        void prjscontribute();
        void prjsupdtraces(std::vector<float> prns = {});
        void prjsupdbws(bool force = false);
        void prjsupdconns();
        void reset();
        void updstate(std::vector<float> prns, bool doupdbw = true);
        void updstate(float prn = 0, bool doupdbw = true);
        virtual void run();
};

#endif // __Netw_included
