#include "base.h"
#include "binmatch.h"
#include "measmapmanagers.h"
#include "pf.h"
#include "targetmodels.h"


int main(){
        std::string optstr =R"(
          {
            "happy": true,
            "Lmax": [30,30]
          }
        )";
        MeasManager measmang(optstr);
        MapManager mapmang(optstr);
        BinMatch bm(optstr);



}
