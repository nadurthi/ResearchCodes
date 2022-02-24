#include "base.h"


json
parseOptions(std::string opt){
        auto options=json::parse(opt);
        return options;
}


json
readOptionsFile(std::string file){
        std::ifstream i(file);
        json j;
        i >> j;
        return j;
}



float getitemXdist(const xdisttype& Xdist,uint16_t p,uint16_t q,uint16_t r,float dmax){
        auto pitr=Xdist.find(p);
        if(pitr!=Xdist.end()) {
                auto qitr=pitr->second.find(q);
                if(qitr!=pitr->second.end()) {
                        auto ritr=qitr->second.find(r);
                        if(ritr!=qitr->second.end()) {
                                return ritr->second;
                        }
                }
        }
        return dmax;
}
