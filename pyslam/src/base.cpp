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
        if(Xdist.find(p)!=Xdist.end()) {
                if(Xdist.at(p).find(q)!=Xdist.at(p).end()) {
                        if(Xdist.at(p).at(q).find(r)!=Xdist.at(p).at(q).end()) {
                                return Xdist.at(p).at(q).at(r);
                        }
                }
        }
        return dmax;
}
