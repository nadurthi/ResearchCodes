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
