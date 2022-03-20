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



Timer::Timer(const std::string &k,timerdictptr &tptr ){
        Tptr=tptr;
        key=k;
        t1 = std::chrono::high_resolution_clock::now();
}
Timer::~Timer(){
        t2 = std::chrono::high_resolution_clock::now();
        // std::chrono::duration<double, std::milli> fp_ms = t2 - t1;
        // typedef std::chrono::high_resolution_clock clock;
        int sec=std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();
        int msec=std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

        // std::cout<< "sec = "<<sec << ",  msec =  " << msec << std::endl;
        //
        // std::cout << "Time  = "<< std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count() << "[s]"
        //           << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "[ms]"
        //           << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << "[µs]" << std::endl;

        // static_cast<float>(sec)+
        (*Tptr)[key].push_back( static_cast<float>(msec)/1000 );

        // typedef std::chrono::high_resolution_clock Time;
        // typedef std::chrono::milliseconds ms;
        // typedef std::chrono::duration<float> fsec;
        //
        // fsec fs = t2 - t1;
        // ms d = std::chrono::duration_cast<ms>(fs);
        // std::cout << fs.count() << "s\n";
        // std::cout << d.count() << "ms\n";

}
