#include "base.h"
#include "binmatch.h"
#include "measmapmanagers.h"
#include "pf.h"
#include "targetmodels.h"


int main(){
        MeasManager measmang("{'a':1}");
        MapManager mapmang("{'a':1}");
        BinMatch bm(Matrix2frow({10,10}),Matrix2frow({1,1}),Matrix2frow({1,1}),"{'a':1}");



}
