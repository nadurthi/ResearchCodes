#include "pf.h"


PF::PF(std::string opt) : model(opt){
        options=json::parse(opt);
}



void PF::propforward(){
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> d{0,1};

  #pragma omp parallel for num_threads(6)
        for(int i=0; i<X.rows(); ++i) {
                Eigen::VectorXf r(X.cols());
                for(int j=0; j<X.cols(); ++j) {
                        r(j)=d(gen);
                }
                X.row(i)=model.propforward(X.row(i).transpose())+r;
        }
}

void PF::measUpdt(Eigen::VectorXf likelihood ){
        for(int i=0; i<X.rows(); ++i) {
                W(i)=W(i)*likelihood(i);
        }
        renormalize();
}

void PF::renormalize(){
        W=W/W.sum();
}

void PF::bootstrapresample(){
        std::random_device rd;
        std::mt19937 gen(rd());
        std::discrete_distribution<> d(W.begin(),W.end());
        for(int i=0; i<X.rows(); ++i) {
                W(i)=d(gen);
        }
}
// float PF::Neff(){
//         return 1/W.pow(2).sum();
// }
