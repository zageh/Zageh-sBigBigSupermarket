#ifndef WAREHOUSE_H
#define WAREHOUSE_H

#include <iostream>
#include <string>
#include <vector>
using namespace std;

class chips{
    public:
    string name;
    double quality;
    string assessment;
    void showInfo(){
        cout<<"名称："<<name<<endl;
        cout<<"口感评分："<<quality<<"/5"<<endl;
        cout<<"综合评价："<<assessment<<endl;
    }
};

extern vector<chips> warehouse;

void initialWareHouse();

#endif