#include "supermarketSystem.h"
#include <vector>
using namespace std;

extern vector<chips> WareHouse;

void showInfo(const string& name);

int main() {
    cout << "欢迎来到zageh大超市薯片区！！！" << endl;

    initialWareHouse();
    while (true) {
        cout << "请输入你想要查找的薯片名称" << endl;
        bool found = false;
        string searchName;
        getline(cin, searchName);

        for (const auto& chips : WareHouse) {
            if (searchName == chips.name) {
                showInfo(chips.name);
                found = true;
                break;
            }
        }
        if (found) {
            break;
        } else {
            cout << "找不到对应薯片名，请重新输入" << endl;
        }
    }

    return 0;
}