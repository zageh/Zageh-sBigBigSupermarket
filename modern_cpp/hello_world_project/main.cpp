#include <iostream>
#include <print>
#include <string>

import my_module;

int main() {
    greeting();

    std::print("please enter the path :");
    std::string path;
    std::cin >> path;

    my_module::bulk_rename(path);

    return 0;
}
