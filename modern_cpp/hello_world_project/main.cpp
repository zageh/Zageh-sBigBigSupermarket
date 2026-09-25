#include<print>
#include<iostream>
#include<string>

int main(){
    std::print("please enter your name :");
    std::string name;
    std::cin >> name;
    std::println("Hello! {}", name);

    return 0;
}