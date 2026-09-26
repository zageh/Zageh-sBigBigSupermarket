module;

#include <filesystem>
#include <iostream>
#include <print>
#include <string>

export module my_module;

export namespace my_module {

    void bulk_rename(const std::string& path) {
        namespace fs = std::filesystem;

        for (const auto& file : fs::directory_iterator(path)) {
            if (!file.is_regular_file()) {
                continue;
            }

            const fs::path& file_path = file.path();
            const fs::path parent_path = file_path.parent_path();
            const fs::path file_name = file_path.stem();
            const fs::path extension = file_path.extension();

            const std::string new_filename = file_name.string()
                + "_dd_ee"
                + extension.string();

            const fs::path new_file_path = parent_path / new_filename;

            fs::rename(file_path, new_file_path);
        }
    }
}

export void greeting() {
    std::print("please enter your name :");
    std::string name;
    std::cin >> name;
    std::println("Hello! {}", name);
}
