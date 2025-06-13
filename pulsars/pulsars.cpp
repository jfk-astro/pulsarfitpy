#include <iostream>

#include "file_loader.h"

int main(const int argc, char* argv[]) {
	if (argc < 0) {
		std::cerr << "Usage: pulsars <file location>\n";
		std::cerr << "Example: pulsars /path/to/pulsars.png\n";

		return -1;
	}

	std::string file_location = argv[1];

	if (file_location.empty()) {
		std::cerr << "Error: File location cannot be empty.\n";
		return -1;
	}

	load_file(file_location);
	std::vector<Point*> points = load_points(file_location);

	return 0;
}