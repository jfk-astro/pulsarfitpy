#ifndef FILE_LOADER_H
#define FILE_LOADER_H

#include <string>
#include <vector>
#include <iostream>

#include "matrix.h"
#include "point.h"

void load_file(std::string file_location);
std::vector<Point*> load_points(const std::string& file_location);

#endif // FILE_LOADER_H