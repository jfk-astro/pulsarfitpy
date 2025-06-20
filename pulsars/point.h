#ifndef POINT_H
#define POINT_H

struct Point {
	Point(double x = 0.0, double y = 0.0) : x_(x), y_(y) {}
	Point(const Point& other) : x_(other.getX()), y_(other.getY()) {}

	double getX() const {
		return x_;
	}

	double getY() const {
		return y_;
	}

	void setX(double newX) {
		x_ = newX;
	}

	void setY(double newY) {
		y_ = newY;
	}

private:
	double x_;
	double y_;

};

#endif // POINT_H