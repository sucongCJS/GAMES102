/*
存放所有的数据
*/

#pragma once

#include <UGM/UGM.h>

#include <_deps/imgui/imgui.h>

struct CanvasData {
	Ubpa::valf2 scrolling{ 0.f,0.f };
	std::vector<Ubpa::pointf2> points;
	bool opt_enable_grid{ true };
	bool opt_enable_context_menu{ true };

	bool opt_lagrange{ false };
	bool opt_gauss{ false };
	bool opt_least_squares{ false };
	bool opt_ridge_regression { false };
	bool opt_equidistant_parameterization{ true };
	bool opt_chordal_parameterization{ true };

	std::vector<ImVec2> lagrangeResults;
	std::vector<ImVec2> gaussResults;
	std::vector<ImVec2> leastSquaresResults;
	std::vector<ImVec2> ridgeRegressionResults;
	std::vector<ImVec2> equidistantParameterizationResults;
	std::vector<ImVec2> chordalParameterizationResults;

	float gaussTheta = 100;
	int leastSquaresM = 1; // 多项式数 m-1为最高次数幂
	float ridgeRegressionLambda = 0.1;  // 正则项的权重
	int ridgeRegressionM = 1;  // 多项式的个数
};

#include "details/CanvasData_AutoRefl.inl"
