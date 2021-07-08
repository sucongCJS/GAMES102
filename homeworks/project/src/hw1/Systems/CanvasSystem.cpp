#include "CanvasSystem.h"

#include "../Components/CanvasData.h"
#include "../Eigen/Dense"

using namespace Ubpa;

float lagrange(std::vector<Ubpa::pointf2>points, float x)
{
	int size = points.size();

	float result = 0;
	for (int i = 0; i < size; i++)
	{
		float base = 1;
		for (int j = 0; j < size; j++)
		{
			if (j == i) continue;
			base *= (x - points[j][0]) / (points[i][0] - points[j][0]);
		}
		result += points[i][1] * base;
	}
	return result;
}

float gauss(std::vector<Ubpa::pointf2>points, float x, float theta)
{
	int size = points.size();

	// Gb =y  ->  b = G^{-1}b
	Eigen::MatrixXf G(size, size);
	Eigen::VectorXf y(size);
	Eigen::VectorXf b(size);
	for (int row = 0; row < size; row++)
		for (int col = 0; col < size; col++)
			G(row, col) = exp(-((points[row][0] - points[col][0]) * (points[row][0] - points[col][0])) / (2 * theta * theta));
	for (int i = 0; i < size; i++)
		y(i) = points[i][1];

	b = G.inverse() * y;

	float ret = 0;
	for (int i = 0; i < size; i++)
		ret += b[i] * exp(-((x - points[i][0]) * (x - points[i][0])) / (2 * theta * theta));

	return ret;
}

float leastSquares(std::vector<Ubpa::pointf2>points, float x, int m)
{
	int size = points.size();

	// B^{T}Y = B^{T}B\alpha  ->  \alpha = (B^{T}B)^{-1}B^{T}Y
	Eigen::MatrixXf B(size, m);
	Eigen::VectorXf Y(size);
	Eigen::VectorXf alpha(m);
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < m; j++)
			B(i, j) = pow(points[i][0], j);
		Y(i) = points[i][1];
	}
	alpha = (B.transpose() * B).inverse() * B.transpose() * Y;

	float ret = 0;
	for (int i = 0; i < m; i++)
		ret += alpha[i] * pow(x, i);

	return ret;
}

float ridgeRegression(std::vector<Ubpa::pointf2>points, float x, int m, float lambda)
{
	int size = points.size();

	Eigen::MatrixXf B(size, m);
	Eigen::VectorXf Y(size);
	Eigen::VectorXf alpha(m);
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < m; j++)
			B(i, j) = pow(points[i][0], j);
		Y(i) = points[i][1];
	}
	Eigen::MatrixXf I(m, m);
	I.setIdentity();
	alpha = (B.transpose() * B + lambda * I).inverse() * B.transpose() * Y;

	float ret = 0;
	for (int i = 0; i < m; i++)
		ret += alpha[i] * pow(x, i);

	return ret;
}

void equidistantParameterization(std::vector<Ubpa::pointf2>points, std::vector<ImVec2> &results)
{
	int size = points.size();
	float distance = 1.0f / (size-1);
	std::vector<float> t(size, 0.0);  // t

	for (size_t i = 1; i < size; ++i)
	{
		t[i] = t[i - 1] + distance; 
	}

	std::vector<Ubpa::pointf2> xt;
	std::vector<Ubpa::pointf2> yt;
	for (size_t i = 0; i < size; ++i)
	{
		xt.push_back(Ubpa::pointf2(t[i], points[i][0]));
		yt.push_back(Ubpa::pointf2(t[i], points[i][1]));
	}

	for (float i = 0; i <= 1; i += 0.01)
	{
		//results.push_back(ImVec2(gauss(xt, i, 150.0), gauss(yt, i, 150.0)));
		results.push_back(ImVec2(lagrange(xt, i), lagrange(yt, i)));
	}
}

void chordalParameterization(std::vector<Ubpa::pointf2>points, std::vector<ImVec2>& results)
{
	int size = points.size();
	float distance = 1.0f / (size - 1);
	std::vector<float> t(size, 0.0);  // t

	for (size_t i = 1; i < size; ++i)
	{
		t[i] = t[i - 1] + distance;
	}

	std::vector<Ubpa::pointf2> xt;
	std::vector<Ubpa::pointf2> yt;
	for (size_t i = 0; i < size; ++i)
	{
		xt.push_back(Ubpa::pointf2(t[i], points[i][0]));
		yt.push_back(Ubpa::pointf2(t[i], points[i][1]));
	}

	for (float i = 0; i <= 1; i += 0.01)
	{
		results.push_back(ImVec2(lagrange(xt, i), lagrange(yt, i)));
	}
}

/*
offset: 偏移量, 一般是画布的原点位置
*/
void addLine(ImDrawList* draw_list, ImVec2 offset, std::vector<ImVec2> results, ImU32 color, float thickness)
{
	if (results.size() < 2) return;

	for (int i = 0; i < results.size() - 1; i++)
		draw_list->AddLine(ImVec2(results[i][0] + offset.x, results[i][1] + offset.y),
			ImVec2(results[i + 1][0] + offset.x, results[i + 1][1] + offset.y), color, thickness);
}

void CanvasSystem::OnUpdate(Ubpa::UECS::Schedule& schedule) {
	schedule.RegisterCommand([](Ubpa::UECS::World* w) {
		auto data = w->entityMngr.GetSingleton<CanvasData>();
		if (!data)
			return;
		
		if (ImGui::Begin("Canvas")) {
			//ImGui::Checkbox("Enable grid", &data->opt_enable_grid);
			//ImGui::Checkbox("Enable context menu", &data->opt_enable_context_menu);
			//ImGui::Text("Mouse Left: drag to add lines,\nMouse Right: drag to scroll, click for context menu.");

			// Typically you would use a BeginChild()/EndChild() pair to benefit from a clipping region + own scrolling.
			// Here we demonstrate that this can be replaced by simple offsetting + custom drawing + PushClipRect/PopClipRect() calls.
			// To use a child window instead we could use, e.g:
			//      ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));      // Disable padding
			//      ImGui::PushStyleColor(ImGuiCol_ChildBg, IM_COL32(50, 50, 50, 255));  // Set a background color
			//      ImGui::BeginChild("canvas", ImVec2(0.0f, 0.0f), true, ImGuiWindowFlags_NoMove);
			//      ImGui::PopStyleColor();
			//      ImGui::PopStyleVar();
			//      [...]
			//      ImGui::EndChild();
			static bool needReCalculate = true;  // 是否需要重新计算函数

			ImGui::Checkbox("Lagrange", &data->opt_lagrange);

			ImGui::Checkbox("Gauss", &data->opt_gauss); ImGui::SameLine(200);
			if (ImGui::SliderFloat("Theta", &data->gaussTheta, 0.0f, 300.0f)) { needReCalculate = true; }

			ImGui::Checkbox("Least Squares", &data->opt_least_squares); ImGui::SameLine(200);
			if (ImGui::InputInt("least squares m", &data->leastSquaresM, 1)) { needReCalculate = true; }
			if (data->leastSquaresM < 1) { data->leastSquaresM = 1; } // set the range
			if (data->leastSquaresM > data->points.size()) { data->leastSquaresM = data->points.size(); }

			ImGui::Checkbox("Ridge Regression", &data->opt_ridge_regression); ImGui::SameLine(200);
			if (ImGui::InputFloat("lamda", &data->ridgeRegressionLambda, 0.01, 1, 3)) { needReCalculate = true; }  ImGui::Indent(192);
			if (ImGui::InputInt("ridge regression m", &data->ridgeRegressionM, 1)) { needReCalculate = true; }
			if (data->ridgeRegressionM < 1) { data->ridgeRegressionM = 1; }  // set the range
			if (data->ridgeRegressionM > data->points.size()) { data->ridgeRegressionM = data->points.size(); }
			if (data->ridgeRegressionLambda < 0) data->ridgeRegressionLambda = 0;
			ImGui::Unindent(192);

			ImGui::Checkbox("Equidistant Parameterization", &data->opt_equidistant_parameterization);
			ImGui::Checkbox("Chordal Parameterization", &data->opt_chordal_parameterization);

			// Using InvisibleButton() as a convenience 1) it will advance the layout cursor and 2) allows us to use IsItemHovered()/IsItemActive()
			ImVec2 canvas_p0 = ImGui::GetCursorScreenPos();      // ImDrawList API uses screen coordinates!
			ImVec2 canvas_sz = ImGui::GetContentRegionAvail();   // Resize canvas to what's available
			if (canvas_sz.x < 50.0f) canvas_sz.x = 50.0f;
			if (canvas_sz.y < 50.0f) canvas_sz.y = 50.0f;
			ImVec2 canvas_p1 = ImVec2(canvas_p0.x + canvas_sz.x, canvas_p0.y + canvas_sz.y);

			// Draw border and background color
			ImGuiIO& io = ImGui::GetIO();
			ImDrawList* draw_list = ImGui::GetWindowDrawList();
			draw_list->AddRectFilled(canvas_p0, canvas_p1, IM_COL32(50, 50, 50, 255));
			draw_list->AddRect(canvas_p0, canvas_p1, IM_COL32(55, 55, 55, 255));  // border

			// This will catch our interactions
			ImGui::InvisibleButton("canvas", canvas_sz, ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
			const bool is_hovered = ImGui::IsItemHovered(); // Hovered
			const bool is_active = ImGui::IsItemActive();   // Held
			const ImVec2 origin(canvas_p0.x + data->scrolling[0], canvas_p0.y + data->scrolling[1]); // Lock scrolled origin
			const pointf2 mouse_pos_in_canvas(io.MousePos.x - origin.x, io.MousePos.y - origin.y);  // 鼠标点击位置减去canvas位置, 得到相对于canvas的位置

			// Add first and second point
			if (is_hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
			{
				data->points.push_back(mouse_pos_in_canvas);
				needReCalculate = true;
			}

			// Draw points 画点
			for (int n = 0; n < data->points.size(); n++)
				draw_list->AddCircleFilled(data->points[n] + origin, 5.0f, IM_COL32(255, 255, 0, 200));

			// Pan (we use a zero mouse threshold when there's no context menu)
			// You may decide to make that threshold dynamic based on whether the mouse is hovering something etc.
			const float mouse_threshold_for_pan = data->opt_enable_context_menu ? -1.0f : 0.0f;
			if (is_active && ImGui::IsMouseDragging(ImGuiMouseButton_Right, mouse_threshold_for_pan))
			{
				data->scrolling[0] += io.MouseDelta.x;
				data->scrolling[1] += io.MouseDelta.y;
			}

			// Context menu (under default mouse threshold)
			ImVec2 drag_delta = ImGui::GetMouseDragDelta(ImGuiMouseButton_Right);
			if (data->opt_enable_context_menu && ImGui::IsMouseReleased(ImGuiMouseButton_Right) && drag_delta.x == 0.0f && drag_delta.y == 0.0f)
				ImGui::OpenPopupContextItem("context");

			// 右键菜单
			if (ImGui::BeginPopup("context"))
			{
				if (ImGui::MenuItem("Remove one", NULL, false, data->points.size() > 0))
				{ 
					data->points.resize(data->points.size() - 1);
					needReCalculate = true;
				}
				if (ImGui::MenuItem("Remove all", NULL, false, data->points.size() > 0))
				{
					data->points.clear();
					needReCalculate = true;
				}
				ImGui::EndPopup();
			}

			// Draw grid + all lines in the canvas
			draw_list->PushClipRect(canvas_p0, canvas_p1, true);
			if (data->opt_enable_grid)
			{
				const float GRID_STEP = 64.0f;
				for (float x = fmodf(data->scrolling[0], GRID_STEP); x < canvas_sz.x; x += GRID_STEP)
					draw_list->AddLine(ImVec2(canvas_p0.x + x, canvas_p0.y), ImVec2(canvas_p0.x + x, canvas_p1.y), IM_COL32(200, 200, 200, 40));
				for (float y = fmodf(data->scrolling[1], GRID_STEP); y < canvas_sz.y; y += GRID_STEP)
					draw_list->AddLine(ImVec2(canvas_p0.x, canvas_p0.y + y), ImVec2(canvas_p1.x, canvas_p0.y + y), IM_COL32(200, 200, 200, 40));
			}
			draw_list->PopClipRect();

			// 计算与绘制
			// if add new points or remove points, 则需要更新要绘制的点
			if (needReCalculate && data->points.size() >= 2) {
				float minX = FLT_MAX;
				float maxX = -1;  // 坐标都是正的, 没有比-1小
				for (int i = 0; i < data->points.size(); i++)
				{
					if (data->points[i][0] < minX) { minX = data->points[i][0]; }
					if (data->points[i][0] > maxX) { maxX = data->points[i][0]; }
				}

				// 重新计算所有点的坐标, 并保存到容器中
				int wrapLength = 10; // 在最小和最大以外多画的点数
				int step = 1;  // 每step个像素之间画一条线
				data->lagrangeResults.clear();
				data->gaussResults.clear();
				data->leastSquaresResults.clear();
				data->ridgeRegressionResults.clear();
				for (float x = minX - wrapLength; x < maxX + wrapLength; x += step)
				{
					if (data->opt_lagrange)			
						data->lagrangeResults.push_back(ImVec2(x, lagrange(data->points, x)));
					if (data->opt_gauss)			
						data->gaussResults.push_back(ImVec2(x, gauss(data->points, x, data->gaussTheta)));
					if (data->opt_least_squares)	
						data->leastSquaresResults.push_back(ImVec2(x, leastSquares(data->points, x, data->leastSquaresM)));
					if (data->opt_ridge_regression) 
						data->ridgeRegressionResults.push_back(ImVec2(x, ridgeRegression(data->points, x, data->ridgeRegressionM, data->ridgeRegressionLambda)));
				}
				
				if (data->opt_equidistant_parameterization) {
					data->equidistantParameterizationResults.clear();
					equidistantParameterization(data->points, data->equidistantParameterizationResults);
				}

				if (data->opt_chordal_parameterization) {
					data->chordalParameterizationResults.clear();
					chordalParameterization(data->points, data->chordalParameterizationResults);
				}
			}

			if (data->points.size() >= 2)  // 少于两个点, 画线无意义
			{
				static float thickness = 2.0f;
				if (data->opt_lagrange)
					addLine(draw_list, origin, data->lagrangeResults, IM_COL32(64, 128, 255, 255), thickness);  // 画拉格朗日插值的线
				if (data->opt_gauss)
					addLine(draw_list, origin, data->gaussResults, IM_COL32(0, 255, 0, 255), thickness);  // 画高斯插值的线
				if (data->opt_least_squares) 
					addLine(draw_list, origin, data->leastSquaresResults, IM_COL32(255, 128, 128, 255), thickness);  // 画最小二乘拟合的线
				if (data->opt_ridge_regression) 
					addLine(draw_list, origin, data->ridgeRegressionResults, IM_COL32(255, 255, 0, 255), thickness);  // 画最岭回归拟合的线

				if (data->opt_equidistant_parameterization)
					addLine(draw_list, origin, data->equidistantParameterizationResults, IM_COL32(255, 0, 0, 255), thickness);
			}
		}

		ImGui::End();
	});
}
