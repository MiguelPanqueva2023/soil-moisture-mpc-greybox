%% NMPC for Soil Water Balance
% ETc dependent on soil moisture (theta)
% Initial implementation - conceptual validation
addpath('C:\Users\migue\OneDrive\Documentos\MATLAB\casadiprueba\casadi-3.7.2-windows64-matlab2018b');
clear; clc; close all;
import casadi.*

% ETc becomes dependent on theta (soil moisture). The corresponding term 
% in the original equation is refactored: ETc,pot(t) and K(θ).
% Ks(θ) is an auxiliary function that captures water stress; it is not 
% a new variable. It is used to refactor ETc(theta,t), where ETc,pot(t) 
% represents atmospheric demand and Ks(θ) models the reduction of that 
% demand based on soil moisture content.
% This reformulation allows introducing nonlinearity into the model in a 
% controlled manner and facilitates its integration within the Nonlinear 
% Predictive Control (NMPC) scheme.

%% -------------------------
% Physical Parameters
% -------------------------
Zr = 0.30;        % Root depth [m]
dt = 1.0;         % Time step [day]
D  = 0.5;         % Deep drainage [mm/day]
theta_fc = 0.30;  % Field capacity
theta_wp = 0.15;  % Wilting point
theta_ref = 0.25; % Target moisture

%% -------------------------
% MPC Horizon
% -------------------------
N = 10;

%% -------------------------
% Symbolic Variables
% -------------------------
x = SX.sym('x');      % Soil moisture
u = SX.sym('u');      % Irrigation
ETc = SX.sym('ETc');  % Potential evapotranspiration
P   = SX.sym('P');    % Precipitation

%% -------------------------
% Water Stress Factor Ks(theta)
% -------------------------
Ks = if_else( ...
    x >= theta_fc, 1, ...
    if_else(x <= theta_wp, 0, ...
    (x - theta_wp)/(theta_fc - theta_wp)));

%% -------------------------
% Dynamic Model (Euler)
% -------------------------
x_next = x + dt*(1/Zr)*(1/1000)*(u + P - ETc - D);
f = Function('f',{x,u,ETc,P},{x_next});

%% -------------------------
% Optimization Variables
% -------------------------
X = SX.sym('X',N+1);
U = SX.sym('U',N);
% Parameters: initial state + disturbances
X0 = SX.sym('X0');
ETc_h = SX.sym('ETc_h',N);
P_h   = SX.sym('P_h',N);
p = [X0; ETc_h; P_h];

%% -------------------------
% Objective Function and Constraints
% -------------------------
J = 0;
g = [];
g = [g; X(1) - X0];
for k = 1:N
    J = J + (X(k)-theta_ref)^2 + 0.01*U(k)^2;
    g = [g; X(k+1) - f(X(k),U(k),ETc_h(k),P_h(k))];
end

%% -------------------------
% NLP Formulation
% -------------------------
OPT = struct('x',[X;U],'f',J,'g',g,'p',p);
opts.ipopt.print_level = 0;
opts.print_time = false;
solver = nlpsol('solver','ipopt',OPT,opts);

%% -------------------------
% Bounds
% -------------------------
lbg = zeros(N+1,1);
ubg = zeros(N+1,1);
lbw = [0.15*ones(N+1,1); 0*ones(N,1)];
ubw = [0.35*ones(N+1,1); 6*ones(N,1)];

%% -------------------------
% Closed-loop Simulation
% -------------------------
Tsim = 30;
xk = 0.18;
theta_hist = zeros(Tsim,1);
u_hist = zeros(Tsim,1);
w0 = zeros((N+1)+N,1);
for t = 1:Tsim
    % Variable ETc (climatic demand)
    ETc_val = 4 + sin(2*pi*t/30);
    % Rain event
    if (t >= 10 && t <= 12) || (t >= 20 && t <= 22)
        P_val = 4;
    else
        P_val = 0;
    end
    ETc_vec = ETc_val*ones(N,1);
    P_vec   = P_val*ones(N,1);
    sol = solver( ...
        'x0', w0, ...
        'lbx', lbw, ...
        'ubx', ubw, ...
        'lbg', lbg, ...
        'ubg', ubg, ...
        'p', [xk; ETc_vec; P_vec] ...
    );
    w_opt = full(sol.x);
    U_opt = w_opt(N+2:end);
    uk = U_opt(1);
    xk = full(f(xk,uk,ETc_val,P_val));
    theta_hist(t) = xk;
    u_hist(t) = uk;
    w0 = [w_opt(2:N+1); w_opt(N+1); U_opt(2:end); U_opt(end)];
end

%% -------------------------
% Plots
% -------------------------
time = 1:Tsim;
fs = 18;          % Master font size (Large and readable)
fs_tit = fs + 4;  % Title size
black = [0, 0, 0];
midBlue = [0, 0.4, 0.8];
stdBlue   = [0, 0.447, 0.741];
stdRed   = [0.85, 0.325, 0.098];
% Create wide figure
figure('Color','w', 'Units', 'normalized', 'Position', [0.05 0.05 0.9 0.85]);

% --- Subplot 1: Soil Moisture (State) ---
subplot(3,1,1)
plot(time, theta_hist, 'Color', midBlue, 'LineWidth', 3); hold on;
% Reference Line
yline(theta_ref, 'r--', 'Reference', 'LineWidth', 2.5, 'FontSize', fs, 'FontWeight', 'bold');
% Moisture Constraints (lbw/ubw)
yline(0.15, 'k:', 'Min. (0.15)', 'Alpha', 0.7, 'FontSize', fs-2, 'FontWeight', 'bold'); 
yline(0.35, 'k:', 'Max. (0.35)', 'Alpha', 0.7, 'FontSize', fs-2, 'FontWeight', 'bold');
ylim([0.10, 0.40]); % Margin for readability
ylabel('\theta', 'FontSize', fs+4, 'Color', black, 'FontWeight', 'bold');
title('Soil Moisture (NMPC)', 'FontSize', fs_tit, 'Color', black, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', fs, 'Color', 'w', 'XColor', black, 'YColor', black, 'GridAlpha', 0.2, 'LineWidth', 1.5);

% --- Subplot 2: Control Action (Irrigation) ---
subplot(3,1,2)
stairs(time, u_hist, 'Color', black, 'LineWidth', 3); hold on;
% Maximum Irrigation Constraint
yline(6.0, 'r:', 'Irrigation Limit (6)', 'LineWidth', 2.5, 'FontSize', fs, 'FontWeight', 'bold');
ylim([-0.5, 8]); % Extra space above
ylabel('Irrigation (mm/day)', 'FontSize', fs, 'Color', black, 'FontWeight', 'bold');
title('MPC Control Action', 'FontSize', fs_tit, 'Color', black, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', fs, 'Color', 'w', 'XColor', black, 'YColor', black, 'GridAlpha', 0.2, 'LineWidth', 1.5);

% --- Subplot 3: Disturbances (Demand and Rain) ---
subplot(3,1,3)
% Re-calculate for consistent plotting
ETc_plot = 4 + sin(2*pi*time/30);
P_plot = zeros(size(time));
P_plot((time >= 10 & time <= 12) | (time >= 20 & time <= 22)) = 4;
plot(time, ETc_plot, 'Color', stdBlue, 'LineWidth', 3); hold on;
stairs(time, P_plot, 'Color', stdRed, 'LineWidth', 3);
% Legend with corrected format
lgd = legend('ETc(t)', 'Precipitation');
lgd.FontSize = fs;
lgd.TextColor = black;
lgd.Color = 'w';
lgd.EdgeColor = black;
ylim([-0.5, max(ETc_plot) + 4]); 
ylabel('mm/day', 'FontSize', fs, 'Color', black, 'FontWeight', 'bold');
xlabel('Time (days)', 'FontSize', fs, 'Color', black, 'FontWeight', 'bold');
title('Climatic Disturbances', 'FontSize', fs_tit, 'Color', black, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', fs, 'Color', 'w', 'XColor', black, 'YColor', black, 'GridAlpha', 0.2, 'LineWidth', 1.5);
