%% MPC for Soil Water Balance (Initial Example)
% Simple Model - Simulated values
% Objective: Verify the initial behavior of the Model Predictive Control (MPC)
% This script simulates a soil moisture control system using CasADi for optimization.

% Add CasADi path (Update to your local path)
addpath('C:\Users\migue\OneDrive\Documentos\MATLAB\casadiprueba\casadi-3.7.2-windows64-matlab2018b');
clear; clc; close all;
import casadi.*

%% Simulation Notes:
% Tested with constant ETc and D (do not depend on theta or time).
% Precipitation removed to verify performance in a controlled environment without rain.
% Only irrigation control (u) affects the state variable theta (soil moisture).

%% -------------------------
% System Parameters
% --------------------------
Zr = 0.30;        % Root depth [m]
dt = 1.0;         % Time step [day]

ETc = 4.0;        % Evapotranspiration [mm/day]
D   = 0.5;        % Drainage [mm/day]

theta_ref = 0.25; % Target soil moisture (Set-point)

%% -------------------------
% MPC Horizon
% --------------------------
N = 10;           % Prediction horizon (steps)

%% -------------------------
% Symbolic Definition
% --------------------------
x = SX.sym('x');  % State: soil moisture [volumetric]
u = SX.sym('u');  % Control: irrigation [mm/day]

% Dynamic Model (Euler discretization)
% Formula: x(k+1) = x(k) + dt * (1/Zr) * (Inflow - Outflow)/1000
x_next = x + dt*(1/Zr)*((u - ETc - D)/1000);
f = Function('f',{x,u},{x_next});

%% -------------------------
% Optimization Variables
% --------------------------
X = SX.sym('X',N+1); % State trajectory
U = SX.sym('U',N);   % Control sequence

X0 = SX.sym('X0');   % Initial state (parameter for the solver)

%% -------------------------
% Objective Function and Constraints
% --------------------------
Q = 100;      % Weight for moisture error (State tracking)
R = 0.001;    % Weight for irrigation effort (Control effort)

J = 0;        % Cost function
g = [];       % Constraints vector

% Initial condition constraint
g = [g; X(1) - X0];

for k = 1:N
    % Stage Cost (Minimize error and control use)
    J = J + Q*(X(k) - theta_ref)^2 + R*U(k)^2;

    % Dynamic Constraints (System physics)
    g = [g; X(k+1) - f(X(k),U(k))];
end

%% -------------------------
% NLP Formulation
% --------------------------
OPT = struct('x',[X;U],'f',J,'g',g,'p',X0);

opts.ipopt.print_level = 0;
opts.print_time = false;

solver = nlpsol('solver','ipopt',OPT,opts);

%% -------------------------
% Bounds (Constraints)
% --------------------------
lbg = zeros(N+1,1); % Lower bound for g (Equality)
ubg = zeros(N+1,1); % Upper bound for g (Equality)

lbw = [ ...
    0.15*ones(N+1,1);   % Minimum theta (Wilting point proxy)
    0.0*ones(N,1)       % Minimum irrigation
];

ubw = [ ...
    0.35*ones(N+1,1);   % Maximum theta (Field capacity proxy)
    6.0*ones(N,1)       % Maximum irrigation
];

%% -------------------------
% Closed-loop Simulation
% --------------------------
Tsim = 30;             % Simulation duration (days)
xk = 0.18;             % Initial soil moisture

theta_hist = zeros(Tsim,1);
u_hist     = zeros(Tsim,1);

% Warm-start initialization for the horizon
w0 = [ ...
    xk*ones(N+1,1);
    zeros(N,1)
];

for t = 1:Tsim
    % Solve the MPC optimization problem
    sol = solver( ...
        'x0', w0, ...
        'lbx', lbw, ...
        'ubx', ubw, ...
        'lbg', lbg, ...
        'ubg', ubg, ...
        'p', xk ...
    );

    w_opt = full(sol.x);

    % Extract optimal control (first action)
    U_opt = w_opt(N+2:end);
    uk = U_opt(1);

    % Apply control to the plant (Simulation)
    xk = full(f(xk,uk));

    % Store results
    theta_hist(t) = xk;
    u_hist(t) = uk;

    % Receding Horizon: Shift initialization for next step
    w0 = [ ...
        w_opt(2:N+1);
        w_opt(N+1);
        U_opt(2:end);
        U_opt(end)
    ];
end

%% -------------------------
% Plotting Results
% --------------------------
time = 1:Tsim;
fs = 18;          % Main font size
fs_tit = fs + 4;  % Title font size
black = [0, 0, 0];
midBlue = [0, 0.4, 0.8];

figure('Color','w', 'Units', 'normalized', 'Position', [0.1 0.1 0.8 0.8]);

% --- Subplot 1: Soil Moisture ---
subplot(2,1,1)
plot(time, theta_hist, 'Color', midBlue, 'LineWidth', 3); hold on;

% Reference Line
yline(theta_ref, 'r--', 'Reference', 'LineWidth', 2, ...
    'FontSize', fs, 'FontWeight', 'bold', 'LabelVerticalAlignment', 'bottom');

% Operational Constraints
yline(0.15, 'k:', 'Minimum (0.15)', 'Alpha', 0.6, 'FontSize', fs-2, 'FontWeight', 'bold'); 
yline(0.35, 'k:', 'Maximum (0.35)', 'Alpha', 0.6, 'FontSize', fs-2, 'FontWeight', 'bold');

ylim([0.10, 0.40]); 
ylabel('$\theta$', 'Interpreter', 'latex', 'FontSize', fs+2, 'Color', black, 'FontWeight', 'bold');
title('Soil Moisture Level', 'FontSize', fs_tit, 'Color', black, 'FontWeight', 'bold');
grid on;

set(gca, 'FontSize', fs, 'Color', 'w', 'XColor', black, 'YColor', black, 'GridAlpha', 0.2, 'LineWidth', 1.5);

% --- Subplot 2: Control Action ---
subplot(2,1,2)
stairs(time, u_hist, 'Color', black, 'LineWidth', 3); hold on;

% Irrigation Limit
yline(6.0, 'r:', 'Irrigation Limit (6)', 'LineWidth', 2, 'FontSize', fs, 'FontWeight', 'bold');

ylim([-0.5, 8]);
ylabel('Irrigation (mm/day)', 'FontSize', fs, 'Color', black, 'FontWeight', 'bold');
xlabel('Time (days)', 'FontSize', fs, 'Color', black, 'FontWeight', 'bold');
title('MPC Control Action', 'FontSize', fs_tit, 'Color', black, 'FontWeight', 'bold');
grid on;

set(gca, 'FontSize', fs, 'Color', 'w', 'XColor', black, 'YColor', black, 'GridAlpha', 0.2, 'LineWidth', 1.5);
