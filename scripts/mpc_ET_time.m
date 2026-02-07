%% MPC for Soil Water Balance
% Time-dependent ETc (Initial Test)
% ------------------------------------------------
% This version introduces:
%  - Precipitation as an external disturbance
%  - Time-varying Evapotranspiration (ETc)
%
% The model remains simple and explanatory
% Author: Miguel Panqueva (Academic Example)

addpath('C:\Users\migue\OneDrive\Documentos\MATLAB\casadiprueba\casadi-3.7.2-windows64-matlab2018b');
clear; clc; close all;
import casadi.*

% ETc becomes time-dependent (variable) using a sine function
% for smooth behavior. This influences irrigation:
% High ETc = More irrigation needed / Low ETc = Less irrigation needed.

%% -------------------------
% System Physical Parameters
% --------------------------
Zr = 0.30;        % Root depth [m]
dt = 1.0;         % Time step [day]
D = 0.5;          % Constant drainage [mm/day]
theta_ref = 0.25; % Target soil moisture (Set-point)

%% -------------------------
% MPC Horizon
% --------------------------
N = 10;           % Prediction horizon

%% -------------------------
% Symbolic Definition
% --------------------------
x = SX.sym('x');     % State: Soil moisture
u = SX.sym('u');     % Control: Irrigation
p  = SX.sym('p');    % Precipitation (Disturbance)
etc = SX.sym('etc'); % Evapotranspiration (Time-dependent)

% Discrete Dynamic Model (Euler)
% Simplified Water Balance Equation
x_next = x + dt*(1/Zr)*(u + p - etc - D);
f = Function('f',{x,u,p,etc},{x_next});

%% -------------------------
% Optimization Variables
% --------------------------
X = SX.sym('X',N+1);   % States over the horizon
U = SX.sym('U',N);     % Controls over the horizon
P = SX.sym('P',N);     % Precipitation over the horizon
ETC = SX.sym('ETC',N); % ETc over the horizon
X0 = SX.sym('X0');     % Initial state (Parameter)

%% -------------------------
% Objective Function and Constraints
% --------------------------
J = 0;
g = [];

% Initial condition constraint
g = [g; X(1) - X0];

for k = 1:N
    % Cost Function:
    % - Maintain theta close to the reference
    % - Penalize excessive irrigation effort
    J = J + (X(k)-theta_ref)^2 + 0.01*U(k)^2;
    
    % System Dynamics Constraints
    g = [g; X(k+1) - f(X(k),U(k),P(k),ETC(k))];
end

%% -------------------------
% NLP Formulation
% --------------------------
OPT = struct( ...
    'x',[X;U], ...
    'f',J, ...
    'g',g, ...
    'p',[X0; P; ETC] ...
);

opts.ipopt.print_level = 0;
opts.print_time = false;

solver = nlpsol('solver','ipopt',OPT,opts);

%% -------------------------
% Bounds (Constraints)
% --------------------------
lbg = zeros(N+1,1);
ubg = zeros(N+1,1);

lbw = [ ...
    0.15*ones(N+1,1);   % Minimum moisture
    0.0*ones(N,1)       % Minimum irrigation
];

ubw = [ ...
    0.35*ones(N+1,1);   % Maximum moisture
    6.0*ones(N,1)       % Maximum irrigation
];

%% -------------------------
% Closed-loop Simulation
% --------------------------
Tsim = 30;          % Simulation days
xk = 0.18;          % Initial moisture
theta_hist = zeros(Tsim,1);
u_hist = zeros(Tsim,1);

% Solver initialization
w0 = zeros((N+1)+N,1);

%% -------------------------
% Disturbance Definition
% --------------------------
% Simulated Precipitation (Intermittent rain events)
Psim = zeros(Tsim,1);
Psim(10:12) = 4;
Psim(20:22) = 3;

% Time-varying Evapotranspiration
% Smooth climatic variation (Sine wave)
t = (1:Tsim)';
ETc_sim = 4 + 1.5*sin(2*pi*t/30);

%% -------------------------
% MPC Simulation Loop
% --------------------------
for k = 1:Tsim
    % Extract disturbance horizon
    P_h = Psim(k:min(k+N-1,Tsim));
    ETc_h = ETc_sim(k:min(k+N-1,Tsim));
    
    % Padding if the end of the simulation is reached
    if length(P_h) < N
        P_h(end+1:N) = P_h(end);
        ETc_h(end+1:N) = ETc_h(end);
    end
    
    % Solve MPC optimization problem
    sol = solver( ...
        'x0', w0, ...
        'lbx', lbw, ...
        'ubx', ubw, ...
        'lbg', lbg, ...
        'ubg', ubg, ...
        'p', [xk; P_h(:); ETc_h(:)] ...
    );
    
    w_opt = full(sol.x);
    U_opt = w_opt(N+2:end);
    uk = U_opt(1);
    
    % Apply control action to the "real" system
    xk = full(f(xk,uk,Psim(k),ETc_sim(k)));
    
    % Store results
    theta_hist(k) = xk;
    u_hist(k) = uk;
    
    % Receding Horizon: Shift warm-start
    w0 = [w_opt(2:N+1); w_opt(N+1); U_opt(2:end); U_opt(end)];
end

%% -------------------------
% Plotting Results
% --------------------------
time = 1:Tsim;
fs = 18;          % Master font size (Large)
fs_tit = fs + 4;  % Title size (Extra Large)
black = [0, 0, 0];
midBlue = [0, 0.4, 0.8];
stdBlue   = [0, 0.447, 0.741];
stdRed   = [0.85, 0.325, 0.098];

% Create wide figure
figure('Color','w', 'Units', 'normalized', 'Position', [0.05 0.05 0.9 0.85]);

% --- Subplot 1: Soil Moisture ---
subplot(3,1,1)
plot(time, theta_hist, 'Color', midBlue, 'LineWidth', 3); hold on;

% Reference Line
yline(theta_ref, 'r--', 'Reference', 'LineWidth', 2.5, 'FontSize', fs, 'FontWeight', 'bold');

% Moisture Constraints (Based on lbw/ubw)
yline(0.15, 'k:', 'Min. (0.15)', 'Alpha', 0.7, 'FontSize', fs-2, 'FontWeight', 'bold'); 
yline(0.35, 'k:', 'Max. (0.35)', 'Alpha', 0.7, 'FontSize', fs-2, 'FontWeight', 'bold');

ylim([0.10, 0.40]); 
ylabel('\theta', 'FontSize', fs+4, 'Color', black, 'FontWeight', 'bold');
title('Soil Moisture Level', 'FontSize', fs_tit, 'Color', black, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', fs, 'Color', 'w', 'XColor', black, 'YColor', black, 'GridAlpha', 0.2, 'LineWidth', 1.5);

% --- Subplot 2: MPC Control Action ---
subplot(3,1,2)
stairs(time, u_hist, 'Color', black, 'LineWidth', 3); hold on;

% Maximum Irrigation Constraint
yline(6.0, 'r:', 'Irrigation Limit (6)', 'LineWidth', 2.5, 'FontSize', fs, 'FontWeight', 'bold');

ylim([-0.5, 8]); 
ylabel('Irrigation (mm/day)', 'FontSize', fs, 'Color', black, 'FontWeight', 'bold');
title('MPC Control Action', 'FontSize', fs_tit, 'Color', black, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', fs, 'Color', 'w', 'XColor', black, 'YColor', black, 'GridAlpha', 0.2, 'LineWidth', 1.5);

% --- Subplot 3: Climatic Disturbances ---
subplot(3,1,3)
plot(time, ETc_sim, 'Color', stdBlue, 'LineWidth', 3); hold on;
stairs(time, Psim, 'Color', stdRed, 'LineWidth', 3);

% Legend configuration
lgd = legend('ETc(t)', 'Precipitation');
lgd.FontSize = fs;
lgd.TextColor = black;
lgd.Color = 'w';
lgd.EdgeColor = black;

ylim([-0.5, max(ETc_sim) + 4]); 
ylabel('mm/day', 'FontSize', fs, 'Color', black, 'FontWeight', 'bold');
xlabel('Time (days)', 'FontSize', fs, 'Color', black, 'FontWeight', 'bold');
title('Climatic Disturbances', 'FontSize', fs_tit, 'Color', black, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', fs, 'Color', 'w', 'XColor', black, 'YColor', black, 'GridAlpha', 0.2, 'LineWidth', 1.5);
